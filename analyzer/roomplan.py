"""Turns a labelled 3D point cloud of a room into a block-level build plan.

Input: points (N,3) in any right-handed world frame (metres if `metric`), per-point RGB colours and
per-point semantic category names (see CATEGORIES). Output: the JSON-serialisable plan consumed by
the Fabric mod (see README / HouseBuilder.java).

Conventions of the plan (Minecraft axes, +x east, +y up, +z south):
  interior air spans x in [0,W), z in [0,D), y in [1,H]; the floor layer is y=0, the ceiling y=H+1;
  walls sit at x=-1 / x=W (west / east) and z=-1 / z=D (north / south).
  Wall grids are indexed grid[row][u] with row 0 = y 1, u along +x (north/south) or +z (west/east).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

from palette import DYE_PALETTE, WALL_COLOURS, WALL_PALETTE, WOOD_PALETTE, Palette, srgb_to_lab

# ADE20K (150 class) label -> our category. Anything unmapped is ignored ("other").
ADE_TO_CATEGORY = {
    "wall": "wall", "floor": "floor", "ceiling": "ceiling",
    "bed": "bed", "cradle": "bed",
    "windowpane": "window", "window": "window",
    "door": "door", "screen door": "door",
    "table": "table", "desk": "desk", "coffee table": "coffee_table", "pool table": "table",
    "counter": "counter", "countertop": "counter", "kitchen island": "counter", "bar": "counter",
    "cabinet": "cabinet", "chest of drawers": "cabinet", "buffet": "cabinet", "case": "cabinet",
    "wardrobe": "wardrobe", "bookcase": "bookshelf", "shelf": "shelf",
    "chair": "chair", "armchair": "chair", "swivel chair": "chair", "seat": "chair", "stool": "chair",
    "ottoman": "chair", "sofa": "sofa", "bench": "sofa",
    "lamp": "lamp", "light": "ceiling_light", "chandelier": "ceiling_light", "sconce": "sconce",
    "plant": "plant", "flower": "plant", "pot": "plant", "vase": "plant",
    "painting": "painting", "poster": "painting", "bulletin board": "painting",
    "rug": "rug", "television receiver": "tv", "screen": "tv", "crt screen": "tv", "monitor": "tv",
    "computer": "tv", "refrigerator": "fridge", "stove": "stove", "oven": "stove", "microwave": "microwave",
    "sink": "sink", "bathtub": "bathtub", "toilet": "toilet", "fireplace": "fireplace",
    "box": "storage", "basket": "storage", "barrel": "storage", "ashcan": "trash",
    "washer": "washer", "dishwasher": "washer", "curtain": "curtain", "blind": "curtain",
    "mirror": "mirror", "clock": "clock",
}
CATEGORIES = sorted(set(ADE_TO_CATEGORY.values()) | {"other"})
STRUCTURAL = {"wall", "floor", "ceiling"}
WALL_MOUNTED = {"window", "door", "painting", "sconce", "curtain", "mirror", "clock"}
# Things that never stand on the floor grid.
NO_FOOTPRINT = WALL_MOUNTED | {"ceiling_light", "rug"}
# Placement priority (earlier wins collisions).
PRIORITY = ["bed", "counter", "fridge", "stove", "sink", "bathtub", "toilet", "wardrobe", "bookshelf",
            "cabinet", "fireplace", "sofa", "desk", "table", "coffee_table", "shelf", "washer", "tv",
            "chair", "lamp", "storage", "trash", "microwave", "plant"]
# Max height in blocks per category (None = up to the ceiling).
MAX_HEIGHT = {"bed": 1, "table": 1, "desk": 1, "coffee_table": 1, "counter": 1, "chair": 1, "sofa": 1,
              "cabinet": 2, "storage": 1, "trash": 1, "washer": 1, "sink": 1, "bathtub": 1, "toilet": 1,
              "stove": 1, "microwave": 1, "plant": 1, "lamp": 2, "fireplace": 2, "tv": 2, "shelf": 3}
# Smallest believable cluster (points after 3 cm voxelisation) per category.
MIN_POINTS = {"window": 60, "door": 120, "bed": 200, "rug": 150, "painting": 40, "plant": 25,
              "lamp": 25, "ceiling_light": 15, "sconce": 10, "microwave": 30, "clock": 10}

SIDES = ("north", "south", "west", "east")
AUTO_SCALE = 1.25  # default blocks per metre
OPPOSITE = {"north": "south", "south": "north", "west": "east", "east": "west"}


# ----------------------------------------------------------------------------------- geometry

def fit_plane(pts: np.ndarray, threshold: float):
    """Robust plane fit (scikit-learn RANSAC in the points' own PCA basis). Returns (normal, point, inliers)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts[:: max(1, len(pts) // 20000)] - c, full_matrices=False)
    local = (pts - c) @ vt.T
    ransac = RANSACRegressor(residual_threshold=threshold, random_state=0, max_trials=200)
    ransac.fit(local[:, :2], local[:, 2])
    a, b = ransac.estimator_.coef_
    n_local = np.array([-a, -b, 1.0])
    n = n_local @ vt
    n /= np.linalg.norm(n)
    p0 = c + ransac.estimator_.intercept_ * vt[2]
    return n, p0, ransac.inlier_mask_


def estimate_normals(pts: np.ndarray, k: int = 16) -> np.ndarray:
    """Per-point normals from local PCA."""
    k = min(k, len(pts))
    _, idx = NearestNeighbors(n_neighbors=k).fit(pts).kneighbors(pts)
    nb = pts[idx] - pts[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def min_area_angle(xy: np.ndarray) -> float:
    """Orientation of the minimum-area bounding rectangle of 2D points (rotating calipers over the hull)."""
    from scipy.spatial import ConvexHull
    hull = xy[ConvexHull(xy).vertices]
    best, best_a = math.inf, 0.0
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = math.atan2(e[1], e[0])
        r = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        q = hull @ r.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best:
            best, best_a = area, a
    return best_a


def room_frame(points, cats):
    """Gravity-aligned, Manhattan-aligned frame. Returns (origin, R) with R rows = (x, up, z) axes."""
    is_ = lambda c: cats == c
    floor, ceil, wall = points[is_("floor")], points[is_("ceiling")], points[is_("wall")]
    if len(floor) >= 200:
        up, p0, _ = fit_plane(floor, 0.03)
    elif len(ceil) >= 200:
        up, p0, _ = fit_plane(ceil, 0.03)
    elif len(wall) >= 300:
        n = estimate_normals(wall)
        _, vecs = np.linalg.eigh(n.T @ n)
        up, p0 = vecs[:, 0], points.mean(0)
    else:
        up, p0 = np.array([0.0, -1.0, 0.0]), points.mean(0)  # OpenCV camera: y points down
    # Point "up" away from the floor (towards the bulk of the room) / away from the ceiling.
    ref = points[~is_("floor")].mean(0) if len(floor) >= 200 else None
    if ref is not None:
        up = up if (ref - p0) @ up > 0 else -up
    elif len(ceil) >= 200:
        up = up if (points.mean(0) - p0) @ up < 0 else -up
    elif up @ np.array([0.0, -1.0, 0.0]) < 0:
        up = -up

    # Any two horizontal basis vectors, then rotate them onto the dominant wall direction.
    t = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    b1 = np.cross(up, t); b1 /= np.linalg.norm(b1)
    b2 = np.cross(b1, up)
    angle = None
    if len(wall) >= 100:
        sub = wall[:: max(1, len(wall) // 30000)]
        n = estimate_normals(sub)
        h = np.stack([n @ b1, n @ b2], 1)
        hn = np.linalg.norm(h, axis=1)
        vertical = hn > 0.85
        if vertical.sum() >= 50:
            th = np.arctan2(h[vertical, 1], h[vertical, 0])
            angle = math.atan2(np.sin(4 * th).sum(), np.cos(4 * th).sum()) / 4
    if angle is None:
        rest = points[~is_("other")] if (cats != "other").sum() > 50 else points
        angle = min_area_angle(np.stack([rest @ b1, rest @ b2], 1))
    ex = math.cos(angle) * b1 + math.sin(angle) * b2
    ez = np.cross(ex, up)  # right-handed like Minecraft: x × y = z
    return p0, np.stack([ex, up, ez])


# ----------------------------------------------------------------------------------- surfaces

def _lab_dist(a, b, lw=0.75):
    d = np.asarray(a) - np.asarray(b)
    d[..., 0] *= lw
    return np.sqrt((d ** 2).sum(-1))


def quantize(cell_rgb: np.ndarray, weights: np.ndarray, palette: Palette, max_k=3, min_de=16.0, min_share=0.08):
    """Cluster surface cell colours into at most max_k clearly distinct materials -> (labels, block ids).

    Shading differences inside one material stay below `min_de`, so a uniformly painted wall becomes
    one block type instead of a blotchy patchwork.
    """
    lab = srgb_to_lab(cell_rgb)
    best = (np.zeros(len(lab), int), [np.average(cell_rgb, axis=0, weights=weights)])
    for k in range(2, max_k + 1):
        if len(lab) < k * 3:
            break
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab, sample_weight=weights)
        share = np.bincount(km.labels_, weights=weights, minlength=k) / weights.sum()
        centers = km.cluster_centers_
        ok = share.min() >= min_share and all(_lab_dist(centers[i], centers[j]) >= min_de
                                              for i in range(k) for j in range(i + 1, k))
        if ok:
            best = (km.labels_, [np.average(cell_rgb[km.labels_ == i], axis=0, weights=weights[km.labels_ == i])
                                 for i in range(k)])
    labels, means = best
    blocks = [palette.nearest(m) for m in means]
    # Merge clusters that snapped to the same block.
    uniq = list(dict.fromkeys(blocks))
    remap = np.array([uniq.index(b) for b in blocks])
    return remap[labels], uniq


def mode_filter(grid: np.ndarray, size=3, passes=2) -> np.ndarray:
    """Majority filter over observed cells (-1 = unknown) to remove speckle."""
    def mode(v):
        v = v[v >= 0].astype(int)
        return np.bincount(v).argmax() if len(v) else -1
    for _ in range(passes):
        filt = ndimage.generic_filter(grid, mode, size=size, mode="nearest")
        grid = np.where(grid >= 0, filt, grid)
    return grid


def fill_unknown(grid: np.ndarray, default: int) -> np.ndarray:
    """Infer unobserved cells from the nearest observed cell."""
    unknown = grid < 0
    if unknown.all():
        return np.full_like(grid, default)
    if not unknown.any():
        return grid
    _, (iy, ix) = ndimage.distance_transform_edt(unknown, return_indices=True)
    return grid[iy, ix]


def cell_colours(ij: np.ndarray, rgb: np.ndarray, shape):
    """Mean colour and point count per grid cell. ij: (N,2) integer cell indices."""
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    n = shape[0] * shape[1]
    cnt = np.bincount(flat, minlength=n)
    sums = np.stack([np.bincount(flat, weights=rgb[:, c].astype(float), minlength=n) for c in range(3)], 1)
    return sums / np.maximum(cnt, 1)[:, None], cnt


# ----------------------------------------------------------------------------------- objects

def axis_dir(dx: float, dz: float) -> str:
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


class Room:
    def __init__(self, xmin, xmax, zmin, zmax, height, scale):
        self.xmin, self.xmax, self.zmin, self.zmax, self.hm, self.s = xmin, xmax, zmin, zmax, height, scale
        self.W = max(3, round((xmax - xmin) * scale))
        self.D = max(3, round((zmax - zmin) * scale))
        self.H = max(3, round(height * scale))

    def bx(self, x):
        return np.clip(np.floor((np.asarray(x) - self.xmin) / (self.xmax - self.xmin) * self.W), 0, self.W - 1).astype(int)

    def bz(self, z):
        return np.clip(np.floor((np.asarray(z) - self.zmin) / (self.zmax - self.zmin) * self.D), 0, self.D - 1).astype(int)

    def by(self, y):  # interior row index, 0 = first air layer
        return np.clip(np.floor(np.asarray(y) / self.hm * self.H), 0, self.H - 1).astype(int)

    def wall_len(self, side):
        return self.W if side in ("north", "south") else self.D

    def wall_dist(self, x, z):
        return {"west": x - self.xmin, "east": self.xmax - x, "north": z - self.zmin, "south": self.zmax - z}

    def u_of(self, side, x, z):
        return self.bx(x) if side in ("north", "south") else self.bz(z)


def summarize_cluster(p, rgb):
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    return {"lo": lo, "hi": hi, "c": np.median(p, axis=0), "rgb": np.median(rgb, axis=0), "n": len(p)}


def cluster_category(p, eps=0.12, min_samples=8):
    if len(p) < min_samples:
        return []
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(p).labels_
    return [np.flatnonzero(lab == l) for l in set(lab) if l >= 0]


# ----------------------------------------------------------------------------------- main entry

def build_plan(points, colors, cats, surface_palette: Palette, scale=None, metric=True, log=print, ctx=None):
    """ctx: optional dict that receives the room frame (origin, R, y_shift, f, room) for texture projection."""
    points = np.asarray(points, float)
    colors = np.asarray(colors, float)
    cats = np.asarray(cats)
    origin, R = room_frame(points, cats)
    loc = (points - origin) @ R.T  # (x, y=height, z)

    floor_m = cats == "floor"
    if floor_m.sum() >= 50:
        y_shift = float(np.median(loc[floor_m, 1]))
    else:
        y_shift = float(np.percentile(loc[cats != "other", 1] if (cats != "other").any() else loc[:, 1], 1))
    loc[:, 1] -= y_shift
    f = 1.0

    # Remove gross outliers (things seen through windows, reconstruction noise).
    indoor = np.isin(cats, [c for c in CATEGORIES if c != "other"])
    ref = loc[indoor] if indoor.sum() > 100 else loc
    lo_q, hi_q = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    pad = 0.25 * (hi_q - lo_q)
    keep = np.all((loc >= lo_q - pad) & (loc <= hi_q + pad), axis=1)
    loc, colors, cats = loc[keep], colors[keep], cats[keep]
    floor_m, ceil_m, wall_m = cats == "floor", cats == "ceiling", cats == "wall"

    if not metric:
        # Fix the unknown global scale from size priors: doors ~2.03 m tall, ceilings ~2.5 m.
        door_h = np.percentile(loc[cats == "door", 1], 97) if (cats == "door").sum() > 100 else None
        ceil_h = np.median(loc[ceil_m, 1]) if ceil_m.sum() > 100 else None
        f = 2.03 / door_h if door_h else 2.5 / ceil_h if ceil_h else 1.0
        if f == 1.0:
            f = 3.5 / max(1e-6, np.percentile(loc[:, 0], 98) - np.percentile(loc[:, 0], 2))
        loc *= f
        log(f"[plan] non-metric reconstruction rescaled by {f:.3f}")

    x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
    wall_n = np.zeros((0, 3))
    if wall_m.sum() >= 30:
        wall_n = estimate_normals(loc[wall_m])

    def side_coord(axis, neg: bool):
        """Wall plane position for one side, or None when that wall was never seen."""
        if len(wall_n) == 0:
            return None
        wp = loc[wall_m]
        facing = np.abs(wall_n[:, axis]) > 0.85
        centre = np.median(loc[cats != "other", axis] if (cats != "other").any() else loc[:, axis])
        sel = facing & ((wp[:, axis] < centre) if neg else (wp[:, axis] > centre))
        if sel.sum() < max(40, 0.02 * len(wp)):
            return None
        return float(np.median(wp[sel, axis]))

    known_cats = cats != "other"
    def inferred(axis, neg):
        v = loc[known_cats & ~wall_m, axis] if (known_cats & ~wall_m).sum() > 50 else loc[:, axis]
        return float(np.percentile(v, 1 if neg else 99))

    bounds = {}
    for name, axis, neg in (("xmin", 0, True), ("xmax", 0, False), ("zmin", 2, True), ("zmax", 2, False)):
        c = side_coord(axis, neg)
        bounds[name] = c if c is not None else inferred(axis, neg)
        if c is None:
            log(f"[plan] wall {name} not visible - inferred from other geometry")
    for lo_k, hi_k in (("xmin", "xmax"), ("zmin", "zmax")):
        if bounds[hi_k] - bounds[lo_k] < 1.5:
            mid = (bounds[hi_k] + bounds[lo_k]) / 2
            bounds[lo_k], bounds[hi_k] = mid - 0.75, mid + 0.75
        if bounds[hi_k] - bounds[lo_k] > 100:
            mid = (bounds[hi_k] + bounds[lo_k]) / 2
            bounds[lo_k], bounds[hi_k] = mid - 50, mid + 50

    if ceil_m.sum() >= 50:
        hm = float(np.median(y[ceil_m]))
    elif wall_m.sum() >= 50 and np.percentile(y[wall_m], 98) > 2.2:
        hm = float(np.percentile(y[wall_m], 98))
    else:
        hm = 2.5
    hm = float(np.clip(hm, 2.0, 25.0))

    wm, dm = bounds["xmax"] - bounds["xmin"], bounds["zmax"] - bounds["zmin"]
    if scale is None:
        # 1.25 blocks per metre (rooms felt a bit small at 1:1), enlarged so the short side has at
        # least 6.25 blocks, at most 2.5 blocks per metre.
        scale = AUTO_SCALE
        if min(wm, dm) * scale < 5 * AUTO_SCALE:
            scale = min(2 * AUTO_SCALE, 5 * AUTO_SCALE / min(wm, dm))
    room = Room(bounds["xmin"], bounds["xmax"], bounds["zmin"], bounds["zmax"], hm, scale)
    if ctx is not None:
        ctx.update(origin=origin, R=R, y_shift=y_shift, f=f, room=room)
    W, D, H = room.W, room.D, room.H
    log(f"[plan] room {wm:.2f} x {dm:.2f} x {hm:.2f} m -> {W} x {D} x {H} blocks (scale {scale:.2f} blocks/m)")

    # ---------------- floor
    fl = floor_m & (x >= room.xmin) & (x <= room.xmax) & (z >= room.zmin) & (z <= room.zmax)
    floor_grid, floor_blocks = surface_grid(np.stack([room.bz(z[fl]), room.bx(x[fl])], 1), colors[fl], (D, W),
                                            surface_palette, default_rgb=(162, 130, 78))

    # ---------------- walls (clustered jointly so the same paint maps to the same block everywhere)
    near = 0.35
    wall_cells = {}
    # Only points whose surface actually faces along the side's axis, so corners do not bleed colours.
    faces_x, faces_z = np.zeros(len(loc), bool), np.zeros(len(loc), bool)
    if len(wall_n):
        faces_x[wall_m] = np.abs(wall_n[:, 0]) > 0.7
        faces_z[wall_m] = np.abs(wall_n[:, 2]) > 0.7
    for side in SIDES:
        if side == "north":
            sel = faces_z & (np.abs(z - room.zmin) < near); u = room.bx(x[sel])
        elif side == "south":
            sel = faces_z & (np.abs(z - room.zmax) < near); u = room.bx(x[sel])
        elif side == "west":
            sel = faces_x & (np.abs(x - room.xmin) < near); u = room.bz(z[sel])
        else:
            sel = faces_x & (np.abs(x - room.xmax) < near); u = room.bz(z[sel])
        shape = (H, room.wall_len(side))
        mean, cnt = cell_colours(np.stack([room.by(y[sel]), u], 1), colors[sel], shape)
        wall_cells[side] = (mean, cnt, shape)
    all_mean = np.concatenate([m[c > 0] for m, c, _ in wall_cells.values()])
    all_cnt = np.concatenate([c[c > 0] for _, c, _ in wall_cells.values()]).astype(float)
    if len(all_mean) == 0:
        all_mean, all_cnt = np.array([[225.0, 222, 215]]), np.array([1.0])
        log("[plan] no wall surface seen - defaulting to an off-white wall")
    # Walls: a second material only for a big, clearly different area (an accent wall, wainscoting).
    # Small things hanging on the wall are drawn by the photo-texture frames instead.
    labels, wall_blocks = quantize(all_mean, all_cnt, WALL_PALETTE, min_de=22.0, min_share=0.25)
    dominant = int(np.bincount(labels, weights=all_cnt).argmax())
    walls, k = {}, 0
    for side in SIDES:
        mean, cnt, shape = wall_cells[side]
        g = np.full(shape[0] * shape[1], -1)
        obs = cnt > 0
        g[obs] = labels[k:k + obs.sum()]
        k += obs.sum()
        g = g.reshape(shape)
        if (g >= 0).sum() < 3:
            g[:] = -1
            log(f"[plan] {side} wall surface unseen - using the dominant wall material")
        walls[side] = {"palette": wall_blocks, "grid": fill_unknown(mode_filter(g), dominant).tolist()}

    # ---------------- ceiling
    if ceil_m.sum() >= 30:
        ceiling_block = surface_palette.nearest(np.median(colors[ceil_m], axis=0))
    else:
        ceiling_block = surface_palette.nearest((235, 235, 230))

    # ---------------- objects
    occupancy = -np.ones((W, D), int)
    openings, objects = [], []
    obj_clusters = []
    for cat in CATEGORIES:
        if cat in STRUCTURAL or cat == "other":
            continue
        m = np.flatnonzero(cats == cat)
        for idx in cluster_category(loc[m]):
            if len(idx) < MIN_POINTS.get(cat, 60):
                continue
            obj_clusters.append((cat, summarize_cluster(loc[m][idx], colors[m][idx])))
    log("[plan] detected: " + ", ".join(f"{n}x {c}" for c, n in Counter(c for c, _ in obj_clusters).items()))

    def nearest_side(cx, cz):
        d = room.wall_dist(cx, cz)
        s = min(d, key=d.get)
        return s, d[s]

    def is_wall_tv(cat, cl):
        return cat == "tv" and nearest_side(cl["c"][0], cl["c"][2])[1] < 0.35

    # Wall openings & wall decorations first.
    for cat, cl in obj_clusters:
        if cat not in WALL_MOUNTED and not is_wall_tv(cat, cl):
            continue
        side, dist = nearest_side(cl["c"][0], cl["c"][2])
        if dist > 0.8 and cat not in ("window", "door"):
            continue
        lo, hi = cl["lo"], cl["hi"]
        u0, u1 = sorted((int(room.u_of(side, lo[0], lo[2])), int(room.u_of(side, hi[0], hi[2]))))
        y0, y1 = int(room.by(max(lo[1], 0))), int(room.by(min(hi[1], hm - 1e-3)))
        L = room.wall_len(side)
        entry = {"side": side, "u": u0, "w": max(1, u1 - u0 + 1)}
        if cat == "door":
            w = 2 if (hi - lo)[[0, 2]].max() * scale >= 1.6 else 1
            centre = (u0 + u1) // 2
            entry.update(type="door", u=int(np.clip(centre - (w - 1) // 2, 0, L - w)), w=w, y=0, h=2,
                         material=WOOD_PALETTE.nearest(cl["rgb"]), detected=True)
        elif cat == "window":
            y0 = max(1, y0) if H >= 4 else y0
            y1 = min(y1, H - 2) if H >= 4 else min(y1, H - 1)
            entry.update(type="window", y=min(y0, y1), h=max(1, y1 - y0 + 1), color=DYE_PALETTE.nearest(cl["rgb"]))
        elif cat in ("painting", "tv"):
            entry.update(type=cat, y=y0, h=max(1, y1 - y0 + 1))
        elif cat == "sconce":
            entry.update(type="sconce", y=min(H - 1, max(1, y1)), w=1, h=1)
        else:
            continue  # curtains / mirrors / clocks only inform us, they are not built
        entry["u"] = int(np.clip(entry["u"], 0, L - 1))
        entry["w"] = int(min(entry["w"], L - entry["u"]))
        openings.append(entry)

    # Reserve the space in front of every detected door so it stays usable.
    def door_front_cells(o):
        cells = []
        for du in range(o["w"]):
            u = o["u"] + du
            for depth in (0, 1):
                if o["side"] == "north": cells.append((u, depth))
                elif o["side"] == "south": cells.append((u, D - 1 - depth))
                elif o["side"] == "west": cells.append((depth, u))
                else: cells.append((W - 1 - depth, u))
        return [(a, b) for a, b in cells if 0 <= a < W and 0 <= b < D]

    for o in openings:
        if o["type"] == "door":
            for a, b in door_front_cells(o):
                occupancy[a, b] = -2

    tables = [cl for c, cl in obj_clusters if c in ("table", "desk", "coffee_table")]
    floor_objs = sorted([(c, cl) for c, cl in obj_clusters if c not in NO_FOOTPRINT and not is_wall_tv(c, cl)],
                        key=lambda t: (PRIORITY.index(t[0]) if t[0] in PRIORITY else 99, -t[1]["n"]))
    for cat, cl in floor_objs:
        if cat == "tv" and cl["lo"][1] > 0.4:
            # A screen on a desk or shelf (laptop, monitor) can't stand on it in the block grid, and on the
            # floor it would become a black pillar.
            log("[plan] screen resting on furniture, skipped")
            continue
        obj = place_object(cat, cl, room, occupancy, tables, nearest_side)
        if obj is None:
            log(f"[plan] no room for {cat}, skipped")
            continue
        objects.append(obj)

    for cat, cl in obj_clusters:
        if cat == "ceiling_light":
            objects.append({"type": "ceiling_light", "x": int(room.bx(cl["c"][0])), "z": int(room.bz(cl["c"][2])),
                            "w": 1, "d": 1, "h": 1})
        elif cat == "rug":
            x0, x1 = int(room.bx(cl["lo"][0])), int(room.bx(cl["hi"][0]))
            z0, z1 = int(room.bz(cl["lo"][2])), int(room.bz(cl["hi"][2]))
            objects.append({"type": "rug", "x": x0, "z": z0, "w": x1 - x0 + 1, "d": z1 - z0 + 1, "h": 1,
                            "color": DYE_PALETTE.nearest(cl["rgb"])})

    # ---------------- make sure the "house" has a usable front door
    if not any(o["type"] == "door" for o in openings):
        door = choose_door(room, openings, occupancy, objects)
        if door:
            door["material"] = "oak"
            openings.append(door)
            log(f"[plan] added front door on the {door['side']} wall at u={door['u']}")

    dom_lab = srgb_to_lab(WALL_COLOURS[wall_blocks[dominant]])
    roof_wood = "spruce" if dom_lab[0] < 40 else "dark_oak"
    return {
        "version": 1,
        "scale": round(float(scale), 3),
        "metres": {"x": round(wm, 2), "z": round(dm, 2), "y": round(hm, 2)},
        "size": {"x": W, "z": D, "y": H},
        "floor": {"palette": floor_blocks, "grid": floor_grid.tolist()},
        "walls": walls,
        "wall_main": wall_blocks[dominant],
        "ceiling": ceiling_block,
        "roof": {"wood": roof_wood, "axis": "x" if W >= D else "z"},
        "openings": openings,
        "objects": objects,
    }


def surface_grid(ij, rgb, shape, palette, default_rgb):
    mean, cnt = cell_colours(ij, rgb, shape) if len(ij) else (np.zeros((shape[0] * shape[1], 3)), np.zeros(shape[0] * shape[1]))
    obs = cnt > 0
    if obs.sum() == 0:
        return np.zeros(shape, int), [palette.nearest(default_rgb)]
    labels, blocks = quantize(mean[obs], cnt[obs].astype(float), palette)
    g = np.full(shape[0] * shape[1], -1)
    g[obs] = labels
    g = g.reshape(shape)
    dominant = int(np.bincount(labels, weights=cnt[obs]).argmax())
    return fill_unknown(mode_filter(g), dominant), blocks


def place_object(cat, cl, room: Room, occupancy, tables, nearest_side):
    W, D, H, s = room.W, room.D, room.H, room.s
    lo, hi, c = cl["lo"], cl["hi"], cl["c"]
    x0, x1 = int(room.bx(lo[0])), int(room.bx(hi[0]))
    z0, z1 = int(room.bz(lo[2])), int(room.bz(hi[2]))
    # Footprint from the real size (rounded), centred on the object's real position.
    w = int(np.clip(round((hi[0] - lo[0]) * s), 1, x1 - x0 + 1))
    d = int(np.clip(round((hi[2] - lo[2]) * s), 1, z1 - z0 + 1))
    x0 = int(round((c[0] - room.xmin) / (room.xmax - room.xmin) * W - w / 2))
    z0 = int(round((c[2] - room.zmin) / (room.zmax - room.zmin) * D - d / 2))
    h = max(1, int(round(hi[1] * s)))
    cap = MAX_HEIGHT.get(cat)
    h = min(h, cap if cap else H - 1, H - 1)
    side, dist = nearest_side(c[0], c[2])
    front = OPPOSITE[side]  # direction pointing from the nearest wall into the room
    obj = {"type": cat, "facing": front}

    if cat == "bed":
        long_x = (hi[0] - lo[0]) >= (hi[2] - lo[2])
        width_m = (hi[2] - lo[2]) if long_x else (hi[0] - lo[0])
        n_beds = int(np.clip(round(width_m * s), 1, 4))
        dists = room.wall_dist(c[0], c[2])
        if long_x:
            head = "west" if dists["west"] < dists["east"] else "east"
            w, d = 2, n_beds
            x0 = (0 if dists["west"] < 0.6 else int(room.bx(lo[0]))) if head == "west" else (W - 2 if dists["east"] < 0.6 else max(0, x1 - 1))
            z0 = int(np.clip(round(room.bz(c[2]) - (n_beds - 1) / 2), 0, D - n_beds))
        else:
            head = "north" if dists["north"] < dists["south"] else "south"
            w, d = n_beds, 2
            z0 = (0 if dists["north"] < 0.6 else int(room.bz(lo[2]))) if head == "north" else (D - 2 if dists["south"] < 0.6 else max(0, z1 - 1))
            x0 = int(np.clip(round(room.bx(c[0]) - (n_beds - 1) / 2), 0, W - n_beds))
        obj["facing"], h = head, 1
        obj["color"] = DYE_PALETTE.nearest(cl["rgb"])
    elif cat in ("chair", "sofa"):
        if dist < 0.8:
            obj["facing"] = side  # back against the wall
        elif tables:
            t = min(tables, key=lambda t: np.hypot(*(t["c"] - c)[[0, 2]]))
            obj["facing"] = axis_dir(c[0] - t["c"][0], c[2] - t["c"][2])  # back away from the table
        else:
            obj["facing"] = side
        obj["material"] = WOOD_PALETTE.nearest(cl["rgb"])
        if cat == "sofa":
            obj["color"] = DYE_PALETTE.nearest(cl["rgb"])  # upholstery
        if cat == "chair":
            w = d = 1
    elif cat in ("table", "desk", "coffee_table", "shelf", "bookshelf", "wardrobe", "cabinet", "counter"):
        obj["material"] = WOOD_PALETTE.nearest(cl["rgb"])
    elif cat in ("lamp", "plant", "trash", "toilet", "microwave", "fridge", "stove", "sink", "washer"):
        if cat in ("lamp", "plant", "trash", "toilet", "microwave"):
            w = d = 1
        if cat == "fridge":
            h = min(H - 1, max(2, h))
        if cat in ("fridge", "stove", "sink", "washer", "toilet") and w * d > 2:
            w, d = (min(w, 2), 1) if front in ("north", "south") else (1, min(d, 2))
    elif cat == "tv":
        w, d = (max(1, w), 1) if front in ("north", "south") else (1, max(1, d))
    w, d = max(1, min(w, W)), max(1, min(d, D))

    # Snap wall-hugging furniture onto the wall it is near (reconstruction noise leaves gaps).
    if dist < 0.6 and cat != "bed":
        if side == "west": x0 = 0
        elif side == "east": x0 = W - w
        elif side == "north": z0 = 0
        else: z0 = D - d
    x0, z0 = int(np.clip(x0, 0, W - w)), int(np.clip(z0, 0, D - d))

    # Find the closest free spot (spiral search up to 3 cells away).
    for r in range(0, 4):
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                ax, az = x0 + dx, z0 + dz
                if ax < 0 or az < 0 or ax + w > W or az + d > D:
                    continue
                if (occupancy[ax:ax + w, az:az + d] == -1).all():
                    occupancy[ax:ax + w, az:az + d] = 1
                    obj.update(x=int(ax), z=int(az), w=int(w), d=int(d), h=int(h))
                    return obj
    return None


def choose_door(room: Room, openings, occupancy, objects):
    """Pick an unblocked spot for a front door: free floor inside, no window, not in a corner."""
    W, D = room.W, room.D
    taken = {(o["side"], u) for o in openings for u in range(o["u"], o["u"] + o["w"]) if o["y"] <= 1}
    beds = [(o["x"] + o["w"] / 2, o["z"] + o["d"] / 2) for o in objects if o["type"] == "bed"]
    best, best_score = None, -math.inf
    for side in SIDES:
        L = room.wall_len(side)
        for u in range(1, L - 1) if L > 2 else range(L):
            if (side, u) in taken:
                continue
            inside = {"north": (u, 0), "south": (u, D - 1), "west": (0, u), "east": (W - 1, u)}[side]
            step = {"north": (u, 1), "south": (u, D - 2), "west": (1, u), "east": (W - 2, u)}[side]
            if occupancy[inside] != -1:
                continue
            score = L * 0.5 - abs(u - (L - 1) / 2)  # long walls, near the centre
            if 0 <= step[0] < W and 0 <= step[1] < D and occupancy[step] == -1:
                score += 3
            if beds:
                score += 0.3 * min(math.hypot(inside[0] - bx, inside[1] - bz) for bx, bz in beds)
            if score > best_score:
                best, best_score = {"type": "door", "side": side, "u": u, "w": 1, "y": 0, "h": 2, "detected": False}, score
    return best

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

from palette import CONCRETE, DYE_PALETTE, WALL_COLOURS, WALL_PALETTE, WOOD_PALETTE, Palette, srgb_to_lab

# ADE20K (150 class) label -> our category. Anything unmapped is ignored ("other").
ADE_TO_CATEGORY = {
    "wall": "wall", "floor": "floor", "ceiling": "ceiling",
    "bed": "bed", "cradle": "bed",
    "windowpane": "window", "window": "window",
    "door": "door", "screen door": "door",
    "table": "table", "desk": "desk", "coffee table": "coffee_table", "pool table": "table",
    "counter": "counter", "countertop": "counter", "kitchen island": "counter", "bar": "counter",
    "cabinet": "cabinet", "chest of drawers": "dresser", "buffet": "cabinet", "case": "cabinet",
    "nightstand": "nightstand", "dresser": "dresser",
    "wardrobe": "wardrobe", "bookcase": "bookshelf", "shelf": "shelf",
    "chair": "chair", "armchair": "chair", "swivel chair": "chair", "seat": "chair", "stool": "chair",
    "ottoman": "chair", "sofa": "sofa", "bench": "sofa",
    "lamp": "lamp", "light": "ceiling_light", "chandelier": "chandelier", "sconce": "sconce",
    "plant": "plant", "flower": "plant", "pot": "plant", "vase": "plant",
    "painting": "painting", "poster": "painting", "bulletin board": "bulletin_board",
    "rug": "rug", "television receiver": "tv", "screen": "tv", "crt screen": "monitor", "monitor": "monitor",
    "computer": "monitor", "refrigerator": "fridge", "stove": "stove", "oven": "stove", "microwave": "microwave",
    "sink": "sink", "bathtub": "bathtub", "toilet": "toilet", "fireplace": "fireplace",
    "box": "storage", "basket": "storage", "barrel": "storage", "ashcan": "trash",
    "washer": "washer", "dishwasher": "washer", "curtain": "curtain", "blind": "curtain",
    "mirror": "mirror", "clock": "clock", "fan": "fan", "radiator": "radiator", "piano": "piano",
    "hood": "range_hood",
}
# Templates only the open-vocabulary detector (perception.perceptron_pass) finds: ADE20K has no such class.
DETECTOR_ONLY = {"whiteboard", "projector", "exit_sign", "fire_extinguisher", "podium", "student_desk",
                 "projection_screen", "ceiling_fan", "chair_stack", "shower", "coat_rack", "wall_hooks",
                 "led_panel", "downlight", "linear_light", "pendant_lamp", "desk_lamp", "laptop",
                 "book_stack", "mug", "small_plant", "printer", "light_switch", "outlet", "thermostat",
                 "fire_alarm", "smoke_detector", "soap_dispenser", "sanitizer_dispenser", "paper_towel_dispenser"}
# Backpacks, globes, lockers, drinking fountains, umbrella stands.
ADE_TO_CATEGORY.update({"bag": "backpack", "fountain": "drinking_fountain"})
DETECTOR_ONLY |= {"globe", "locker", "umbrella_stand"}
CATEGORIES = sorted(set(ADE_TO_CATEGORY.values()) | DETECTOR_ONLY | {"other"})
# Detected things without a template are "generic:<detected name>" (see perception.perceptron_pass); they are
# built as a plain box and keep their name in the plan, so missing templates show up.
GENERIC = "generic:"
# Detected things that are a vanilla block: "vanilla:<block id>:<cube|model>:<detected name>" (perception.map_names).
VANILLA = "vanilla:"


def vanilla_parts(cat):
    """(block id, is full cube, detected name) of a vanilla category."""
    _, bid, kind, name = cat.split(":", 3)
    return bid, kind == "cube", name
STRUCTURAL = {"wall", "floor", "ceiling"}
TEMPLATES = set(CATEGORIES) - STRUCTURAL - {"other"}
# What each template is, for the detector's name mapping (perception.map_names_with_model).
TEMPLATE_INFO = {
    "bed": "a bed", "window": "a window, including stained glass", "door": "a door or doorway",
    "table": "a table: dining, reading or library table, or a long bench-like table people sit at",
    "desk": "a desk or writing table", "coffee_table": "a low table in front of a sofa",
    "counter": "a kitchen counter or island", "cabinet": "a cabinet", "dresser": "a chest of drawers",
    "nightstand": "a small bedside table with drawers", "podium": "a lecture podium or lectern",
    "student_desk": "a classroom chair with an attached writing arm", "chair_stack": "several chairs stacked together",
    "wardrobe": "a wardrobe or closet", "bookshelf": "a bookcase or shelving unit full of books",
    "shelf": "a single shelf board fixed to a wall, with books or things on it", "chair": "any chair, armchair or stool",
    "sofa": "a sofa, couch or upholstered bench", "lamp": "a floor or table lamp standing on something",
    "ceiling_light": "a small light fixed flat on the ceiling",
    "chandelier": "a chandelier, pendant or any light hanging down from the ceiling on chains or rods",
    "sconce": "a light fixed to a wall", "plant": "a potted plant or flowers", "painting": "a painting, poster, "
    "picture or poster on a wall", "bulletin_board": "a cork or bulletin board with pinned papers",
    "rug": "a rug or carpet on the floor", "tv": "a television", "monitor": "a computer monitor on a desk",
    "projection_screen": "a pull-down projection screen on a wall",
    "fridge": "a refrigerator", "stove": "a stove or oven", "microwave": "a microwave",
    "sink": "a sink", "bathtub": "a bathtub", "shower": "a glass shower stall",
    "toilet": "a toilet", "fireplace": "a fireplace", "piano": "a piano",
    "storage": "a box, basket or crate", "trash": "a trash can", "washer": "a washing machine or dishwasher",
    "curtain": "a curtain or blind", "mirror": "a mirror", "clock": "a wall clock",
    "radiator": "a wall radiator or heater", "range_hood": "a kitchen range or extractor hood above a stove",
    "coat_rack": "a freestanding coat rack", "wall_hooks": "coat hooks mounted on a wall",
    "whiteboard": "a whiteboard, dry-erase board or chalkboard on a wall",
    "projector": "a video projector, usually hanging from the ceiling",
    "exit_sign": "an exit sign, usually lit and mounted above a door",
    "fire_extinguisher": "a fire extinguisher or its wall cabinet",
    "fan": "a floor, tower or pedestal fan", "ceiling_fan": "a ceiling fan with blades",
    "led_panel": "a wide flat rectangular LED panel or fluorescent troffer in the ceiling",
    "downlight": "a small round recessed ceiling downlight", "linear_light": "a long narrow suspended ceiling light",
    "pendant_lamp": "a single hanging pendant lamp with a shade", "desk_lamp": "a small desk or table lamp",
    "laptop": "an open laptop computer on furniture", "book_stack": "a stack of books on furniture",
    "mug": "a mug or cup on furniture", "small_plant": "a small potted plant on furniture",
    "printer": "a desktop printer", "light_switch": "a wall light switch", "outlet": "a wall power outlet plate",
    "thermostat": "a wall thermostat", "fire_alarm": "a red fire alarm pull station or wall alarm device",
    "smoke_detector": "a round ceiling smoke detector", "soap_dispenser": "a wall soap dispenser",
    "sanitizer_dispenser": "a wall hand sanitizer dispenser",
    "paper_towel_dispenser": "a wall paper towel dispenser",
}
TEMPLATE_INFO.update({
    "backpack": "a backpack, school bag or handbag", "globe": "a globe of the Earth, on a desk or a floor stand",
    "locker": "a locker or a bank of lockers", "drinking_fountain": "a drinking fountain or water bottle filler",
    "umbrella_stand": "an umbrella stand",
})
assert set(TEMPLATE_INFO) == TEMPLATES, set(TEMPLATE_INFO) ^ TEMPLATES
WALL_MOUNTED = {"window", "door", "painting", "sconce", "curtain", "mirror", "clock", "whiteboard", "exit_sign",
                "fire_extinguisher", "bulletin_board", "projection_screen", "radiator", "range_hood", "wall_hooks",
                "light_switch", "outlet", "thermostat", "fire_alarm", "soap_dispenser",
                "sanitizer_dispenser", "paper_towel_dispenser"}
TABLETOP = {"monitor", "laptop", "book_stack", "mug", "small_plant", "printer", "desk_lamp"}
CEILING_FIXTURES = {"ceiling_light", "led_panel", "downlight", "linear_light", "pendant_lamp", "smoke_detector"}
# Things that never stand on the floor grid.
NO_FOOTPRINT = WALL_MOUNTED | CEILING_FIXTURES | {"chandelier", "ceiling_fan", "projector", "rug"} | TABLETOP
# Placement priority (earlier wins collisions).
PRIORITY = ["bed", "counter", "fridge", "stove", "sink", "bathtub", "shower", "toilet", "wardrobe", "bookshelf",
            "cabinet", "dresser", "nightstand", "piano", "podium", "fireplace", "sofa", "desk", "student_desk",
            "table", "coffee_table", "washer", "tv", "chair", "chair_stack", "coat_rack",
            "lamp", "fan", "storage", "trash", "microwave", "plant"]
# Max height in blocks per category (None = up to the ceiling).
MAX_HEIGHT = {"bed": 1, "table": 1, "desk": 1, "coffee_table": 1, "counter": 1, "chair": 1, "sofa": 1,
              "cabinet": 2, "storage": 1, "trash": 1, "washer": 1, "sink": 1, "bathtub": 1, "toilet": 1,
              "stove": 1, "microwave": 1, "plant": 1, "lamp": 2, "fireplace": 2, "tv": 2, "fan": 2,
              "podium": 2, "student_desk": 1, "chair_stack": 2, "nightstand": 1, "dresser": 2,
              "piano": 2, "shower": 2, "coat_rack": 2}
# Smallest believable cluster (points after 3 cm voxelisation) per category.
MIN_POINTS = {"window": 60, "door": 120, "bed": 200, "rug": 150, "painting": 40, "plant": 25,
              "lamp": 25, "ceiling_light": 15, "chandelier": 25, "sconce": 10, "microwave": 30, "clock": 10,
              "whiteboard": 40, "exit_sign": 8, "fire_extinguisher": 12, "projector": 12, "fan": 20,
              "bulletin_board": 35, "projection_screen": 40, "radiator": 30, "range_hood": 20,
              "mirror": 20, "curtain": 25, "ceiling_fan": 20, "monitor": 15, "wall_hooks": 12,
              **{c: 8 for c in ("led_panel", "downlight", "linear_light", "pendant_lamp", "smoke_detector",
                               "desk_lamp", "laptop", "book_stack", "mug", "small_plant", "printer",
                               "light_switch", "outlet", "thermostat", "fire_alarm", "soap_dispenser",
                               "sanitizer_dispenser", "paper_towel_dispenser")}}

WALL_MOUNTED |= {"drinking_fountain", "shelf"}
NO_FOOTPRINT |= {"drinking_fountain", "globe", "shelf"}   # a globe sits on its desk (or is placed on the floor below)
# Raised beds. A bed cluster with most of its points above RAISED_Y is a bunk or a loft bed: a loft when furniture
# stands under it (a desk, a dresser), else a bunk when part of the bed is also low. Thresholds are first guesses.
# RAISED_Y is above an ordinary bed's headboard (most stop below 1.2 m), so a bed box that includes a headboard (a
# RoomPlan box is sampled down its sides) does not read as raised; a bunk's or loft's upper deck is at about 1.3-1.8 m.
RAISED_Y, LOW_Y = 1.2, 0.7          # metres above the floor
RAISED_SHARE, LOW_SHARE = 0.35, 0.2
# Furniture that stands under a loft bed. Not chairs, nightstands or bins: they stand next to any bed.
UNDER_LOFT = {"desk", "table", "dresser", "cabinet", "wardrobe", "sofa", "storage", "fridge", "bookshelf",
              "student_desk"}
UNDER_SHARE = 0.5                   # of the furniture's own footprint that is under the bed
PAINTING_MAX_M = 1.2                # posters and paintings, per side
TALL_CABINET_M = 1.3                # a cabinet lower than this is built one block high
UPPER_BUNK = 3                      # block y of a raised bed: two blocks of headroom below it
PRIORITY += ["locker", "umbrella_stand", "backpack"]
MAX_HEIGHT.update({"locker": 2, "umbrella_stand": 1, "backpack": 1})
MIN_POINTS.update({"backpack": 20, "globe": 15, "locker": 60, "drinking_fountain": 15, "umbrella_stand": 12})

SIDES = ("north", "south", "west", "east")
AUTO_SCALE = 1.25  # default blocks per metre
ENLARGE = 1.25     # automatic scales are this much larger: one-block furniture leaves more room to move
                   # (tuned with the Apache MapAnything, the default; the research variant measures rooms ~20% larger)
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

def _is_whiteboard_colour(rgb) -> bool:
    """Bright and nearly colourless (CIELAB L > 70, chroma < 15): a whiteboard, not a switched-off screen."""
    L, a, b = srgb_to_lab(np.asarray(rgb, float)[None])[0]
    return L > 70 and math.hypot(a, b) < 15


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


DUP_OVERLAP = 0.5  # same-type clusters whose floor footprints overlap this much (of the smaller one) are one object


def footprint_overlap(a, b):
    """How much two clusters' floor (x, z) boxes overlap, as a share of the smaller one."""
    ix = min(a["hi"][0], b["hi"][0]) - max(a["lo"][0], b["lo"][0])
    iz = min(a["hi"][2], b["hi"][2]) - max(a["lo"][2], b["lo"][2])
    if ix <= 0 or iz <= 0:
        return 0.0
    area = lambda c: max(1e-6, (c["hi"][0] - c["lo"][0]) * (c["hi"][2] - c["lo"][2]))
    return ix * iz / min(area(a), area(b))


def merge_duplicates(obj_clusters, log=print):
    """One object seen in several photos whose depth disagreed shows up as overlapping copies: merge same-type
    clusters whose floor (x, z) footprints overlap by DUP_OVERLAP of the smaller one, averaging their boxes by
    point count. Side-by-side objects do not overlap and stay apart; wall-mounted things are left to the wall code.
    (The two mattresses of a bunk bed, when they come out as two clusters, merge too: their height shares average.)"""
    overlap = footprint_overlap

    out, merged = [], 0
    for cat in dict.fromkeys(c for c, _ in obj_clusters):
        cls = [cl for c, cl in obj_clusters if c == cat]
        if cat in WALL_MOUNTED or len(cls) == 1:
            out += [(cat, cl) for cl in cls]
            continue
        parent = list(range(len(cls)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                if overlap(cls[i], cls[j]) >= DUP_OVERLAP:
                    parent[root(j)] = root(i)
        groups = {}
        for i in range(len(cls)):
            groups.setdefault(root(i), []).append(cls[i])
        for g in groups.values():
            if len(g) == 1:
                out.append((cat, g[0]))
                continue
            w = np.array([c["n"] for c in g], float)
            avg = lambda k: np.average(np.stack([c[k] for c in g]), axis=0, weights=w)
            big = max(g, key=lambda c: c["n"])
            shares = {k: float(avg(k)) for k in ("raised", "low") if all(k in c for c in g)}
            out.append((cat, dict(big, lo=avg("lo"), hi=avg("hi"), c=avg("c"), rgb=avg("rgb"), n=int(w.sum()), **shares)))
            merged += len(g) - 1
    if merged:
        log(f"[plan] merged {merged} duplicate object(s) (one thing seen from photos whose depth disagreed)")
    return out


def summarize_cluster(p, rgb):
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    return {"lo": lo, "hi": hi, "c": np.median(p, axis=0), "rgb": np.median(rgb, axis=0), "n": len(p), **heights(p)}


def heights(p):
    """Shares of a cluster's points high up and low down (y is metres above the floor): tells raised beds apart."""
    return {"raised": float(np.mean(p[:, 1] > RAISED_Y)), "low": float(np.mean(p[:, 1] < LOW_Y))}


def cluster_category(p, eps=0.12, min_samples=8):
    if len(p) < min_samples:
        return []
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(p).labels_
    return [np.flatnonzero(lab == l) for l in set(lab) if l >= 0]


# ----------------------------------------------------------------------------------- main entry

def build_plan(points, colors, cats, surface_palette: Palette, scale=None, metric=True, log=print, ctx=None,
               known_height=None, object_instances=None, enlarge=ENLARGE, planes=None):
    """ctx: optional dict that receives the room frame (origin, R, y_shift, f, room) for texture projection.
    known_height: the room's real floor-to-ceiling height in metres; rescales the whole reconstruction to it.
    enlarge: factor on the automatic scale (an explicit `scale` is used exactly as given).
    planes: flat surfaces measured by ARKit, [{"normal", "point" (same world frame as `points`), "area", "label"}]:
    labelled walls set the wall positions and ceilings the room height, instead of the photos' guessed depth."""
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
    indoor = cats != "other"  # every known category, including generic ones
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
    # (Point spacing grows with every rescale here, so clustering radii are multiplied by f.)
    if known_height:
        # Monocular metric scale is a learned guess and can be far off outside ordinary rooms (a 20 m hall
        # came out at 0.4x). One real measurement fixes it everywhere, if the shape itself is right.
        ceil_y = np.median(loc[ceil_m, 1]) if ceil_m.sum() >= 50 else None
        if ceil_y and ceil_y > 0.5:
            stretch = known_height / ceil_y
            loc *= stretch
            f *= stretch
            log(f"[plan] measured ceiling {ceil_y:.2f} m, known {known_height:.2f} m: rescaled by {stretch:.3f}")
        else:
            log("[plan] --known-height ignored: too little ceiling seen to measure it")

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
        # A reconstruction can repeat the same wall behind its real surface. Find
        # supported parallel layers, then take the one facing the room interior.
        pts = wp[sel]
        values = pts[:, axis]
        tangent = pts[:, 2 if axis == 0 else 0]
        span = np.percentile(loc[:, axis], 99) - np.percentile(loc[:, axis], 1)
        edge = np.percentile(values, 2 if neg else 98)
        band = max(1.2, 0.3 * span)
        eligible = (values <= edge + band) if neg else (values >= edge - band)
        values, tangent, heights = values[eligible], tangent[eligible], pts[eligible, 1]
        if len(values) < 40:
            return float(np.median(pts[:, axis]))
        width = 0.08
        edges = np.arange(values.min() - width, values.max() + 2 * width, width)
        counts, _ = np.histogram(values, bins=edges)
        smooth = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
        peaks = np.flatnonzero((smooth >= np.r_[smooth[0], smooth[:-1]]) &
                                   (smooth >= np.r_[smooth[1:], smooth[-1]]))
        peaks = sorted(peaks, reverse=neg)
        min_support = max(40, 0.12 * float(smooth.max()))
        for peak in peaks:
            centre_layer = (edges[peak] + edges[peak + 1]) / 2
            support = np.abs(values - centre_layer) < 0.14
            if support.sum() < min_support:
                continue
            if np.ptp(np.percentile(heights[support], [10, 90])) < 0.6:
                continue
            if np.ptp(np.percentile(tangent[support], [10, 90])) < 0.2 * max(0.5, np.ptp(np.percentile(tangent, [10, 90]))):
                continue
            return float(np.median(values[support]))
        return float(np.median(values))

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
    # Measured walls beat walls from guessed depth: each side takes the area-weighted median of the ARKit planes
    # labelled wall that face along its axis.
    meas = [(np.asarray(p["normal"], float) @ R.T,
             ((np.asarray(p["point"], float) - origin) @ R.T - np.array([0, y_shift, 0])) * f,
             float(p["area"]), p["label"]) for p in planes or []]
    for name, axis, neg in (("xmin", 0, True), ("xmax", 0, False), ("zmin", 2, True), ("zmax", 2, False)):
        mid = (bounds[name[0] + "min"] + bounds[name[0] + "max"]) / 2
        hits = [(q[axis], a) for n, q, a, lab in meas if lab == "wall" and abs(n[axis]) > 0.9 and (q[axis] < mid) == neg]
        if sum(a for _, a in hits) >= 0.5:
            c, a = np.array(hits).T
            order = np.argsort(c)
            bounds[name] = float(c[order][np.searchsorted(np.cumsum(a[order]), a.sum() / 2)])
            log(f"[plan] wall {name} from {len(hits)} measured ARKit plane(s)")
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
    ceilings = [(q[1], a) for n, q, a, lab in meas if lab == "ceiling" and abs(n[1]) > 0.9 and q[1] > 1.5]
    if sum(a for _, a in ceilings) >= 0.5:
        hm = float(np.average([c for c, _ in ceilings], weights=[a for _, a in ceilings]))
        log(f"[plan] ceiling height {hm:.2f} m from {len(ceilings)} measured ARKit plane(s)")
    hm = float(np.clip(hm, 2.0, 40.0 if known_height else 25.0))

    wm, dm = bounds["xmax"] - bounds["xmin"], bounds["zmax"] - bounds["zmin"]
    if scale is None:
        # 1.25 blocks per metre (rooms felt a bit small at 1:1), enlarged so the short side has at
        # least 6.25 blocks, at most 2.5 blocks per metre.
        scale = AUTO_SCALE
        if min(wm, dm) * scale < 5 * AUTO_SCALE:
            scale = min(2 * AUTO_SCALE, 5 * AUTO_SCALE / min(wm, dm))
        # Furniture is whole blocks (a chair is one), so at true proportions rooms feel crowded: build bigger.
        scale *= enlarge
    room = Room(bounds["xmin"], bounds["xmax"], bounds["zmin"], bounds["zmax"], hm, scale)
    if ctx is not None:
        ctx.update(origin=origin, R=R, y_shift=y_shift, f=f, room=room)
    W, D, H = room.W, room.D, room.H
    log(f"[plan] room {wm:.2f} x {dm:.2f} x {hm:.2f} m -> {W} x {D} x {H} blocks (scale {scale:.2f} blocks/m)")

    # ---------------- floor
    fl = floor_m & (x >= room.xmin) & (x <= room.xmax) & (z >= room.zmin) & (z <= room.zmax)
    floor_grid, floor_blocks = surface_grid(np.stack([room.bz(z[fl]), room.bx(x[fl])], 1), colors[fl], (D, W),
                                            surface_palette, default_rgb=(162, 130, 78))
    # Preserve alcoves and angled edges where the photographed floor supplies enough coverage.
    footprint = np.ones((D, W), dtype=bool)
    if fl.sum() >= 100 and min(W, D) >= 4:
        counts = np.zeros((D, W), dtype=np.int32)
        np.add.at(counts, (np.clip(room.bz(z[fl]), 0, D - 1),
                           np.clip(room.bx(x[fl]), 0, W - 1)), 1)
        seen = counts >= max(3, int(np.percentile(counts[counts > 0], 20)) if np.any(counts) else 3)
        if seen.mean() >= 0.3:
            seen = ndimage.binary_closing(seen, structure=np.ones((3, 3)), border_value=1)
            seen = ndimage.binary_fill_holes(seen)
            labels, ncomp = ndimage.label(seen)
            if ncomp:
                seen = labels == (1 + np.argmax(np.bincount(labels.ravel())[1:]))
            if seen.mean() >= 0.3 and seen.mean() < 0.97:
                # Missing floor pixels often mean a bed or poor camera coverage.
                # Cut out an alcove only when a vertical wall supports its edge.
                wp = loc[wall_m]
                supported = total_edges = 0
                for row in range(D):
                    for col in range(W):
                        for dr, dc in ((0, 1), (1, 0)):
                            rr, cc = row + dr, col + dc
                            if rr >= D or cc >= W or seen[row, col] == seen[rr, cc]:
                                continue
                            total_edges += 1
                            if dc:
                                edge = room.xmin + (col + 1) / scale
                                middle = room.zmin + (row + 0.5) / scale
                                hit = (np.abs(wp[:, 0] - edge) < 0.12) & (np.abs(wp[:, 2] - middle) < 0.5 / scale)
                            else:
                                edge = room.zmin + (row + 1) / scale
                                middle = room.xmin + (col + 0.5) / scale
                                hit = (np.abs(wp[:, 2] - edge) < 0.12) & (np.abs(wp[:, 0] - middle) < 0.5 / scale)
                            if hit.sum() >= 30 and np.ptp(np.percentile(wp[hit, 1], [10, 90])) > 0.8:
                                supported += 1
                if total_edges >= 2 and supported >= 2 and supported / total_edges >= 0.5:
                    footprint = seen
                    log(f"[plan] traced non-rectangular floor: {int(seen.sum())}/{W * D} cells")
                else:
                    log(f"[plan] kept flat wall outline: only {supported}/{total_edges} floor cut edges have wall support")

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
    for cat in CATEGORIES + sorted(c for c in set(cats.tolist()) if c.startswith((GENERIC, VANILLA))):
        if cat in STRUCTURAL or cat == "other":
            continue
        m = np.flatnonzero(cats == cat)
        # Chandeliers are open metalwork alone in the air: a wider radius keeps one from splitting into parts.
        for idx in cluster_category(loc[m], eps=(0.35 if cat in ("chandelier", "ceiling_fan", "linear_light", "led_panel") else 0.12) * f):
            small = cat.startswith(VANILLA) and not vanilla_parts(cat)[1]  # a lantern or candle is small
            if len(idx) < MIN_POINTS.get(cat, 15 if small else 60):
                continue
            obj_clusters.append((cat, summarize_cluster(loc[m][idx], colors[m][idx])))
    matched = 0
    used_clusters = set()
    for inst in object_instances or []:
        p = (np.asarray(inst["points"], float) - origin) @ R.T
        p[:, 1] -= y_shift
        p *= f
        center = (np.asarray(inst["center"], float) - origin) @ R.T
        center[1] -= y_shift
        center *= f
        if len(p) < 20 or not np.isfinite(p).all() or not np.isfinite(center).all():
            continue
        cl = {"lo": np.percentile(p, 5, axis=0), "hi": np.percentile(p, 95, axis=0),
              "c": center, "rgb": np.median(inst["colors"], axis=0), "n": len(p),
              "source_views": inst["views"], **heights(p)}
        candidates = [(np.linalg.norm(old["c"] - center), j) for j, (cat, old) in enumerate(obj_clusters)
                      if cat == inst["category"] and j not in used_clusters]
        if candidates:
            dist, j = min(candidates)
            if dist < 0.8 * f:
                obj_clusters[j] = (inst["category"], cl)
                used_clusters.add(j)
                matched += 1
                continue
        obj_clusters.append((inst["category"], cl))
        matched += 1
    if matched:
        log(f"[plan] {matched} object(s) located from matching detections in multiple views")
    obj_clusters = merge_duplicates(obj_clusters, log)
    log("[plan] detected: " + ", ".join(f"{n}x {c}" for c, n in Counter(c for c, _ in obj_clusters).items()))

    # Generic segmentation light labels are refined from shape and position.
    refined = []
    for cat, cl in obj_clusters:
        span = cl["hi"] - cl["lo"]
        if cat == "ceiling_light":
            width = max(span[0], span[2])
            if width > 0.8 and min(span[0], span[2]) > 0.35:
                cat = "led_panel"
            elif width > 0.7 and min(span[0], span[2]) < 0.35:
                cat = "linear_light"
            elif cl["hi"][1] < hm - 0.25:
                cat = "pendant_lamp"
            else:
                cat = "downlight"
        elif cat == "lamp" and cl["lo"][1] > 0.55:
            cat = "desk_lamp"
        elif cat == "plant" and cl["lo"][1] > 0.55:
            cat = "small_plant"
        elif cat == "shelf" and cl["lo"][1] < 0.3 and cl["hi"][1] - cl["lo"][1] > 0.9:
            cat = "bookshelf"  # shelving standing on the floor, not one board on a wall (SegFormer calls both "shelf")
        refined.append((cat, cl))
    obj_clusters = refined

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
        elif cat == "tv" and _is_whiteboard_colour(cl["rgb"]):
            # ADE20K has no whiteboard: SegFormer calls one a "screen". A TV that is off is dark; a bright, colourless
            # "screen" on the wall is a whiteboard or projection screen.
            width_m = max(0.1, (hi - lo)[0 if side in ("north", "south") else 2])
            kind = "projection_screen" if (hi[1] - lo[1]) / width_m > 0.55 else "whiteboard"
            entry.update(type=kind, y=y0, h=max(1, y1 - y0 + 1), color="white")
        elif cat == "painting":
            # A poster's mask spreads past its edges (and runs into neighbouring frames): at most PAINTING_MAX_M a
            # side, around the middle of what was seen.
            cap = max(1, int(round(PAINTING_MAX_M * scale)))
            w, h = min(entry["w"], cap), min(y1 - y0 + 1, cap)
            entry.update(type="painting", u=u0 + (entry["w"] - w) // 2, w=w, y=y0 + (y1 - y0 + 1 - h) // 2, h=max(1, h))
        elif cat == "tv":
            entry.update(type=cat, y=y0, h=max(1, y1 - y0 + 1))
        elif cat == "sconce":
            entry.update(type="sconce", y=min(H - 1, max(1, y1)), w=1, h=1)
        elif cat in ("whiteboard", "bulletin_board", "projection_screen", "curtain", "mirror", "clock",
                     "radiator", "range_hood", "wall_hooks", "light_switch", "outlet", "thermostat",
                     "fire_alarm", "soap_dispenser", "sanitizer_dispenser", "paper_towel_dispenser"):
            # Its real extent; a white board or a green/black chalkboard.
            entry.update(type=cat, y=y0, h=max(1, y1 - y0 + 1), color=DYE_PALETTE.nearest(cl["rgb"]))
            if cat in ("clock", "wall_hooks", "range_hood", "light_switch", "outlet", "thermostat",
                       "fire_alarm", "soap_dispenser", "sanitizer_dispenser", "paper_towel_dispenser"):
                entry.update(w=1, u=(u0 + u1) // 2, h=1)
        elif cat == "exit_sign":
            # Mounted high, above a door (NFPA 101 7.10.1.9 caps it at 6 ft 8 in above the opening).
            entry.update(type="exit_sign", y=min(H - 1, max(1, y1)), w=1, h=1)
        elif cat == "fire_extinguisher":
            # Low on the wall (NFPA 10: top at most 5 ft, bottom at least 4 in above the floor).
            entry.update(type="fire_extinguisher", y=min(y0, max(0, H - 2)), w=1, h=1)
        elif cat == "drinking_fountain":
            # On the wall, its basin at counter height: the bottom block row.
            entry.update(type="drinking_fountain", y=0, w=1, h=1)
        elif cat == "shelf":
            # One board with a row of books, in the block row its bottom was seen in (never on the floor row).
            entry.update(type="shelf", y=int(np.clip(y0, 1, max(1, H - 2))), h=1, material=WOOD_PALETTE.nearest(cl["rgb"]))
        else:
            continue
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
    lofts = []  # opened up for the furniture under them once every bed is placed (else a second bed could go there)
    all_lofts = []
    for cat, cl in floor_objs:
        if cat != "bed":
            for o in lofts:
                occupancy[o["x"]:o["x"] + o["w"], o["z"]:o["z"] + o["d"]] = -1
            lofts = []
        # A detector sometimes labels an entire vertical stack as one ordinary chair.
        if cat == "chair" and cl["hi"][1] - cl["lo"][1] > 1.25 and max(cl["hi"][0] - cl["lo"][0],
                                                                  cl["hi"][2] - cl["lo"][2]) < 1.3:
            cat = "chair_stack"
        if cat == "tv" and cl["lo"][1] > 0.4:
            continue
        if cat.startswith(GENERIC) and cl["lo"][1] > 0.4:
            log(f"[plan] {cat[len(GENERIC):]} is not standing on the floor, skipped")
            continue
        if cat.startswith(VANILLA) and cl["lo"][1] > 0.4 and not vanilla_parts(cat)[1]:
            # (Full cubes - bookcases, barrels - always stand on the floor; tables and chairs in front often
            # hide their lower part, so they are placed from the floor up like any furniture.)
            bid, cube, name = vanilla_parts(cat)
            # A single block in the air (a lantern, a bell): it hangs from its top, at its real position.
            objects.append({"type": "vanilla", "block": "minecraft:" + bid, "name": name, "cube": False, "hanging": True,
                            "x": int(room.bx(cl["c"][0])), "z": int(room.bz(cl["c"][2])), "w": 1, "d": 1, "h": 1,
                            "y": int(room.by(min(cl["hi"][1], hm - 1e-3))) + 1})
            continue
        obj = place_object(cat, cl, room, occupancy, tables, nearest_side)
        if obj is None:
            log(f"[plan] no room for {cat}, skipped")
            continue
        if cat == "bed":
            style = bed_style(cl, obj_clusters, H)
            log(f"[plan] bed at x={obj['x']}, z={obj['z']}: {cl.get('raised', 0):.0%} of it above {RAISED_Y} m, "
                f"{cl.get('low', 0):.0%} below {LOW_Y} m -> {style or 'ordinary'}")
            if style:
                obj.update(style=style, upper=UPPER_BUNK)
                if style == "loft":
                    lofts.append(obj)  # a desk may go under it
                    all_lofts.append(obj)
                for a, b in ladder_cells(obj):
                    if 0 <= a < W and 0 <= b < D and occupancy[a, b] == -1:
                        occupancy[a, b] = 1  # the way up stays clear
        if cat != "bed" and any(obj["x"] < o["x"] + o["w"] and o["x"] < obj["x"] + obj["w"] and
                                obj["z"] < o["z"] + o["d"] and o["z"] < obj["z"] + obj["d"] for o in all_lofts):
            obj["h"] = min(obj["h"], UPPER_BUNK - 1)  # under a loft bed: below its deck
        if "source_views" in cl:
            obj["source_views"] = cl["source_views"]
            obj["estimated_center_m"] = [round(float(x), 3) for x in cl["c"]]
        objects.append(obj)

    # Small objects share the supporting furniture's floor cell, above its top.
    for cat, cl in obj_clusters:
        if cat not in TABLETOP | {"tv"} or (cat == "tv" and (is_wall_tv(cat, cl) or cl["lo"][1] <= 0.4)):
            continue
        x, z = int(room.bx(cl["c"][0])), int(room.bz(cl["c"][2]))
        support = next((o for o in objects if o["type"] in ("desk", "table", "counter", "student_desk")
                        and o["x"] <= x < o["x"] + o["w"] and o["z"] <= z < o["z"] + o["d"]), None)
        if support:
            objects.append({"type": "monitor" if cat == "tv" else cat, "x": x, "z": z, "w": 1, "d": 1,
                            "h": 1, "y": 2, "facing": support.get("facing", "south"),
                            "color": DYE_PALETTE.nearest(cl["rgb"])})

    # A globe stands on the desk or table it was seen on, or on the floor (a floor-stand globe).
    for cat, cl in obj_clusters:
        if cat != "globe":
            continue
        x, z = int(room.bx(cl["c"][0])), int(room.bz(cl["c"][2]))
        support = next((o for o in objects if o["type"] in ("desk", "table", "counter", "student_desk", "cabinet",
                                                             "dresser", "nightstand")
                        and o["x"] <= x < o["x"] + o["w"] and o["z"] <= z < o["z"] + o["d"]), None)
        if support:
            objects.append({"type": "globe", "x": x, "z": z, "w": 1, "d": 1, "h": 1, "y": 1 + support["h"],
                            "facing": support.get("facing", "south")})
        elif cl["lo"][1] <= 0.4:
            obj = place_object("globe", cl, room, occupancy, tables, nearest_side)
            if obj:
                obj["y"] = 1
                objects.append(obj)
    screens = [o for o in openings if o["type"] in ("whiteboard", "tv")]
    for cat, cl in obj_clusters:
        if cat == "projector":
            # Hangs from the ceiling on a rod and points at the whiteboard or screen wall, if there is one.
            cx, cz = int(room.bx(cl["c"][0])), int(room.bz(cl["c"][2]))
            facing = screens[0]["side"] if screens else OPPOSITE[nearest_side(cl["c"][0], cl["c"][2])[0]]
            objects.append({"type": "projector", "x": cx, "z": cz, "w": 1, "d": 1, "h": 1, "facing": facing,
                            "y": int(np.clip(int(room.by(min(cl["hi"][1], hm - 1e-3))) + 1, 1, H))})
        elif cat in ("chandelier", "ceiling_fan", "linear_light", "pendant_lamp"):
            # Footprint and hanging height from its real extent; the mod hangs it from the ceiling on a chain.
            x0, x1 = int(room.bx(cl["lo"][0])), int(room.bx(cl["hi"][0]))
            z0, z1 = int(room.bz(cl["lo"][2])), int(room.bz(cl["hi"][2]))
            w, d = int(np.clip(x1 - x0 + 1, 1, 4)), int(np.clip(z1 - z0 + 1, 1, 4))
            if cat == "pendant_lamp": w = d = 1
            cx, cz = int(room.bx(cl["c"][0])), int(room.bz(cl["c"][2]))
            y0, y1 = int(room.by(max(cl["lo"][1], 0))) + 1, int(room.by(min(cl["hi"][1], hm - 1e-3))) + 1
            h = int(np.clip(y1 - y0 + 1, 1, 3))
            objects.append({"type": cat, "x": int(np.clip(cx - (w - 1) // 2, 0, W - w)),
                            "z": int(np.clip(cz - (d - 1) // 2, 0, D - d)), "w": w, "d": d,
                            "y": int(np.clip(y1 - h + 1, 1, H)), "h": h})
        elif cat in CEILING_FIXTURES:
            x0, x1 = int(room.bx(cl["lo"][0])), int(room.bx(cl["hi"][0]))
            z0, z1 = int(room.bz(cl["lo"][2])), int(room.bz(cl["hi"][2]))
            w = int(np.clip(x1 - x0 + 1, 1, 4)) if cat == "led_panel" else 1
            d = int(np.clip(z1 - z0 + 1, 1, 4)) if cat == "led_panel" else 1
            objects.append({"type": cat, "x": int(np.clip(room.bx(cl["c"][0]) - (w - 1) // 2, 0, W - w)),
                            "z": int(np.clip(room.bz(cl["c"][2]) - (d - 1) // 2, 0, D - d)),
                            "w": w, "d": d, "h": 1})
        elif cat == "rug":
            x0, x1 = int(room.bx(cl["lo"][0])), int(room.bx(cl["hi"][0]))
            z0, z1 = int(room.bz(cl["lo"][2])), int(room.bz(cl["hi"][2]))
            objects.append({"type": "rug", "x": x0, "z": z0, "w": x1 - x0 + 1, "d": z1 - z0 + 1, "h": 1,
                            "color": DYE_PALETTE.nearest(cl["rgb"])})

    # ---------------- make sure the "house" has a usable front door
    if not any(o["type"] == "door" for o in openings):
        door = choose_door(room, openings, occupancy, objects, footprint)
        if door:
            door["material"] = "oak"
            openings.append(door)
            log(f"[plan] added front door on the {door['side']} wall at u={door['u']}")

    dom_lab = srgb_to_lab(WALL_COLOURS[wall_blocks[dominant]])
    roof_wood = "spruce" if dom_lab[0] < 40 else "dark_oak"
    return {
        "version": 1,
        "interior_shell": {"origin": origin.tolist(), "rotation": R.tolist(),
                           "y_shift": float(y_shift), "scale": float(f),
                           "bounds": {**{k: float(v) for k, v in bounds.items()},
                                      "ymin": 0.0, "ymax": float(hm)}},
        "scale": round(float(scale), 3),
        "metres": {"x": round(wm, 2), "z": round(dm, 2), "y": round(hm, 2)},
        "size": {"x": W, "z": D, "y": H},
        "floor": {"palette": floor_blocks, "grid": floor_grid.tolist()},
        "footprint": footprint.astype(int).tolist(),
        "walls": walls,
        "wall_main": wall_blocks[dominant],
        "ceiling": ceiling_block,
        "roof": {"wood": roof_wood, "axis": "x" if W >= D else "z"},
        "openings": openings,
        "objects": objects,
    }


def surface_grid(ij, rgb, shape, palette, default_rgb):
    """One block over the whole surface: the material covering most of what was seen. Shadows under tables and sun
    by a window once split one carpet into a patchwork of two blocks; rugs are separate objects."""
    mean, cnt = cell_colours(ij, rgb, shape) if len(ij) else (np.zeros((shape[0] * shape[1], 3)), np.zeros(shape[0] * shape[1]))
    obs = cnt > 0
    if obs.sum() == 0:
        return np.zeros(shape, int), [palette.nearest(default_rgb)]
    labels, blocks = quantize(mean[obs], cnt[obs].astype(float), palette)
    dominant = int(np.bincount(labels, weights=cnt[obs]).argmax())
    return np.zeros(shape, int), [blocks[dominant]]


def bed_style(cl, obj_clusters, H):
    """"bunk", "loft" or None (an ordinary bed) from the bed cluster's heights and what stands under it.
    H is the room's interior height in blocks: a raised bed needs UPPER_BUNK blocks below it and two above (to climb
    on and stand on it; HouseBuilder builds an ordinary bed otherwise)."""
    if H < UPPER_BUNK + 2 or cl.get("raised", 0.0) < RAISED_SHARE:
        return None
    def share_under(o):
        ix = min(o["hi"][0], cl["hi"][0]) - max(o["lo"][0], cl["lo"][0])
        iz = min(o["hi"][2], cl["hi"][2]) - max(o["lo"][2], cl["lo"][2])
        return max(ix, 0) * max(iz, 0) / max(1e-6, (o["hi"][0] - o["lo"][0]) * (o["hi"][2] - o["lo"][2]))
    under = any(c in UNDER_LOFT and o["hi"][1] < RAISED_Y + 0.3 and share_under(o) >= UNDER_SHARE
                for c, o in obj_clusters)
    if under:
        return "loft"
    return "bunk" if cl.get("low", 0.0) >= LOW_SHARE else "loft"


def ladder_cells(bed):
    """Floor cells just past the foot of each bed of a placed bed object (where HouseBuilder puts the ladder)."""
    x0, z0, w, d, head = bed["x"], bed["z"], bed["w"], bed["d"], bed["facing"]
    if head in ("west", "east"):
        x = x0 + 2 if head == "west" else x0 - 1
        return [(x, z0 + i) for i in range(d)]
    z = z0 + 2 if head == "north" else z0 - 1
    return [(x0 + i, z) for i in range(w)]


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
    cap = 2 if cat.startswith(GENERIC) else MAX_HEIGHT.get(cat)  # a noisy cluster must not become a pillar
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
    elif cat in ("chair", "sofa", "student_desk", "chair_stack"):
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
        if cat in ("chair", "chair_stack", "student_desk"):
            w = d = 1
    elif cat in ("table", "desk", "coffee_table", "bookshelf", "wardrobe", "cabinet", "dresser",
                 "nightstand", "counter", "podium", "piano"):
        obj["material"] = WOOD_PALETTE.nearest(cl["rgb"])
        if cat in ("nightstand", "podium"):
            w = d = 1
        if cat == "cabinet" and hi[1] < TALL_CABINET_M:
            h = 1  # dresser height, level with the desks and tables beside it (they are one block)
        if cat == "piano":
            w, d = max(2, w), 1
    elif cat in ("lamp", "plant", "trash", "toilet", "microwave", "fridge", "stove", "sink", "washer", "fan",
                 "shower", "coat_rack", "backpack", "umbrella_stand", "globe"):
        if cat in ("lamp", "plant", "trash", "toilet", "microwave", "fan", "coat_rack", "backpack", "umbrella_stand",
                   "globe"):
            w = d = 1
        if cat == "backpack":
            obj["color"] = DYE_PALETTE.nearest(cl["rgb"])
        if cat == "trash":
            obj["color"] = DYE_PALETTE.nearest(cl["rgb"])  # black trash, blue recycling, green compost, ...
        if cat == "fridge":
            h = min(H - 1, max(2, h))
        if cat in ("fridge", "stove", "sink", "washer", "toilet", "shower") and w * d > 2:
            w, d = (min(w, 2), 1) if front in ("north", "south") else (1, min(d, 2))
    elif cat == "locker":
        obj["color"] = DYE_PALETTE.nearest(cl["rgb"])  # lockers come painted: grey, blue, red, ...
        w, d = (max(1, w), 1) if front in ("north", "south") else (1, max(1, d))  # one block deep, along its wall
    elif cat == "tv":
        w, d = (max(1, w), 1) if front in ("north", "south") else (1, max(1, d))
    elif cat.startswith(VANILLA):
        bid, cube, name = vanilla_parts(cat)
        obj.update(type="vanilla", block="minecraft:" + bid, name=name, cube=cube, y=1)
        if not cube:
            w = d = h = 1
    elif cat.startswith(GENERIC):
        obj.update(type="generic", name=cat[len(GENERIC):], color=Palette(CONCRETE).nearest(cl["rgb"]))  # built as <color>_concrete
    w, d = max(1, min(w, W)), max(1, min(d, D))

    # Snap wall-hugging furniture onto the wall it is near (reconstruction noise leaves gaps). Bookcases are
    # usually built against a wall, often in front of piers or a recess that puts the fitted wall further back.
    snap = 1.5 if cat == "bookshelf" or cat.startswith("vanilla:bookshelf:") else 0.6
    if dist < snap and cat != "bed":
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


def choose_door(room: Room, openings, occupancy, objects, footprint=None):
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
            if footprint is not None and not footprint[inside[1], inside[0]]:
                continue
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


# ----------------------------------------------------------------------------------- wall features

WINDOW_SHARE = 0.5      # a wall cell is glass when this weighted share of visible views shows window
SHELF_SHARE = 0.5       # ... and built-in shelving when this share shows bookcase
FEATURE_MIN_SEEN = 0.2  # a wall needs this share of its cells seen for the photos to decide its windows
MAX_WINDOW = 0.6        # windows guessed from 3D clusters alone never take more of a wall than this
MIRROR_UNSEEN, MIRROR_SEEN = 0.1, 0.3


def _grid_of(openings, side, kind, rows, cols):
    g = np.zeros((rows, cols), bool)
    for o in openings:
        if o["side"] == side and o["type"] == kind:
            g[o["y"]:o["y"] + o["h"], o["u"]:o["u"] + o["w"]] = True
    return g


def _rects(grid):
    """Vertical runs per column, merged across neighbouring columns with the same run: [(u, w, y, h)]."""
    runs = []
    for u in range(grid.shape[1]):
        col = np.r_[False, grid[:, u], False].astype(int)
        starts, ends = np.flatnonzero(np.diff(col) == 1), np.flatnonzero(np.diff(col) == -1)
        runs += [(u, y0, y1 - y0) for y0, y1 in zip(starts, ends)]
    out = []
    for u, y, h in sorted(runs, key=lambda r: (r[1], r[2], r[0])):
        if out and out[-1][2] == y and out[-1][3] == h and out[-1][0] + out[-1][1] == u:
            out[-1] = (out[-1][0], out[-1][1] + 1, y, h)
        else:
            out.append((u, 1, y, h))
    return out


def apply_wall_features(plan, features, log=print, mirror=False):
    """Windows and built-in bookshelves from what the photos show on each wall (walltex.project_features).

    Windows: where a wall was seen well, glass goes exactly where the photos show glass, so the stone between
    windows stays. Unseen parts keep the windows guessed from 3D clusters, but those never take more than
    MAX_WINDOW of a wall: overlapping clusters once turned a whole wall into glass. Bookshelves seen on the
    wall become vanilla bookshelf furniture standing on the floor right in front of it (never part of the wall
    itself), and bookcases already placed at those spots are dropped as duplicates. mirror=True: a wall the photos hardly saw copies the
    opposite wall when that one was seen well (halls, libraries, churches)."""
    decided = {}
    for side, f in features.items():
        rows, cols = f["seen"].shape
        seen = f["seen"] >= 0.5
        doors = _grid_of(plan["openings"], side, "door", rows, cols)
        guessed = _grid_of(plan["openings"], side, "window", rows, cols)
        if seen.mean() >= FEATURE_MIN_SEEN:
            win = ((f["window"] >= WINDOW_SHARE) & seen) | (guessed & ~seen)
            if win.mean() > MAX_WINDOW:
                win &= seen
        else:
            win = guessed.copy()
            if win.mean() > MAX_WINDOW:
                # No photo evidence and implausibly much glass: keep every fourth column as a stone pier.
                win &= (np.arange(cols) % 4 != 3)[None, :]
                log(f"[plan] {side} wall: windows from clusters alone covered {guessed.mean():.0%}; piers kept")
        win[0] = win[-1] = False             # not at floor level nor right under the ceiling
        win &= ~doors
        shelf = (f["bookshelf"] >= SHELF_SHARE) & seen & ~win & ~doors
        decided[side] = {"win": win, "shelf": shelf, "seen": seen.mean()}

    opposite = {"north": "south", "south": "north", "west": "east", "east": "west"}
    mirrored = set()
    if mirror:
        for side, d in decided.items():
            o = opposite[side]
            if d["seen"] < MIRROR_UNSEEN and o in decided and decided[o]["seen"] >= MIRROR_SEEN:
                own_doors = _grid_of(plan["openings"], side, "door", *decided[o]["win"].shape)
                decided[side] = {"win": decided[o]["win"] & ~own_doors, "shelf": decided[o]["shelf"] & ~own_doors,
                                 "seen": d["seen"]}
                mirrored.add((side, o))
                log(f"[plan] {side} wall ({d['seen']:.0%} seen) mirrors the {o} wall ({decided[o]['seen']:.0%} seen)")

    colour = {o["side"]: o.get("color") for o in plan["openings"] if o["type"] == "window" and o.get("color")}
    keep = [o for o in plan["openings"] if not (o["type"] == "window" and o["side"] in decided)]
    for side, d in decided.items():
        for u, w, y, h in _rects(d["win"]):
            keep.append({"type": "window", "side": side, "u": int(u), "w": int(w), "y": int(y), "h": int(h),
                         **({"color": colour[side]} if side in colour else {})})
        log(f"[plan] {side} wall: {d['seen']:.0%} seen, {d['win'].mean():.0%} glass, "
            f"shelving seen in {int(d['shelf'].any(0).sum())} column(s)")
    plan["openings"] = keep

    W, D = plan["size"]["x"], plan["size"]["z"]
    for side, d in decided.items():
        if not d["shelf"].any():
            continue
        cols_ = np.flatnonzero(d["shelf"].any(0))
        near = lambda o: {"north": o["z"] <= 2, "south": o["z"] + o["d"] >= D - 2,
                          "west": o["x"] <= 2, "east": o["x"] + o["w"] >= W - 2}[side]
        span = lambda o: range(o["x"], o["x"] + o["w"]) if side in ("north", "south") else range(o["z"], o["z"] + o["d"])
        before = len(plan["objects"])
        plan["objects"] = [o for o in plan["objects"] if not (
            (o["type"] == "bookshelf" or o.get("block") == "minecraft:bookshelf")
            and near(o) and set(span(o)) & set(cols_.tolist()))]
        if len(plan["objects"]) < before:
            log(f"[plan] {side} wall: {before - len(plan['objects'])} bookcase(s) replaced by the shelving seen on it")
        # Shelving stands on the floor: each column of shelf cells becomes a stack from the floor up to its top.
        top = np.where(d["shelf"].any(0), d["shelf"].shape[0] - np.argmax(d["shelf"][::-1], 0), 0)
        top[top < 2] = 0  # a lone block of "shelf" is noise, not furniture
        # Not in a doorway (a door column, and the two cells in front of any door) nor where furniture stands.
        taken = np.zeros((W, D), bool)
        for o in plan["objects"]:
            if o.get("w") and o.get("d") and o["type"] not in ("ceiling_light", "chandelier", "projector", "rug"):
                taken[o["x"]:o["x"] + o["w"], o["z"]:o["z"] + o["d"]] = True
        for o in plan["openings"]:
            if o["type"] == "door":
                for du in range(o["w"]):
                    uu = o["u"] + du
                    for depth in (0, 1):
                        cx, cz = {"north": (uu, depth), "south": (uu, D - 1 - depth),
                                  "west": (depth, uu), "east": (W - 1 - depth, uu)}[o["side"]]
                        if 0 <= cx < W and 0 <= cz < D:
                            taken[cx, cz] = True
        cell = {"north": lambda uu: (uu, 0), "south": lambda uu: (uu, D - 1),
                "west": lambda uu: (0, uu), "east": lambda uu: (W - 1, uu)}[side]
        for uu in range(len(top)):
            cx, cz = cell(uu)
            if not (0 <= cx < W and 0 <= cz < D) or taken[cx, cz]:
                top[uu] = 0
        front = {"north": "south", "south": "north", "west": "east", "east": "west"}[side]
        u = 0
        while u < len(top):
            if top[u] == 0:
                u += 1
                continue
            v = u
            while v + 1 < len(top) and top[v + 1] == top[u]:
                v += 1
            w = v - u + 1
            x, z, ww, dd = {"north": (u, 0, w, 1), "south": (u, D - 1, w, 1),
                            "west": (0, u, 1, w), "east": (W - 1, u, 1, w)}[side]
            plan["objects"].append({"type": "vanilla", "block": "minecraft:bookshelf", "name": "bookshelf", "cube": True,
                                    "facing": front, "x": int(x), "z": int(z), "w": int(ww), "d": int(dd),
                                    "h": int(top[u]), "y": 1})
            u = v + 1
    for side, o in mirrored:
        tex = plan.get("textures", {}).get("walls", {})
        if o in tex:
            tex[side] = dict(tex[o])          # same image and flip: the same picture at the same x (or z)
        else:
            tex.pop(side, None)               # blocks and pictures always come from the same wall
        plan["walls"][side] = {"palette": list(plan["walls"][o]["palette"]), "grid": [r[:] for r in plan["walls"][o]["grid"]]}
    clear_around_doors(plan)


def clear_around_doors(plan, margin=1):
    """No picture frames on the wall around a door: its columns plus `margin` either side, up to one row above it.
    A door the photos' labels missed, or one placed a little off where the photos show it, would otherwise be
    drawn on the maps next to the real door (or its exit sign above it)."""
    walls = plan.get("textures", {}).get("walls", {})
    for o in plan["openings"]:
        tex = walls.get(o["side"]) if o["type"] == "door" else None
        if not tex:
            continue
        t = np.array(tex["tiles"])
        r1 = min(t.shape[0], o["y"] + o["h"] + 1)
        u0, u1 = max(0, o["u"] - margin), min(t.shape[1], o["u"] + o["w"] + margin)
        t[:r1, u0:u1] = 0
        tex["tiles"] = t.tolist()

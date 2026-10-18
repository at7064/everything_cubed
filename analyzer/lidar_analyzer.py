"""everything_cubed LiDAR analyzer: a scan streamed from the iOS scanner app -> plan.json for the mod.

Usage:
  python lidar_analyzer.py --images <scan folder> [--out <folder>/plan.json] [--mc-jar client.jar]
                           [--scale BLOCKS_PER_METRE] [--segformer]

The photo analyzer has to reconstruct the room from the photos (MapAnything / VGGT) and then guess what every
pixel is (SegFormer). A LiDAR scan already carries metric depth, the camera pose of every photo, gravity and
(usually) Apple RoomPlan's room model, so this analyzer needs no ML model at all by default:

1. Geometry: every LiDAR depth pixel is unprojected with ARKit's intrinsics and pose. Only frames with normal
   tracking are used (when enough remain). Depth up to 5 m is kept at medium confidence, up to --max-depth (10 m)
   at high confidence, so the far walls of big rooms are measured too.
2. Labels from geometry (all at the depth map's 256x192, normals straight from the depth image grid):
   - floor / ceiling: the lowest / highest horizontal plane along gravity (ARKit +Y);
   - walls: in each of the room's 4 directions (from RoomPlan's walls, else the LiDAR normals), the innermost
     vertical plane that faces into the room, is at least wall-high (furniture sides are not) and that the LiDAR
     never sees the floor through (a row of pillars or a glass railing is not a wall);
   - windows: the LiDAR sees through glass, so depth *behind* a wall plane can only come through a window (or
     an opening). Those pixels are labelled window and moved onto the wall where their ray crossed it;
   - ceiling lights: the near-white, much-brighter-than-the-rest patches of the ceiling plane;
   - furniture: RoomPlan's object boxes (beds, tables, sofas, chairs, storage, appliances, ...) and its
     doors / windows / openings; plus table tops found as flat horizontal surfaces at table height;
   - everything else (people, clutter) stays unlabelled and is not built.
3. --segformer (optional, needs transformers): SegFormer also names the small things geometry can't
   (lamps, plants, paintings, TVs, rugs, shelves). Geometry keeps the final say on floor, walls and ceiling.
4. The existing roomplan.build_plan and walltex.render_walls turn this into the plan as for photos, at the
   measured size (no --enlarge), with RoomPlan's windows and wall ends where it has them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from perception import (ARKIT_MAX_DEPTH, ARKIT_TO_CV, ARKIT_WIDTH, _upright, category_lut, fuse,  # noqa: E402
                        upright_camera)

# Labels this analyzer gives pixels; index = label id. Names are roomplan categories.
LABELS = ["other", "floor", "ceiling", "wall", "window", "door", "bed", "table", "coffee_table", "desk", "chair",
          "sofa", "cabinet", "wardrobe", "bookshelf", "shelf", "counter", "fridge", "stove", "sink", "washer",
          "toilet", "bathtub", "fireplace", "tv", "lamp", "ceiling_light", "chandelier", "sconce", "plant",
          "painting", "rug", "microwave", "storage", "trash", "curtain", "mirror", "clock"]
L = {n: i for i, n in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
IDENTITY = {n: n for n in LABELS}

# Apple RoomPlan (CapturedRoom.Object.Category) -> roomplan category.
ROOMPLAN_OBJECTS = {"storage": "cabinet", "refrigerator": "fridge", "stove": "stove", "oven": "stove", "bed": "bed",
                    "sink": "sink", "washerDryer": "washer", "dishwasher": "washer", "toilet": "toilet",
                    "bathtub": "bathtub", "table": "table", "sofa": "sofa", "chair": "chair",
                    "fireplace": "fireplace", "television": "tv"}
ROOMPLAN_OPENINGS = {"doors": "door", "openings": "door", "windows": "window"}

FAR_DEPTH = 10.0       # metres: high-confidence LiDAR depth is kept this far (medium only up to ARKIT_MAX_DEPTH)
MIN_WALL_AREA = 1.5    # m^2 of surface (summed over views) for a vertical plane to be a wall candidate
MIN_WALL_SPAN = 1.2    # metres: a wall rises at least this high (or 60% of the room), furniture sides do not
WALL_SURFACE_TOL = (0.06, 0.02)  # wall pictures only from photo pixels whose depth is this close to the wall
LIGHT_LUMA = 235  # ceiling pixels this bright (and well above the rest of the ceiling) are light fixtures
TABLE_BAND = (0.6, 0.95)  # metres above the floor where table tops are looked for
TABLE_AREA = (0.15, 8.0)   # m^2 (tables far away or at a grazing angle are seen in pieces)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------------- loading

def _mat(t):
    """RoomPlan / simd 4x4 matrices are stored column by column."""
    return np.array(t, dtype=np.float64).reshape(4, 4).T


def _unproject(depth, K, c2w):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def _normals(P, valid, depth, C, step=2):
    """Per-pixel normals from the organised depth grid (central differences, `step` pixels apart), facing the
    camera. NaN where a neighbour is missing or across a depth edge."""
    h, w = depth.shape
    n = np.full((h, w, 3), np.nan)
    s = step
    a = P[s:-s, 2 * s:] - P[s:-s, :-2 * s]
    b = P[2 * s:, s:-s] - P[:-2 * s, s:-s]
    c = np.cross(b, a)
    d = depth[s:-s, s:-s]
    ok = valid[s:-s, s:-s] & valid[s:-s, 2 * s:] & valid[s:-s, :-2 * s] & valid[2 * s:, s:-s] & valid[:-2 * s, s:-s]
    tol = 0.04 * d + 0.03
    for nb in (depth[s:-s, 2 * s:], depth[s:-s, :-2 * s], depth[2 * s:, s:-s], depth[:-2 * s, s:-s]):
        ok &= np.abs(nb - d) < tol
    norm = np.linalg.norm(c, axis=-1)
    ok &= norm > 1e-9
    c = c / np.maximum(norm, 1e-12)[..., None]
    flip = ((C - P[s:-s, s:-s]) * c).sum(-1) < 0
    c[flip] *= -1
    c[~ok] = np.nan
    n[s:-s, s:-s] = c
    return n


def frame_geometry(f, depth, conf, img, max_depth=FAR_DEPTH, width=ARKIT_WIDTH):
    """One frame's depth-resolution geometry ("geo") used for labelling and the photo at `width` ("view", the
    format perception's backends return) used for wall textures. `f` holds the frame's capture.json fields,
    `depth` its depth map (metres, depthSize), `conf` its ARKit confidence (or None), `img` its PIL photo."""
    from PIL import Image

    dh, dw = depth.shape
    depth = depth.astype(np.float64)
    if conf is None:
        conf = np.full((dh, dw), 2, np.uint8)
    valid = np.isfinite(depth) & (depth > 0.1) & (((depth < ARKIT_MAX_DEPTH) & (conf >= 1)) |
                                                  ((depth < max_depth) & (conf >= 2)))
    iw, ih = f.get("imageSize", (dw * 7.5, dh * 7.5))
    K = np.array(f["K"], dtype=np.float64).reshape(3, 3)
    c2w = np.array(f["cam2world"], dtype=np.float64).reshape(4, 4) @ ARKIT_TO_CV
    Kd = K.copy(); Kd[:2] *= dw / iw
    P = _unproject(np.where(valid, depth, 0), Kd, c2w)
    C = c2w[:3, 3]
    img = img.convert("RGB")
    W, H = width, round(ih * width / iw)
    rgb = img.resize((W, H), Image.BILINEAR)
    return {"name": f.get("image"), "depth": depth, "conf": conf, "valid": valid, "P": P, "C": C, "c2w": c2w,
            "K": K, "imageSize": (iw, ih), "normal": _normals(P, valid, depth, C),
            "area": (depth / Kd[0, 0]) * (depth / Kd[1, 1]),   # m^2 a pixel covers, facing the camera
            "rgb_d": np.asarray(rgb.resize((dw, dh), Image.BOX)), "rgb": np.asarray(rgb)}


def load_scan(folder: Path, max_depth=FAR_DEPTH, width=ARKIT_WIDTH):
    """Frames of a capture.json, each as frame_geometry() returns it, and RoomPlan's room model (or None)."""
    from PIL import Image

    cap = json.loads((folder / "capture.json").read_text())
    frames = [f for f in cap["frames"] if "depth" in f]
    if not frames:
        raise SystemExit(f"{folder / 'capture.json'} has no LiDAR depth")
    good = [f for f in frames if f.get("tracking", "normal") == "normal"]
    if len(good) >= max(3, len(frames) // 3):  # poses from limited tracking drift; drop them when we can
        frames = good
    out = []
    for f in frames:
        dw, dh = f["depthSize"]
        fmt = f["depth"].get("format", "f16")
        depth = np.fromfile(folder / f["depth"]["file"], dtype="<f2" if fmt == "f16" else "<f4").reshape(dh, dw)
        conf = (np.fromfile(folder / f["confidence"]["file"], dtype=np.uint8).reshape(dh, dw)
                if "confidence" in f else None)
        out.append(frame_geometry(f, depth, conf, Image.open(folder / f["image"]), max_depth, width))
    room = None
    if (folder / cap.get("room", "room.json")).is_file():
        room = json.loads((folder / cap.get("room", "room.json")).read_text())
    return cap, out, room


# ----------------------------------------------------------------------------------- room frame from gravity

def _weighted_mode(values, weights, bin_size, lo=None, hi=None):
    if not len(values):
        return None, 0.0
    lo = values.min() if lo is None else lo
    hi = values.max() if hi is None else hi
    nb = max(1, int(np.ceil((hi - lo) / bin_size)) + 1)
    hist = np.bincount(np.clip(((values - lo) / bin_size).astype(int), 0, nb - 1), weights, nb)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    i = int(hist.argmax())
    near = np.abs(values - (lo + (i + 0.5) * bin_size)) < 1.5 * bin_size
    return float(np.average(values[near], weights=weights[near])), float(hist[i])


def manhattan_axes(frames, room, log):
    """The room's horizontal axes (ex, ez) in ARKit's world frame, from RoomPlan's walls when it found any (it
    fits them to the whole scan), else from the LiDAR's vertical surfaces (4-fold symmetric mean angle)."""
    th, w, src = [], [], "LiDAR normals"
    if room and room.get("walls"):
        for wl in room["walls"]:
            T = _mat(wl["transform"])
            th.append(np.arctan2(T[2, 0], T[0, 0])); w.append(max(0.1, wl["dimensions"][0] * wl["dimensions"][1]))
        src = f"RoomPlan ({len(room['walls'])} wall(s))"
    else:
        for fr in frames:
            n = fr["normal"].reshape(-1, 3)
            a = fr["area"].ravel()
            ok = np.isfinite(n[:, 0]) & (np.abs(n[:, 1]) < 0.15)
            th.append(np.arctan2(n[ok, 2], n[ok, 0])); w.append(a[ok])
        th, w = np.concatenate(th), np.concatenate(w)
    th, w = np.asarray(th), np.asarray(w)
    if not len(th) or w.sum() <= 0:
        a = 0.0
    else:
        a = float(np.arctan2((w * np.sin(4 * th)).sum(), (w * np.cos(4 * th)).sum()) / 4)
    log(f"[lidar] wall directions from {src}: {np.degrees(a):.1f} deg")
    ex = np.array([np.cos(a), 0.0, np.sin(a)])
    ez = np.cross(ex, [0.0, 1.0, 0.0])
    return ex, ez


def floor_ceiling(frames, room, log):
    ys, ws, ups = [], [], []
    for fr in frames:
        n = fr["normal"]
        horiz = np.isfinite(n[..., 1]) & (np.abs(n[..., 1]) > 0.9)
        ys.append(fr["P"][horiz][:, 1]); ws.append(fr["area"][horiz]); ups.append(n[horiz][:, 1] > 0)
    y, w, up = np.concatenate(ys), np.concatenate(ws), np.concatenate(ups)
    cam_y = float(np.median([fr["C"][1] for fr in frames]))
    floor_y, _ = _weighted_mode(y[up & (y < cam_y - 0.3)], w[up & (y < cam_y - 0.3)], 0.02)
    rp = [_mat(f["transform"])[1, 3] for f in (room or {}).get("floors", [])]
    if floor_y is None and rp:
        floor_y = float(np.median(rp))
    if floor_y is None:
        floor_y = cam_y - 1.4
        log("[lidar] floor not seen; assuming the phone was held 1.4 m above it")
    down = ~up & (y > cam_y + 0.2)
    ceil_y, ceil_w = _weighted_mode(y[down], w[down], 0.02)
    if ceil_y is not None and ceil_w < 0.5:  # a few m^2 of overhang is not a ceiling
        ceil_y = None
    log(f"[lidar] floor at {floor_y:.2f} m" + (f" (RoomPlan {np.median(rp):.2f})" if rp else "") +
        (f", ceiling {ceil_y - floor_y:.2f} m above it" if ceil_y is not None else ", ceiling not seen"))
    return floor_y, ceil_y


def find_walls(frames, ex, ez, floor_y, ceil_y, log):
    """The innermost vertical plane on each side that faces into the room, is wall-high and that the floor is
    not seen through: {(axis, sign): coordinate along ex (axis 0) or ez (axis 1)}."""
    axes = (ex, ez)
    pts, nrm, area = [], [], []
    for fr in frames:
        n = fr["normal"]
        ok = np.isfinite(n[..., 0]) & (np.abs(n[..., 1]) < 0.3)
        pts.append(fr["P"][ok]); nrm.append(n[ok]); area.append(fr["area"][ok])
    P, N, A = np.concatenate(pts), np.concatenate(nrm), np.concatenate(area)
    cam = np.median([fr["C"] for fr in frames], axis=0)
    room_h = (ceil_y - floor_y) if ceil_y is not None else 2.5
    span_needed = min(MIN_WALL_SPAN, 0.6 * room_h)
    walls = {}
    for ai, ax in enumerate(axes):
        c, nc = P @ ax, N @ ax
        c0 = cam @ ax
        for sign in (-1, 1):
            # A wall on the +side faces -axis (towards the room), and lies beyond the camera.
            sel = (nc * -sign > 0.9) & (sign * (c - c0) > 0.3)
            if A[sel].sum() < MIN_WALL_AREA:
                continue
            cs, ws, ys = c[sel], A[sel], P[sel, 1]
            lo = cs.min()
            nb = int((cs.max() - lo) / 0.05) + 1
            bi = ((cs - lo) / 0.05).astype(int)
            hist = np.convolve(np.bincount(bi, ws, nb), [1, 1, 1], "same")
            thr = max(MIN_WALL_AREA, 0.1 * hist.max())
            # Innermost first: what is seen through a window (the building opposite) lies further out.
            order = np.arange(nb) if sign > 0 else np.arange(nb)[::-1]
            for b in order:
                if hist[b] < thr:
                    continue
                near = np.abs(bi - b) <= 1
                if np.percentile(ys[near], 95) - np.percentile(ys[near], 5) < span_needed:
                    continue
                cw = float(np.average(cs[near], weights=ws[near]))
                seen_past = _area_past(frames, ax, sign, cw, floor_y)
                if seen_past > 0.5 * hist[b]:   # a row of pillars, a railing: the floor goes on behind it
                    continue
                walls[(ai, sign)] = cw
                break
    names = {(0, -1): "x-", (0, 1): "x+", (1, -1): "z-", (1, 1): "z+"}
    log("[lidar] walls: " + ", ".join(f"{names[k]} {v - cam @ axes[k[0]]:+.2f} m" for k, v in sorted(walls.items())) +
        (f"; not found: {', '.join(names[k] for k in names if k not in walls)}" if len(walls) < 4 else ""))
    return walls


def _area_past(frames, ax, sign, c, floor_y, low=0.5):
    """m^2 of surface seen *through* the plane coordinate c along `ax`, with the ray crossing it less than `low`
    metres above the floor. Windows don't reach the floor, so a real wall has almost none (doors are narrow)."""
    total = 0.0
    for fr in frames:
        C, P = fr["C"], fr["P"]
        if (C @ ax - c) * sign >= 0:
            continue
        cp = P @ ax
        past = fr["valid"] & ((cp - c) * sign > 0.3)
        if not past.any():
            continue
        t = (c - C @ ax) / (cp[past] - C @ ax)
        y = C[1] + t * (P[past][:, 1] - C[1])
        total += float(fr["area"][past][(y < floor_y + low) & (y > floor_y - 0.2)].sum())
    return total


def roomplan_walls(frames, room, ex, ez, log):
    """RoomPlan's walls, as {(axis, sign): coordinate} like find_walls. RoomPlan fits each wall to the whole scan,
    so where it found one it wins over the LiDAR's innermost plane, which can land on a fridge front or a desk
    side and turn everything behind it into glass. On a side with several RoomPlan walls (an L-shaped room) the
    longest one is used."""
    if not room or not room.get("walls"):
        return {}
    cam = np.median([fr["C"] for fr in frames], axis=0)
    best = {}
    for wl in room["walls"]:
        T = _mat(wl["transform"])
        n = T[:3, 2]                       # a RoomPlan wall's normal is its local z
        ai = 0 if abs(n @ ex) >= abs(n @ ez) else 1
        ax = (ex, ez)[ai]
        c = float(T[:3, 3] @ ax)
        key = (ai, 1 if c > cam @ ax else -1)
        if key not in best or wl["dimensions"][0] > best[key][1]:
            best[key] = (c, wl["dimensions"][0])
    names = {(0, -1): "x-", (0, 1): "x+", (1, -1): "z-", (1, 1): "z+"}
    log("[lidar] RoomPlan walls used: " + ", ".join(f"{names[k]} {v[0] - cam @ (ex, ez)[k[0]]:+.2f} m"
                                                   for k, v in sorted(best.items())))
    return {k: v[0] for k, v in best.items()}


def roomplan_window_cells(room, frame):
    """RoomPlan's windows as rectangles of wall cells for roomplan.apply_wall_features ({"side", "u0", "u1", "y0",
    "y1"}), through the plan's ARKit mapping `frame` (plan["arkit"]). Each goes on the wall its centre is nearest."""
    if not room or not room.get("windows"):
        return []
    origin, R = np.array(frame["origin"]), np.array(frame["R"])
    W, D, H = frame["W"], frame["D"], frame["H"]

    def cell(P):     # (block x, block z, row) of an ARKit world point
        loc = ((P - origin) @ R.T - [0, frame["y_shift"], 0]) * frame["f"]
        return ((loc[0] - frame["xmin"]) / (frame["xmax"] - frame["xmin"]) * W,
                (loc[2] - frame["zmin"]) / (frame["zmax"] - frame["zmin"]) * D, loc[1] / frame["hm"] * H)

    out = []
    for w in room["windows"]:
        T, (dw, dh) = _mat(w["transform"]), w["dimensions"][:2]
        corners = np.array([cell(T[:3, 3] + sx * T[:3, 0] * dw / 2 + sy * T[:3, 1] * dh / 2)
                            for sx in (-1, 1) for sy in (-1, 1)])
        x, z, _ = corners.mean(0)
        side = min((("west", x), ("east", W - x), ("north", z), ("south", D - z)), key=lambda t: abs(t[1]))[0]
        u = corners[:, 0] if side in ("north", "south") else corners[:, 1]
        out.append({"side": side, "u0": float(u.min()), "u1": float(u.max()),
                    "y0": float(corners[:, 2].min()), "y1": float(corners[:, 2].max())})
    return out


def roomplan_wall_ends(room):
    """Both end points (on the floor line) of every RoomPlan wall, in ARKit world coordinates."""
    ends = []
    for wl in (room or {}).get("walls", []):
        T = _mat(wl["transform"])
        half = T[:3, 0] * wl["dimensions"][0] / 2          # a RoomPlan wall runs along its local x
        ends += [T[:3, 3] - half, T[:3, 3] + half]
    return ends


def _under_roomplan_object(table, room, log, share=0.3, seat_share=0.8):
    """A table top found in the depth that is mostly the top of something RoomPlan already reported (a bed's
    mattress is at table height, so is a dresser, and RoomPlan's own tables), which would otherwise be built a
    second time as a table. Chairs and sofas stand partly under tables, so they only count when the top lies almost
    wholly inside one (seat_share): then it is part of the seat (a flat patch at a chair's top), never a table."""
    t_lo, t_hi, _, x0, z0, m, sl = table
    i, k = np.nonzero(m)
    P = np.stack([x0 + (i + sl[0].start + 0.5) * 0.05, np.full(len(i), (t_lo + t_hi) / 2),
                  z0 + (k + sl[1].start + 0.5) * 0.05], 1)
    for o in (room or {}).get("objects", []):
        T, d = _mat(o["transform"]), np.asarray(o["dimensions"], float)
        loc = (P - T[:3, 3]) @ T[:3, :3]
        inside = (np.abs(loc[:, 0]) <= d[0] / 2 + 0.1) & (np.abs(loc[:, 2]) <= d[2] / 2 + 0.1)
        if inside.mean() >= (seat_share if next(iter(o["category"])) in ("chair", "sofa") else share):
            log(f"[lidar] a table top found in the depth is RoomPlan's {next(iter(o['category']))}: not a table")
            return True
    return False


# ----------------------------------------------------------------------------------- per-pixel labels

def _box_mask(P, T, dims, margin=0.04):
    loc = (P - T[:3, 3]) @ T[:3, :3]
    return np.all(np.abs(loc) <= np.asarray(dims) / 2 + margin, axis=-1)


def _ray_through_rect(C, P, centre, ax_u, ax_v, n, half_u, half_v, beyond):
    """Pixels whose surface lies more than `beyond` behind the plane (centre, n) as seen from C, and whose ray
    crossed it inside the rectangle. Returns (mask, crossing points)."""
    dc = (C - centre) @ n
    dp = (P - centre) @ n
    side = np.sign(dc) if dc != 0 else 1.0
    behind = dp * side < -beyond
    t = dc / np.where(np.abs(dc - dp) > 1e-9, dc - dp, np.inf)
    X = C + t[..., None] * (P - C)
    rel = X - centre
    inside = (np.abs(rel @ ax_u) <= half_u) & (np.abs(rel @ ax_v) <= half_v)
    return behind & inside & (t > 0) & (t < 1), X


def label_frame(fr, ex, ez, floor_y, ceil_y, walls, room, tables):
    P, n, d, valid = fr["P"], fr["normal"], fr["depth"], fr["valid"]
    lab = np.zeros(d.shape, np.int32)
    has_n = np.isfinite(n[..., 0])
    tol = np.maximum(0.05, 0.012 * d)
    y = P[..., 1]
    top = ceil_y if ceil_y is not None else floor_y + 40
    lab[valid & (np.abs(y - floor_y) < tol) & (~has_n | (n[..., 1] > 0.7))] = L["floor"]
    if ceil_y is not None:
        lab[valid & (np.abs(y - ceil_y) < tol) & (~has_n | (n[..., 1] < -0.7))] = L["ceiling"]
        # Light fixtures sit flush in the ceiling: the brightest, near-white part of it (no ML needed).
        ceil = lab == L["ceiling"]
        if ceil.sum() >= 200:
            luma = fr["rgb_d"].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            bright = ceil & (luma >= max(LIGHT_LUMA, np.median(luma[ceil]) + 40)) & (fr["rgb_d"].min(-1) >= LIGHT_LUMA - 20)
            lab[bright] = L["ceiling_light"]

    moved = P.copy()
    axes = (ex, ez)
    for (ai, sign), c in walls.items():
        ax = axes[ai]
        cp = P @ ax
        faces = ~has_n | (np.abs(n @ ax) > 0.7)
        between = (y > floor_y + 0.5 * tol) & (y < top - 0.5 * tol)
        lab[valid & (np.abs(cp - c) < 1.5 * tol) & faces & between] = L["wall"]
        # Seen through the wall plane: through glass (or an open door). The ray crossed the wall between the
        # floor and the ceiling, inside the other two walls. When RoomPlan measured the walls, only its own
        # windows, doors and openings (below) are see-through: a wall plane a little off, or depth leaking past
        # a corner, would otherwise turn the furniture behind the plane and the wall itself into glass.
        C = fr["C"]
        if (C @ ax - c) * sign >= 0 or (room or {}).get("walls"):
            continue
        other = axes[1 - ai]
        lo_o, hi_o = walls.get((1 - ai, -1), -np.inf), walls.get((1 - ai, 1), np.inf)
        dp = (P - C) @ ax
        t = (c - C @ ax) / np.where(np.abs(dp) > 1e-9, dp, np.inf)
        X = C + t[..., None] * (P - C)
        xo = X @ other
        through = (valid & ((cp - c) * sign > np.maximum(0.3, 0.05 * d)) & (t > 0) & (t < 1)
                   & (X[..., 1] > floor_y + 0.3) & (X[..., 1] < (top if ceil_y is not None else floor_y + 3.0) - 0.1)
                   & (xo > lo_o + 0.1) & (xo < hi_o - 0.1))
        lab[through] = L["window"]
        moved[through] = X[through]

    rp = room or {}
    for key, cat in ROOMPLAN_OPENINGS.items():
        for o in rp.get(key, []):
            T = _mat(o["transform"])
            w, h = o["dimensions"][0], o["dimensions"][1]
            on = valid & _box_mask(P, T, (w, h, 0.3), 0.02)
            lab[on] = L[cat]
            through, X = _ray_through_rect(fr["C"], P, T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2], w / 2, h / 2, 0.15)
            through &= valid
            lab[through] = L[cat]
            moved[through] = X[through]
    for o in rp.get("objects", []):
        kind = next(iter(o["category"]))
        cat = ROOMPLAN_OBJECTS.get(kind)
        if cat is None:
            continue
        if cat == "table" and o.get("attributes", {}).get("TableType") == "coffee":
            cat = "coffee_table"
        inside = valid & _box_mask(P, _mat(o["transform"]), o["dimensions"]) & (y > floor_y + tol)
        lab[inside & (lab != L["wall"])] = L[cat]

    # Table tops found from the geometry (flat, horizontal, at table height) that RoomPlan did not report.
    for t_lo, t_hi, cells, x0, z0, *_ in tables:
        band = valid & (lab == 0) & (y > t_lo) & (y < t_hi)
        if band.any():
            ci = np.floor((P[..., 0] - x0) / 0.05).astype(int), np.floor((P[..., 2] - z0) / 0.05).astype(int)
            okc = band & (ci[0] >= 0) & (ci[0] < cells.shape[0]) & (ci[1] >= 0) & (ci[1] < cells.shape[1])
            hit = np.zeros_like(okc)
            hit[okc] = cells[ci[0][okc], ci[1][okc]]
            lab[hit] = L["table"]
    lab[~valid] = 0
    return lab, moved


def find_tables(frames, floor_y, log, cell=0.05):
    """Flat horizontal surfaces at table height: [(y lo, y hi, footprint cells, x0, z0)] in world x/z."""
    from scipy import ndimage

    pts, areas = [], []
    for fr in frames:
        n = fr["normal"]
        h = fr["P"][..., 1] - floor_y
        ok = np.isfinite(n[..., 1]) & (n[..., 1] > 0.8) & (h > TABLE_BAND[0]) & (h < TABLE_BAND[1])
        pts.append(fr["P"][ok]); areas.append(fr["area"][ok])
    P, A = np.concatenate(pts), np.concatenate(areas)
    if len(P) < 50:
        return []
    x0, z0 = P[:, 0].min(), P[:, 2].min()
    i, k = ((P[:, 0] - x0) / cell).astype(int), ((P[:, 2] - z0) / cell).astype(int)
    shape = (i.max() + 1, k.max() + 1)
    cover = np.zeros(shape); np.add.at(cover, (i, k), A)
    hsum = np.zeros(shape); np.add.at(hsum, (i, k), A * P[:, 1])
    occ = ndimage.binary_closing(cover > 0, np.ones((3, 3)))
    comp, _ = ndimage.label(occ)
    tables = []
    for ci, sl in enumerate(ndimage.find_objects(comp), start=1):
        m = comp[sl] == ci
        area = m.sum() * cell * cell
        if not TABLE_AREA[0] <= area <= TABLE_AREA[1]:
            continue
        seen = m & (cover[sl] > 0)
        ys, ws = hsum[sl][seen] / cover[sl][seen], cover[sl][seen]
        top = float(np.average(ys, weights=ws))
        if np.sqrt(np.average((ys - top) ** 2, weights=ws)) > 0.04:   # not one flat surface
            continue
        # Mostly filled, not a thin ring (a lap, a chair's arms).
        if m.mean() < 0.4:
            continue
        cells = np.zeros(shape, bool)
        cells[sl] = m
        cells = ndimage.binary_dilation(cells, np.ones((3, 3)))
        tables.append((top - 0.05, top + 0.04, cells, x0, z0, m, sl))
    log(f"[lidar] {len(tables)} table top(s) found in the depth")
    return tables


def _box_surface(T, dims, floor_y, step=0.04):
    """Points on the top and the sides of a RoomPlan box (down to the floor), `step` apart."""
    hx, hy, hz = np.asarray(dims, float) / 2
    gx, gz = np.arange(-hx, hx + 1e-9, step), np.arange(-hz, hz + 1e-9, step)
    bottom = max(-hy, floor_y - T[1, 3])
    gy = np.arange(bottom, hy + 1e-9, step)
    faces = [np.stack(np.meshgrid(gx, [hy], gz, indexing="ij"), -1).reshape(-1, 3)]
    for x in (-hx, hx):
        faces.append(np.stack(np.meshgrid([x], gy, gz, indexing="ij"), -1).reshape(-1, 3))
    for z in (-hz, hz):
        faces.append(np.stack(np.meshgrid(gx, gy, [z], indexing="ij"), -1).reshape(-1, 3))
    loc = np.concatenate(faces)
    return loc @ T[:3, :3].T + T[:3, 3]


def solid_objects(room, tables, floor_y, pts, rgb, cats):
    """RoomPlan's boxes and the table tops found, as surfaces sampled every 4 cm. LiDAR sees furniture only in
    patches (one table came out as 3 clusters), so roomplan.build_plan gets each one whole and at its measured
    size. Colours come from the nearest real point of the same category (or any point), within 15 cm."""
    from scipy.spatial import cKDTree

    new_p, new_c = [], []
    for o in (room or {}).get("objects", []):
        cat = ROOMPLAN_OBJECTS.get(next(iter(o["category"])))
        if cat is None:
            continue
        if cat == "table" and o.get("attributes", {}).get("TableType") == "coffee":
            cat = "coffee_table"
        new_p.append(_box_surface(_mat(o["transform"]), o["dimensions"], floor_y)); new_c.append(cat)
    for t_lo, t_hi, _, x0, z0, m, sl in tables:
        i, k = np.nonzero(m)
        top = np.full(len(i), (t_lo + t_hi) / 2 + 0.005)
        new_p.append(np.stack([x0 + (i + sl[0].start + 0.5) * 0.05, top, z0 + (k + sl[1].start + 0.5) * 0.05], 1))
        new_c.append("table")
    if not new_p:
        return pts, rgb, cats
    tree_all = cKDTree(pts)
    P2, C2, L2 = [pts], [rgb], [cats]
    for p, cat in zip(new_p, new_c):
        same = np.flatnonzero(cats == cat)
        tree = cKDTree(pts[same]) if len(same) else tree_all
        d, j = tree.query(p, distance_upper_bound=0.15)
        hit = np.isfinite(d)
        src = same[j[hit]] if len(same) else j[hit]
        col = np.empty((len(p), 3))
        col[hit] = rgb[src]
        if (~hit).any():
            _, j2 = tree_all.query(p[~hit])
            near = rgb[j2]
            col[~hit] = np.median(rgb[src], axis=0) if hit.any() else near
        P2.append(p); C2.append(col); L2.append(np.full(len(p), cat, dtype=cats.dtype))
    log(f"[lidar] solid furniture: {len(new_p)} object(s) at their measured size")
    return np.concatenate(P2), np.concatenate(C2), np.concatenate(L2)


# ----------------------------------------------------------------------------------- optional SegFormer

# What SegFormer may take from geometric labels: the things that sit on, or in, a wall, floor or ceiling.
SEG_MAY_TAKE = {"wall": {"window", "door", "painting", "sconce", "curtain", "mirror", "clock", "tv", "bookshelf",
                         "shelf"},
                "floor": {"rug"}, "ceiling": {"ceiling_light", "chandelier"}}


def merge_segformer(lab640, seg, seg_names):
    """SegFormer labels (our category names per pixel) fill in what geometry left unnamed."""
    out = lab640.copy()
    names = np.array(LABELS, dtype=object)[lab640]
    for cat in set(np.unique(seg_names).tolist()) - {"other", "wall", "floor", "ceiling"}:
        if cat not in L:
            continue
        m = seg_names == cat
        take = m & (names == "other")
        for geo, cats in SEG_MAY_TAKE.items():
            if cat in cats:
                take |= m & (names == geo)
        out[take] = L[cat]
    return out


# ----------------------------------------------------------------------------------- main

def to_views(frames, labels):
    """perception-style views: at depth resolution for fusing ("geo"), at photo resolution for wall textures."""
    geo_views, geo_labels, views, label_maps = [], [], [], []
    for fr, (lab, moved) in zip(frames, labels):
        dh, dw = fr["depth"].shape
        turns = upright_camera(np.eye(3), fr["c2w"], 1, 1)[0]   # the turns _upright gave this view
        valid = fr["valid"]
        # The labelled cloud: window pixels sit where their ray crossed the wall, not outdoors.
        Kd = fr["K"].copy(); Kd[:2] *= dw / fr["imageSize"][0]
        g = {"points": moved.astype(np.float32), "K": Kd, "cam2world": fr["c2w"].copy(),
             "conf": np.ones((dh, dw), np.float32), "mask": valid, "rgb": fr["rgb_d"]}
        geo_views.append(_upright(g)); geo_labels.append(np.ascontiguousarray(np.rot90(lab, turns)))

        H, W = fr["rgb"].shape[:2]
        s = W / fr["imageSize"][0]
        K = fr["K"].copy(); K[:2] *= s
        yi = np.minimum((np.arange(H) + 0.5) * dh / H, dh - 1).astype(int)
        xi = np.minimum((np.arange(W) + 0.5) * dw / W, dw - 1).astype(int)
        d = np.where(valid, fr["depth"], 0)[yi][:, xi]
        v = {"points": _unproject(d, K, fr["c2w"]).astype(np.float32), "K": K, "cam2world": fr["c2w"].copy(),
             "conf": np.ones((H, W), np.float32), "mask": valid[yi][:, xi], "rgb": fr["rgb"]}
        views.append(_upright(v)); label_maps.append(np.ascontiguousarray(np.rot90(lab[yi][:, xi], turns)))
    return geo_views, geo_labels, views, label_maps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="scan folder with capture.json (from the iOS app)")
    ap.add_argument("--out", type=Path, help="output plan (default: <images>/plan.json)")
    ap.add_argument("--mesh-out", type=Path, help="write measured coloured surface mesh as PLY")
    ap.add_argument("--mesh-max-side", type=int, default=0,
                    help="downsample each mesh view to this side length; 0 keeps every depth pixel")
    ap.add_argument("--mc-jar", help="Minecraft client jar, used to read real block texture colours")
    ap.add_argument("--scale", type=float, help="blocks per metre (default: automatic)")
    ap.add_argument("--known-height", type=float, help="the room's real floor-to-ceiling height in metres")
    ap.add_argument("--max-depth", type=float, default=FAR_DEPTH,
                    help=f"metres; high-confidence LiDAR depth is used up to this far (default {FAR_DEPTH})")
    ap.add_argument("--segformer", action="store_true",
                    help="also name small things (lamps, plants, paintings, TVs, rugs) with SegFormer; needs torch + "
                         "transformers and is much slower")
    ap.add_argument("--device", help="torch device for --segformer (default: cuda > mps > cpu)")
    ap.add_argument("--seg-model", default="nvidia/segformer-b5-finetuned-ade-640-640")
    # uniform: every wall block is a map in the room's one paint colour, with the things on the wall on top.
    ap.add_argument("--wall-detail", default="uniform", choices=["uniform", "style", "pattern", "items", "photo", "none"])
    ap.add_argument("--texture-px", type=int, help="texture pixels per block (default 128 for photo, 64 otherwise)")
    ap.add_argument("--save-cloud", action="store_true", help="also write the labelled point cloud (.npz)")
    ap.add_argument("--mirror-walls", action="store_true",
                    help="a wall the scan hardly saw copies the opposite wall (halls, libraries, churches)")
    # Accepted for compatibility with everything_cubed_analyzer.py's flags (config extraArgs); unused here.
    opts, unknown = ap.parse_known_args(argv)
    if unknown:
        log(f"[lidar] ignoring options that only apply to photo reconstruction: {' '.join(unknown)}")

    from palette import surface_palette
    from roomplan import ADE_TO_CATEGORY, apply_wall_features, build_plan

    t0 = time.time()
    cap, frames, room = load_scan(opts.images, opts.max_depth)
    log(f"[lidar] {len(frames)} posed photo(s) with depth from {cap.get('device', 'iPhone')}"
        + (f", RoomPlan model: {len(room.get('walls', []))} wall(s), {len(room.get('objects', []))} object(s), "
           f"{sum(len(room.get(k, [])) for k in ROOMPLAN_OPENINGS)} door/window/opening(s)" if room else ""))
    ex, ez = manhattan_axes(frames, room, log)
    floor_y, ceil_y = floor_ceiling(frames, room, log)
    walls = find_walls(frames, ex, ez, floor_y, ceil_y, log)
    walls.update(roomplan_walls(frames, room, ex, ez, log))
    tables = [t for t in find_tables(frames, floor_y, log) if not _under_roomplan_object(t, room, log)]
    labels = [label_frame(fr, ex, ez, floor_y, ceil_y, walls, room, tables) for fr in frames]
    geo_views, geo_labels, views, label_maps = to_views(frames, labels)
    if opts.mesh_out:
        from mesh_export import write_ply
        vertices, faces = write_ply(geo_views, opts.mesh_out, max_side=opts.mesh_max_side)
        log(f"[mesh] wrote {opts.mesh_out}: {vertices} vertices, {faces} triangles")
    del frames
    log(f"[lidar] labelled from geometry in {time.time() - t0:.1f}s")

    if opts.segformer:
        from perception import pick_device, segment
        ade_maps, ade_names = segment(views, pick_device(opts.device), opts.seg_model, log)
        ade_lut = category_lut(ade_names, ADE_TO_CATEGORY)
        label_maps = [merge_segformer(lm, a, ade_lut(a)) for lm, a in zip(label_maps, ade_maps)]
        for i, (g, lm) in enumerate(zip(geo_views, label_maps)):   # the cloud follows (nearest pixel)
            gh, gw = g["mask"].shape
            H, W = lm.shape
            yi = np.minimum(((np.arange(gh) + 0.5) * H / gh).astype(int), H - 1)
            xi = np.minimum(((np.arange(gw) + 0.5) * W / gw).astype(int), W - 1)
            geo_labels[i] = lm[yi][:, xi]

    pts, rgb, cats = fuse(geo_views, geo_labels, ID2LABEL, IDENTITY, conf_drop=0)
    log(f"[lidar] fused cloud: {len(pts)} points; " +
        ", ".join(f"{c} {int((cats == c).sum())}" for c in ("floor", "wall", "ceiling", "window") if (cats == c).any()))
    pts, rgb, cats = solid_objects(room, tables, floor_y, pts, rgb, cats)
    if opts.save_cloud:
        np.savez_compressed(opts.images / "cloud.npz", points=pts, colors=rgb, cats=cats)

    cache = Path(__file__).resolve().parent
    ctx = {}
    # LiDAR measures the room itself: built at its measured size (photo rooms are enlarged, see --enlarge).
    plan = build_plan(pts, rgb, cats, surface_palette(opts.mc_jar, cache), scale=opts.scale, metric=True, log=log,
                      ctx=ctx, known_height=opts.known_height, enlarge=1.0,
                      wall_ends=roomplan_wall_ends(room), seat_nudge=False)
    plan["source"] = {"photos": [v_name for v_name in (f.get("image") for f in cap["frames"]) if v_name],
                      "metric": True, "lidar": True, "roomplan": room is not None}
    # How ARKit world coordinates map onto the plan's blocks, for spectate (spectate_analyzer.PlanFrame): a point P
    # is at room metres loc = ((P - origin) @ R.T - [0, y_shift, 0]) * f, block x = (loc.x - xmin) / (xmax - xmin) * W,
    # z likewise, interior row y / hm * H (row 0 = first air layer). Spectate continues this scan's RoomPlan
    # session, so its live objects are in this scan's world coordinates.
    r = ctx["room"]
    plan["arkit"] = {"origin": [float(v) for v in ctx["origin"]], "R": [[float(v) for v in row] for row in ctx["R"]],
                     "y_shift": float(ctx["y_shift"]), "f": float(ctx["f"]),
                     "xmin": float(r.xmin), "xmax": float(r.xmax), "zmin": float(r.zmin), "zmax": float(r.zmax),
                     "hm": float(r.hm), "W": int(r.W), "D": int(r.D), "H": int(r.H)}
    out = opts.out or opts.images / "plan.json"
    if opts.wall_detail != "none":
        try:
            from walltex import render_walls
            from wallitems import wall_item_masks
            px = opts.texture_px or (128 if opts.wall_detail == "photo" else 64)
            tex = render_walls(views, label_maps, category_lut(ID2LABEL, IDENTITY), ctx, out.parent, px=px,
                               detail=opts.wall_detail, log=log,
                               surface_tol=WALL_SURFACE_TOL, item_masks=wall_item_masks(opts, views, None, log))
            plan["walls"] = tex.pop("wall_blocks")
            plan["wall_main"] = tex.pop("wall_main")
            plan["textures"] = tex
            apply_wall_features(plan, tex.pop("features"), log, mirror=opts.mirror_walls, tex_dir=out.parent,
                                windows=roomplan_window_cells(room, plan["arkit"]))
        except Exception as ex_:  # never lose the whole analysis over the wall pictures
            import traceback
            traceback.print_exc()
            log(f"[tex] wall pictures failed ({ex_}); building with plain wall blocks")
    out.write_text(json.dumps(plan, indent=1))
    from plan_viewer import write_viewer
    write_viewer(plan, out.with_name("view.html"), opts.mesh_out)
    log(f"[analyzer] wrote {out} in {time.time() - t0:.0f}s")
    log(f"[analyzer] 3D viewer: {out.with_name('view.html')}")


if __name__ == "__main__":
    main()

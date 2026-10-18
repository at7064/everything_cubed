"""Sanity test for roomplan.build_plan on a synthetic, arbitrarily rotated room point cloud.

Room: 4.0 m (x) by 3.2 m (z), 2.6 m high, wooden floor, blue walls with a green accent (east) wall,
the south wall never "photographed". Bed against the north wall, window on the east wall, table + chair.
Run: python test_synthetic.py
"""
import json

import numpy as np

from palette import Palette, builtin_surface_entries
from roomplan import build_plan

rng = np.random.default_rng(0)
pts, cols, cats = [], [], []


def add(n, lo, hi, rgb, cat, jitter=8):
    p = rng.uniform(lo, hi, size=(n, 3))
    pts.append(p)
    cols.append(np.clip(np.array(rgb) + rng.normal(0, jitter, (n, 3)), 0, 255))
    cats.extend([cat] * n)


W, D, H = 4.0, 3.2, 2.6
add(20000, (0, 0, 0), (W, 0, D), (165, 125, 80), "floor")
add(12000, (0, H, 0), (W, H, D), (240, 240, 236), "ceiling")
add(9000, (0, 0, 0), (W, H, 0), (150, 175, 205), "wall")          # north
add(8000, (0, 0, 0), (0, H, D), (150, 175, 205), "wall")          # west
add(8000, (W, 0, 0), (W, H, D), (70, 110, 60), "wall")            # east, accent
add(1500, (W, 0.9, 1.0), (W, 2.0, 2.2), (200, 220, 240), "window")
add(4000, (0.8, 0.0, 0.0), (2.4, 0.55, 2.0), (180, 30, 30), "bed")   # 1.6 m wide double bed, head north
add(1500, (2.6, 0.7, 2.0), (3.6, 0.75, 2.8), (110, 80, 50), "table")
add(600, (2.9, 0.0, 2.85), (3.3, 0.9, 3.15), (110, 80, 50), "chair")
add(400, (0.1, 0.0, 2.6), (0.4, 1.2, 2.9), (50, 140, 50), "plant")
P = np.concatenate(pts)

# Hide the room in an arbitrary camera frame: random rotation + translation + small noise.
a, b = 0.6, 0.25
Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
flip = np.diag([1, -1, -1.0])  # OpenCV-ish: y down (keeps right-handedness)
P_world = (P @ (flip @ Rx @ Ry).T) + np.array([0.3, 1.4, 2.0]) + rng.normal(0, 0.01, P.shape)

plan = build_plan(P_world, np.concatenate(cols), np.array(cats), Palette(builtin_surface_entries()))
# A two-view object estimate replaces the cloud cluster before block placement.
chair_start = sum(len(p) for p in pts[:8])
chair_world = P_world[chair_start:chair_start + len(pts[8])]
shift = np.array([0, 0, -0.6]) @ (flip @ Rx @ Ry).T
chair_match = {"category": "chair", "views": [0, 1], "points": chair_world + shift,
               "center": np.median(chair_world, axis=0) + shift, "colors": cols[8]}
matched_logs = []
matched_plan = build_plan(P_world, np.concatenate(cols), np.array(cats), Palette(builtin_surface_entries()),
                          object_instances=[chair_match], log=matched_logs.append)
assert any("1 object(s) located from matching detections" in m for m in matched_logs)
matched_chairs = [o for o in matched_plan["objects"] if o["type"] == "chair"]
assert len(matched_chairs) == 1 and matched_chairs[0]["source_views"] == [0, 1]
s = plan["size"]
print(json.dumps({k: plan[k] for k in ("scale", "metres", "size", "wall_main", "ceiling", "roof")}, indent=1))
print("floor:", plan["floor"]["palette"], "| walls:", plan["walls"]["north"]["palette"])
for side in ("north", "south", "west", "east"):
    g = np.array(plan["walls"][side]["grid"])
    print(side, [plan["walls"][side]["palette"][i] for i in np.unique(g)])
for o in plan["openings"]:
    print("opening", o)
for o in plan["objects"]:
    print("object ", o)

assert abs(plan["metres"]["x"] - W) < 0.35 or abs(plan["metres"]["x"] - D) < 0.35
assert abs(plan["metres"]["y"] - H) < 0.15
assert any(o["type"] == "door" for o in plan["openings"]), "a front door must always exist"
assert any(o["type"] == "window" for o in plan["openings"])
assert any(o["type"] == "bed" for o in plan["objects"])

# --enlarge: the same room at 1.0 and 1.5 - real metres unchanged, blocks about 1.5x; an explicit scale is exact.
small = build_plan(P_world, np.concatenate(cols), np.array(cats), Palette(builtin_surface_entries()), enlarge=1.0, log=lambda m: None)
big = build_plan(P_world, np.concatenate(cols), np.array(cats), Palette(builtin_surface_entries()), enlarge=1.5, log=lambda m: None)
# (base_scale: before the nudge that lets seats be centred on their tables; scale is stored rounded)
assert small["metres"] == big["metres"] and abs(big["base_scale"] / small["base_scale"] - 1.5) < 0.01
# Blocks: bigger in every direction (not exactly 1.5x, each scale may be nudged for its seating).
assert all(big["size"][k] > small["size"][k] for k in "xyz"), (small["size"], big["size"])
fixed = build_plan(P_world, np.concatenate(cols), np.array(cats), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.5, log=lambda m: None)
assert fixed["scale"] == 2.0
print("enlarge:", small["size"], "->", big["size"])
print("OK")

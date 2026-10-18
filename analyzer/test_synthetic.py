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
# Duplicates: two overlapping copies of one bed merge; two beds side by side stay two.
from roomplan import merge_duplicates  # noqa: E402
bed = lambda x0, z0: {"lo": np.array([x0, 0, z0]), "hi": np.array([x0 + 2.0, 0.5, z0 + 1.0]), "c": np.array([x0 + 1, 0.25, z0 + 0.5]),
                      "rgb": np.array([200.0, 50, 50]), "n": 500}
merged = merge_duplicates([("bed", bed(0, 0)), ("bed", bed(0.3, 0.2)), ("chair", bed(5, 5))], log=print)
assert sorted(c for c, _ in merged) == ["bed", "chair"] and abs(merged[0][1]["lo"][0] - 0.15) < 1e-9, merged
apart = merge_duplicates([("bed", bed(0, 0)), ("bed", bed(0, 1.1))], log=print)
assert len(apart) == 2, apart
print("duplicate merge OK")

# Measured ARKit planes: the cloud's east wall sits 0.5 m too far out (bad depth); a wall plane at the true x=4 and a
# ceiling plane at 2.6 m override it. The cloud here is in the room's own frame (no rotation).
pts2, cols2, cats2 = [], [], []
def add2(n, lo, hi, cat):
    pts2.append(rng.uniform(lo, hi, (n, 3))); cols2.append(np.full((n, 3), 180.0)); cats2.extend([cat] * n)
add2(20000, (0, 0, 0), (4, 0, 3), "floor"); add2(12000, (0, 2.9, 0), (4, 2.9, 3), "ceiling")
add2(8000, (0, 0, 0), (4, 2.9, 0), "wall"); add2(8000, (0, 0, 0), (0, 2.9, 3), "wall")
add2(8000, (4.5, 0, 0), (4.5, 2.9, 3), "wall"); add2(8000, (0, 0, 3), (4, 2.9, 3), "wall")
P2 = np.concatenate(pts2)
planes = [{"normal": [-1.0, 0, 0], "point": [4.0, 1.3, 1.5], "area": 6.0, "label": "wall"},
          {"normal": [0, -1.0, 0], "point": [2.0, 2.6, 1.5], "area": 8.0, "label": "ceiling"}]
plain = build_plan(P2, np.concatenate(cols2), np.array(cats2), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0, log=lambda m: None)
withp = build_plan(P2, np.concatenate(cols2), np.array(cats2), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0, log=print, planes=planes)
long_plain = max(plain["metres"]["x"], plain["metres"]["z"]); long_planes = max(withp["metres"]["x"], withp["metres"]["z"])
assert abs(long_plain - 4.5) < 0.15 and abs(long_planes - 4.0) < 0.15, (plain["metres"], withp["metres"])
assert abs(plain["metres"]["y"] - 2.9) < 0.1 and abs(withp["metres"]["y"] - 2.6) < 0.1, (plain["metres"], withp["metres"])
print("measured planes OK", plain["metres"], "->", withp["metres"])

# Raised beds and wall shelves, in a 6 m x 4 m room 2.6 m high, in its own frame, at 2 blocks/m (5 blocks high).
# West: a loft bed (mattress at 1.5 m, posts down to the floor) with a desk under it. East: a bunk bed (mattresses at
# 0.3 m and 1.5 m). Middle of the north wall: an ordinary bed. On the south wall at 1.5 m: a shelf with books.
pts3, cols3, cats3 = [], [], []
def add3(n, lo, hi, cat, rgb=(180, 180, 180)):
    pts3.append(rng.uniform(lo, hi, (n, 3))); cols3.append(np.full((n, 3), rgb, float)); cats3.extend([cat] * n)
add3(30000, (0, 0, 0), (6, 0, 4), "floor"); add3(15000, (0, 2.6, 0), (6, 2.6, 4), "ceiling")
for lo, hi in (((0, 0, 0), (6, 2.6, 0)), ((0, 0, 4), (6, 2.6, 4)), ((0, 0, 0), (0, 2.6, 4)), ((6, 0, 0), (6, 2.6, 4))):
    add3(9000, lo, hi, "wall")
add3(3000, (0.05, 1.45, 0.1), (1.0, 1.8, 2.1), "bed", (40, 60, 160))        # loft mattress and bedding
for x, z in ((0.05, 0.1), (0.95, 0.1), (0.05, 2.05), (0.95, 2.05)):
    add3(150, (x, 0, z), (x + 0.05, 1.45, z + 0.05), "bed", (40, 60, 160))  # its posts
add3(1500, (0.1, 0.7, 0.4), (0.9, 0.76, 1.6), "desk", (120, 90, 60))        # the desk under it
add3(600, (0.1, 0, 0.4), (0.9, 0.7, 0.5), "desk", (120, 90, 60))
for y0 in (0.25, 1.45):
    add3(2000, (5.0, y0, 0.1), (5.95, y0 + 0.35, 2.1), "bed", (160, 40, 40))  # bunk: two mattresses
add3(3000, (2.5, 0, 0.05), (3.5, 0.55, 2.05), "bed", (40, 140, 40))          # ordinary bed
add3(250, (2.4, 1.45, 3.75), (3.4, 1.75, 3.98), "shelf", (150, 110, 70))     # shelf board and books
P3 = np.concatenate(pts3)
rp = build_plan(P3, np.concatenate(cols3), np.array(cats3), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0,
                log=print)
beds = {o["color"]: o for o in rp["objects"] if o["type"] == "bed"}   # the plan may turn the room: tell beds by colour
assert {c: o.get("style") for c, o in beds.items()} == {"blue": "loft", "lime": None, "red": "bunk"}, beds
beds = list(beds.values())
assert all(o["upper"] == 3 for o in beds if o.get("style")), beds
loft = next(o for o in beds if o.get("style") == "loft")
desks = [o for o in rp["objects"] if o["type"] == "desk"]
assert len(desks) == 1, rp["objects"]
dk = desks[0]
under = loft["x"] <= dk["x"] < loft["x"] + loft["w"] and loft["z"] <= dk["z"] < loft["z"] + loft["d"]
assert under, ("the desk stands under the loft bed", loft, dk)
from roomplan import ladder_cells
for b in beds:
    if b.get("style"):
        for a, c in ladder_cells(b):
            assert not any(o["x"] <= a < o["x"] + o["w"] and o["z"] <= c < o["z"] + o["d"] for o in rp["objects"]
                           if o is not b), ("the ladder cell stays clear", b, (a, c))
shelves = [o for o in rp["openings"] if o["type"] == "shelf"]
assert len(shelves) == 1 and shelves[0]["y"] == 2 and shelves[0]["h"] == 1, shelves
assert not any(o["type"] == "shelf" for o in rp["objects"]), "a shelf hangs on its wall, it takes no floor space"
# bed_style alone: half the bed high and part of it low is a bunk, unless furniture stands under it (a loft whose
# posts and hanging bedding read as low); a mostly low bed is ordinary; a room under 4 blocks has no raised beds.
from roomplan import bed_style
box = lambda lo, hi, **k: {"lo": np.array(lo, float), "hi": np.array(hi, float), **k}
b = box((0, 0, 0), (1, 1.8, 2), raised=0.5, low=0.4)
desk = box((0.1, 0, 0.3), (0.9, 0.76, 1.5))
assert bed_style(b, [("bed", b)], 5) == "bunk"
assert bed_style(b, [("bed", b), ("desk", desk)], 5) == "loft"
assert bed_style(b, [("bed", b), ("desk", box((3, 0, 0), (4, 0.76, 1)))], 5) == "bunk", "a desk elsewhere"
assert bed_style(b, [("bed", b), ("plant", desk)], 5) == "bunk", "only furniture makes it a loft"
assert bed_style(dict(b, raised=0.1), [("bed", b)], 5) is None
assert bed_style(b, [("bed", b)], 4) is None, "no standing room on the upper bed"
# Second room, the harder cases. A loft whose lower part reads as bed too (bedding hanging down: a bunk by its
# heights) with a 1.3 m wardrobe under it: furniture under it makes it a loft, and the wardrobe is cut below its deck.
# A second, ordinary bed overlapping the loft by less than a duplicate must not be put in the loft's footprint.
# Floor-standing shelving labelled "shelf" becomes a bookshelf, not a board on the wall.
pts3, cols3, cats3 = [], [], []
add3(30000, (0, 0, 0), (6, 0, 4), "floor"); add3(15000, (0, 2.6, 0), (6, 2.6, 4), "ceiling")
for lo, hi in (((0, 0, 0), (6, 2.6, 0)), ((0, 0, 4), (6, 2.6, 4)), ((0, 0, 0), (0, 2.6, 4)), ((6, 0, 0), (6, 2.6, 4))):
    add3(9000, lo, hi, "wall")
add3(3000, (0.05, 1.45, 0.1), (1.0, 1.8, 2.1), "bed", (40, 60, 160))
add3(1500, (0.05, 0.0, 0.1), (1.0, 0.6, 0.4), "bed", (40, 60, 160))
add3(2000, (0.1, 0, 0.15), (0.9, 1.3, 0.95), "wardrobe", (120, 90, 60))  # at the head end, inside the built 2-block bed
add3(3000, (0.55, 0, 0.6), (1.5, 0.55, 2.6), "bed", (40, 140, 40))
add3(1500, (4.0, 0, 3.6), (5.0, 1.6, 3.98), "shelf", (150, 110, 70))
P3 = np.concatenate(pts3)
rp2 = build_plan(P3, np.concatenate(cols3), np.array(cats3), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0,
                 log=print)
beds2 = [o for o in rp2["objects"] if o["type"] == "bed"]
assert sorted((o["color"], o.get("style")) for o in beds2) == [("blue", "loft"), ("lime", None)], beds2
a, b = beds2
assert not (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and a["z"] < b["z"] + b["d"] and b["z"] < a["z"] + a["d"]), \
    ("two beds in one footprint", beds2)
loft2 = next(o for o in beds2 if o.get("style") == "loft")
wd = [o for o in rp2["objects"] if o["type"] == "wardrobe"]
assert len(wd) == 1, rp2["objects"]
wd_under = wd[0]["x"] < loft2["x"] + loft2["w"] and loft2["x"] < wd[0]["x"] + wd[0]["w"] and \
    wd[0]["z"] < loft2["z"] + loft2["d"] and loft2["z"] < wd[0]["z"] + wd[0]["d"]
assert wd_under and wd[0]["h"] <= loft2["upper"] - 1, ("the wardrobe under the loft stays below its deck", wd, loft2)
W2, D2 = rp2["size"]["x"], rp2["size"]["z"]
assert all(0 <= x < W2 and 0 <= z < D2 for x, z in ladder_cells(loft2)), ("ladder inside the room", loft2)
assert any(o["type"] == "bookshelf" for o in rp2["objects"]) and not any(o["type"] == "shelf" for o in rp2["openings"]), \
    (rp2["objects"], rp2["openings"])

# A loft (mattress only) and an ordinary bed against the same wall, overlapping by less than a duplicate: two beds,
# side by side. (Rounding to blocks already keeps them apart here, so this does not exercise the loft deferral.)
pts3, cols3, cats3 = [], [], []
add3(30000, (0, 0, 0), (6, 0, 4), "floor"); add3(15000, (0, 2.6, 0), (6, 2.6, 4), "ceiling")
for lo, hi in (((0, 0, 0), (6, 2.6, 0)), ((0, 0, 4), (6, 2.6, 4)), ((0, 0, 0), (0, 2.6, 4)), ((6, 0, 0), (6, 2.6, 4))):
    add3(9000, lo, hi, "wall")
add3(3000, (0.05, 1.45, 0.1), (1.0, 1.8, 2.1), "bed", (40, 60, 160))
add3(3000, (0.55, 0, 0.1), (1.5, 0.55, 2.1), "bed", (40, 140, 40))
P3 = np.concatenate(pts3)
rp3 = build_plan(P3, np.concatenate(cols3), np.array(cats3), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0,
                 log=print)
beds3 = [o for o in rp3["objects"] if o["type"] == "bed"]
assert sorted(o.get("style") or "" for o in beds3) == ["", "loft"], beds3
a, b = beds3
assert not (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and a["z"] < b["z"] + b["d"] and b["z"] < a["z"] + a["d"]), \
    ("two beds in one footprint", beds3)

# The Lander scan's second loft: a bed set at a medium height (mattress about 1 m up, little of it below 0.7 m)
# with its end frame 15 cm away seen as a separate thin sliver of "bed". One loft, not a bed plus a 1-block loft.
from roomplan import absorb_fragments
assert bed_style(box((0, 0, 0), (2, 1.6, 1), raised=0.22, low=0.21), [], 5) == "loft", "raised by having little low"
pts3, cols3, cats3 = [], [], []
add3(30000, (0, 0, 0), (6, 0, 4), "floor"); add3(15000, (0, 2.6, 0), (6, 2.6, 4), "ceiling")
for lo, hi in (((0, 0, 0), (6, 2.6, 0)), ((0, 0, 4), (6, 2.6, 4)), ((0, 0, 0), (0, 2.6, 4)), ((6, 0, 0), (6, 2.6, 4))):
    add3(9000, lo, hi, "wall")
add3(3000, (0.3, 0.75, 0.05), (2.4, 1.3, 1.1), "bed", (90, 90, 160))     # mattress and bedding at a medium height
add3(500, (0.3, 0.45, 0.05), (2.4, 0.7, 1.1), "bed", (90, 90, 160))      # its frame below
add3(700, (0.0, 0.5, 0.05), (0.15, 1.7, 1.05), "bed", (90, 90, 160))     # the end frame, a thin sliver
P3 = np.concatenate(pts3)
logs = []
rp5 = build_plan(P3, np.concatenate(cols3), np.array(cats3), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0,
                 log=logs.append)
beds5 = [o for o in rp5["objects"] if o["type"] == "bed"]
assert len(beds5) == 1 and beds5[0].get("style") == "loft", (beds5, [m for m in logs if "bed" in m])
assert any("1 thin piece(s) joined" in m for m in logs), logs
# A sliver with no bed beside it is dropped; a whole bed is never treated as a piece.
lone = absorb_fragments([("bed", box((0, 0, 0), (0.2, 1, 2), n=100, raised=0.5, low=0.1)),
                         ("bed", box((3, 0, 0), (5, 0.6, 1), n=900, raised=0.0, low=1.0))], log=lambda m: None)
assert len(lone) == 1 and lone[0][1]["lo"][0] == 3, lone

# Posters and cabinets: a 2.2 m wide poster mask is built at most PAINTING_MAX_M a side; a 1.0 m cabinet is one block
# (level with the desks), a 1.8 m one stays tall.
from roomplan import PAINTING_MAX_M
pts3, cols3, cats3 = [], [], []
add3(30000, (0, 0, 0), (6, 0, 4), "floor"); add3(15000, (0, 2.6, 0), (6, 2.6, 4), "ceiling")
for lo, hi in (((0, 0, 0), (6, 2.6, 0)), ((0, 0, 4), (6, 2.6, 4)), ((0, 0, 0), (0, 2.6, 4)), ((6, 0, 0), (6, 2.6, 4))):
    add3(9000, lo, hi, "wall")
add3(1500, (1.5, 1.0, 0.0), (3.7, 2.2, 0.03), "painting", (200, 60, 120))
add3(1500, (4.5, 0, 3.5), (5.5, 1.0, 3.95), "cabinet", (150, 80, 40))
add3(1500, (0.05, 0, 1.5), (0.5, 1.8, 2.5), "cabinet", (60, 40, 30))
P3 = np.concatenate(pts3)
rp4 = build_plan(P3, np.concatenate(cols3), np.array(cats3), Palette(builtin_surface_entries()), scale=2.0, enlarge=1.0,
                 log=lambda m: None)
pa = [o for o in rp4["openings"] if o["type"] == "painting"]
assert len(pa) == 1 and pa[0]["w"] <= round(PAINTING_MAX_M * 2) and pa[0]["h"] <= round(PAINTING_MAX_M * 2), pa
assert sorted(o["h"] for o in rp4["objects"] if o["type"] == "cabinet") == [1, 2], rp4["objects"]

# A RoomPlan bed box 1.2 m high (its headboard), sampled down its sides the way lidar_analyzer does: an ordinary bed.
from lidar_analyzer import _box_surface
from roomplan import heights
T = np.eye(4); T[:3, 3] = [1, 0.6, 1]
box_pts = _box_surface(T, (1.6, 1.2, 2.0), 0.0)
assert bed_style(dict(box((0.2, 0, 0), (1.8, 1.2, 2)), **heights(box_pts)), [], 5) is None, heights(box_pts)
print("raised beds + wall shelf OK:", [(o["x"], o["z"], o.get("style")) for o in beds], shelves[0])
print("OK")

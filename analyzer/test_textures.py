"""Checks photo-texture projection (walltex.py) on ray-traced synthetic photos.

The walls are painted with a colour gradient that encodes the world position, so every texture pixel
can be checked against where it should have come from. Also checks that textures are not mirrored:
image columns must run left-to-right as seen from inside the room.
Run: python test_textures.py
"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from palette import WALL_COLOURS, Palette, builtin_surface_entries
from perception import category_lut, fuse
from roomplan import ADE_TO_CATEGORY, build_plan
from walltex import render_walls, wall_pixels

W, H, D = 4.0, 2.5, 3.0
ID2LABEL = {0: "wall", 1: "floor", 2: "ceiling"}


def paint(p):
    return np.stack([p[:, 0] / W, p[:, 1] / H, p[:, 2] / D], 1) * 220 + 20


def render(cam_pos, yaw, size=(240, 320), f=110.0, paint=paint):
    """Ray-trace the inside of the box [0,W]x[0,H]x[0,D] (y up) from a pinhole camera (OpenCV axes)."""
    h, w = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    fwd = np.array([np.sin(yaw), 0, -np.cos(yaw)])
    right = np.cross(fwd, [0, 1, 0]); down = np.array([0, -1.0, 0])
    c2w = np.eye(4); c2w[:3, :3] = np.stack([right, down, fwd], 1); c2w[:3, 3] = cam_pos
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u, float)], -1).reshape(-1, 3) @ c2w[:3, :3].T
    t_best = np.full(len(rays), np.inf); lab = np.zeros(len(rays), int)
    for axis, val, label in ((0, 0, 0), (0, W, 0), (2, 0, 0), (2, D, 0), (1, 0, 1), (1, H, 2)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - cam_pos[axis]) / rays[:, axis]
        t = np.where(t > 0, t, np.inf)
        better = t < t_best
        t_best[better] = t[better]; lab[better] = label
    pts = cam_pos + rays * t_best[:, None]
    rgb = paint(pts).clip(0, 255).astype(np.uint8)
    return {"points": pts.reshape(h, w, 3), "conf": np.ones((h, w)), "mask": np.ones((h, w), bool),
            "rgb": rgb.reshape(h, w, 3), "K": K, "cam2world": c2w}, lab.reshape(h, w)


# Four photos, each taken from one wall looking across at the opposite wall.
views, labels = zip(*[render(np.array([2.0, 1.4, 2.9]), 0.0), render(np.array([2.0, 1.4, 0.1]), np.pi),
                      render(np.array([0.1, 1.4, 1.5]), np.pi * 0.5), render(np.array([3.9, 1.4, 1.5]), -np.pi * 0.5)])
pts, rgb, cats = fuse(views, labels, ID2LABEL, ADE_TO_CATEGORY, voxel=0.03)
ctx = {}
plan = build_plan(pts, rgb, cats, Palette(builtin_surface_entries()), ctx=ctx, log=lambda m: None)

out = Path(tempfile.mkdtemp())
tex = render_walls(views, labels, category_lut(ID2LABEL, ADE_TO_CATEGORY), ctx, out, px=16, detail="photo", log=print)
room, R, origin, f, ys = ctx["room"], ctx["R"], ctx["origin"], ctx["f"], ctx["y_shift"]
to_world = lambda loc: origin + (loc / f + np.array([0, ys, 0])) @ R
up_world = R[1]
for side, info in tex["walls"].items():
    img = np.asarray(Image.open(out / info["image"]), float).reshape(-1, 3)
    loc, n_room, shape = wall_pixels(side, room, 16)
    P = to_world(loc)
    expected = paint(P)
    tiles = np.array(info["tiles"])
    err = np.abs(img - expected).mean(1).reshape(shape)
    err_tiles = err.reshape(room.H, 16, -1, 16).mean(axis=(1, 3))[::-1]  # [row][col] like "tiles", unflipped
    if info["flip"]:
        err_tiles = err_tiles[:, ::-1]
    covered = err_tiles[tiles.astype(bool)]
    print(f"{side}: colour error in textured blocks {covered.mean():.1f}/255, tiles {tiles.sum()}/{tiles.size}")
    assert tiles.mean() > 0.8 and covered.mean() < 8, side
    # Not mirrored: columns advance towards the viewer's right (viewer faces the wall from inside).
    Pg = P.reshape(shape + (3,))
    facing = -(n_room @ R)
    right = np.cross(facing, up_world)
    assert (Pg[0, -1] - Pg[0, 0]) @ right > 0, f"{side} texture is mirrored"
    assert (Pg[0, 0] - Pg[-1, 0]) @ up_world > 0, f"{side} texture is upside down"
assert len(tex["walls"]) == 4
print("photo mode OK")

# --- items mode: plain, unevenly lit paint with one red poster on the z=0 wall.
rng = np.random.default_rng(1)
POSTER = (1.2, 2.0, 1.0, 1.8)  # x0, x1, y0, y1 on the z=0 wall


def paint_plain(p):
    shade = 1.0 - 0.18 * p[:, 1] / H - 0.08 * np.abs(p[:, 0] - W / 2) / W  # soft lighting falloff
    c = np.array([205.0, 195, 175]) * shade[:, None] + rng.normal(0, 4, (len(p), 3))
    on = (np.abs(p[:, 2]) < 1e-6) & (p[:, 0] > POSTER[0]) & (p[:, 0] < POSTER[1]) & (p[:, 1] > POSTER[2]) & (p[:, 1] < POSTER[3])
    c[on] = [180, 30, 35]
    return c


views, labels = zip(*[render(np.array([2.0, 1.4, 2.9]), 0.0, paint=paint_plain),
                      render(np.array([2.0, 1.4, 0.1]), np.pi, paint=paint_plain),
                      render(np.array([0.1, 1.4, 1.5]), np.pi * 0.5, paint=paint_plain),
                      render(np.array([3.9, 1.4, 1.5]), -np.pi * 0.5, paint=paint_plain)])
pts, rgb, cats = fuse(views, labels, ID2LABEL, ADE_TO_CATEGORY, voxel=0.03)
ctx = {}
plan = build_plan(pts, rgb, cats, Palette(builtin_surface_entries()), ctx=ctx, log=lambda m: None)
for side in ("north", "south", "west", "east"):
    assert len(plan["walls"][side]["palette"]) == 1 or len(set(sum(plan["walls"][side]["grid"], []))) == 1, \
        f"{side}: plain wall should be one block type"
out = Path(tempfile.mkdtemp())
tex = render_walls(views, labels, category_lut(ID2LABEL, ADE_TO_CATEGORY), ctx, out, px=32, log=print)
assert tex["dither"] is False
block = tex["wall_main"]
assert block in WALL_COLOURS
for side, wb in tex["wall_blocks"].items():
    assert {wb["palette"][i] for row in wb["grid"] for i in row} == {block}, f"{side}: every wall block = paint block"
print(f"paint block {block} on every wall block")
assert list(tex["walls"]) and len(tex["walls"]) == 1, f"only the poster wall gets frames, got {list(tex['walls'])}"
(side, info), = tex["walls"].items()
tiles = np.array(info["tiles"]).astype(bool)
img = np.asarray(Image.open(out / info["image"])).astype(float)
room = ctx["room"]
rows, cols = room.H, room.wall_len(side)
img_tiles = tiles[::-1]
if info["flip"]:
    img_tiles = img_tiles[:, ::-1]
area = np.repeat(np.repeat(img_tiles, 32, 0), 32, 1)
framed = img[area]
print(f"{side}: {tiles.sum()} framed blocks, mean colour {framed.mean(0).round()}, colours {len(np.unique(framed, axis=0))}")
# Poster covers x 1.2-2.0 m, y 1.0-1.8 m at ~1.67 blocks/m: about 1-2 x 1-2 blocks.
assert 1 <= tiles.sum() <= 4
assert np.abs(framed.mean(0) - [180, 30, 35]).max() < 15, "framed blocks must be all poster, no paint"
print("items mode OK")

# --- one photo only: unseen walls are still the same paint block, no frames where no items.
out = Path(tempfile.mkdtemp())
tex = render_walls(views[:1], labels[:1], category_lut(ID2LABEL, ADE_TO_CATEGORY), ctx, out, px=32, log=print)
for side, wb in tex["wall_blocks"].items():
    assert {wb["palette"][i] for row in wb["grid"] for i in row} == {tex["wall_main"]}
assert set(tex["walls"]) <= {side}
print("unseen walls OK")

# A hall (30 x 16 m, 12 m high) seen from far away, where the reconstruction's depth is off by up to 4%
# (what Suzzallo's walls showed at 10-20 m). A fixed 24 cm depth tolerance rejected nearly all of it; the
# tolerance that grows with distance must keep most of each wall without accepting surfaces far off it.
import walltex  # noqa: E402

W, H, D = 30.0, 12.0, 16.0
hall = [render(np.array([15.0, 1.6, 15.5]), 0.0, f=150.0), render(np.array([15.0, 1.6, 0.5]), np.pi, f=150.0)]
for v, _ in hall:
    C = v["cam2world"][:3, 3]
    ripple = 1 + 0.04 * np.sin(np.linspace(0, 6 * np.pi, v["points"].shape[1]))[None, :, None]
    v["points"] = C + (v["points"] - C) * ripple
hv, hl = zip(*hall)
hp, hr, hc = fuse(hv, hl, ID2LABEL, ADE_TO_CATEGORY, voxel=0.06)
hctx = {}
build_plan(hp, hr, hc, Palette(builtin_surface_entries()), ctx=hctx, log=lambda m: None)
hroom, hR, horigin, hf, hys = hctx["room"], hctx["R"], hctx["origin"], hctx["f"], hctx["y_shift"]
h_to_world = lambda loc: horigin + (loc / hf + np.array([0, hys, 0])) @ hR
lut = category_lut(ID2LABEL, ADE_TO_CATEGORY)
far_wall = max(("north", "south", "west", "east"), key=lambda s: hroom.wall_len(s))  # the frame may swap axes
cover = {rel: walltex.project_wall(far_wall, hroom, h_to_world, hR, hv, hl, lut, 4, 0.12 / hf, rel_tol=rel)[1].mean()
         for rel in (0.0, walltex.REL_TOL)}
print(f"far wall seen: fixed 24 cm {cover[0.0]:.0%}, relative {cover[walltex.REL_TOL]:.0%}")
assert cover[walltex.REL_TOL] > 0.5 and cover[walltex.REL_TOL] > 2 * cover[0.0], cover
# ... but a surface 25% off the wall (something in front of it) is still rejected.
for v, _ in hall:
    C = v["cam2world"][:3, 3]
    v["points"] = C + (v["points"] - C) * 0.75
off = walltex.project_wall(far_wall, hroom, h_to_world, hR, hv, hl, lut, 4, 0.12 / hf)[1].mean()
assert off < 0.05, off
print("relative depth tolerance OK")

# Pattern mode: a plain wall is judged plain, a striped one patterned; filling keeps every photo pixel and
# continues the stripes with their own two colours instead of averaging them into mush.
px = 16
yy, xx = np.mgrid[0:6 * px, 0:12 * px]
cA, cB = np.array([170, 120, 80.0]), np.array([225, 210, 180.0])
stripes = np.where(((xx // (px // 2)) % 2 == 0)[..., None], cA, cB) + np.random.default_rng(1).normal(0, 3, xx.shape + (3,))
plain = np.full(xx.shape + (3,), 200.0) + np.random.default_rng(2).normal(0, 3, xx.shape + (3,))
seen = xx < 5 * px                                      # the photos saw the left 5 of 12 blocks
paint_lab = walltex.srgb_to_lab(np.array([[200.0, 200, 200], [190, 160, 120]]))
off_p, grain_p = walltex.wall_pattern_stats(plain, seen, paint_lab, px)
off_s, grain_s = walltex.wall_pattern_stats(stripes, seen, paint_lab, px)
print(f"plain: off {off_p:.0%} grain {grain_p:.1f} | stripes: off {off_s:.0%} grain {grain_s:.1f}")
assert not (off_p >= walltex.PATTERN_OFF or grain_p >= walltex.PATTERN_GRAIN), "plain wall must stay plain"
assert off_s >= walltex.PATTERN_OFF or grain_s >= walltex.PATTERN_GRAIN, "striped wall must be patterned"
rng = np.random.default_rng(0)
pool = walltex._exemplars(stripes, seen, px, 1, 4, 500, rng)
filled = walltex.synthesize_wall(np.where(seen[..., None], stripes, 0), seen, pool, px, rng)
assert np.array_equal(filled[seen], stripes[seen].astype(np.float32)), "photo pixels must not change"
near = np.minimum(np.abs(filled[~seen] - cA).max(-1), np.abs(filled[~seen] - cB).max(-1))
share_a = (np.abs(filled[~seen] - cA).max(-1) < 30).mean()
print(f"filled: {np.mean(near < 30):.0%} of pixels are one of the two stripe colours, {share_a:.0%} the darker one")
assert np.mean(near < 30) > 0.85 and 0.3 < share_a < 0.7
print("pattern walls OK")

# Viewing angle and excluded pixels: the cosine of the chosen view stays in (0.2, 1]; pixels masked as
# window or door in every view are never taken as wall.
side0 = "north"
img0, seen0, cos0 = walltex.project_wall(side0, room, to_world, R, views, labels, category_lut(ID2LABEL, ADE_TO_CATEGORY), 8, 0.12 / f)
mid = cos0[cos0.shape[0] // 2, cos0.shape[1] // 2]
print(f"{side0}: seen {seen0.mean():.0%}, head-on cosine at the centre {mid:.2f}, min seen {cos0[seen0].min():.2f}")
assert seen0.mean() > 0.9 and cos0[seen0].min() > 0.2 and (cos0 <= 1.0001).all() and cos0.max() > 0.8
masks = [np.ones(v["rgb"].shape[:2], bool) for v in views]   # every view: "all of this is a window"
_, seen_x, _ = walltex.project_wall(side0, room, to_world, R, views, labels, category_lut(ID2LABEL, ADE_TO_CATEGORY), 8, 0.12 / f, exclude=masks)
assert seen_x.mean() == 0, seen_x.mean()
print("view angle + exclusion OK")

# Wall features. Projection: the small room's walls are all seen, no window or shelf anywhere.
from roomplan import apply_wall_features  # noqa: E402
feat = walltex.project_features("north", room, to_world, R, views, labels, category_lut(ID2LABEL, ADE_TO_CATEGORY), f)
assert feat["seen"].mean() > 0.9 and feat["window"].max() == 0 and feat["bookshelf"].max() == 0, feat["seen"].mean()
# One view mislabeling the whole wall as glass cannot overrule two agreeing views.
window_labels = [np.full_like(lm, 3) for lm in labels]
window_lut = category_lut({**ID2LABEL, 3: "window"}, ADE_TO_CATEGORY)
agreed = walltex.project_features("north", room, to_world, R, views * 3,
                                  list(labels) + list(labels) + window_labels, window_lut, f)
assert agreed["window"].max() < 0.34, agreed["window"].max()

# Decisions, on a hand-made 6 x 10 wall plan.
rows, cols = 6, 10
def fplan():
    return {"size": {"x": cols, "z": 8, "y": rows},
            "openings": [{"type": "window", "side": "north", "u": 0, "w": 10, "y": 0, "h": 6},   # a "whole wall" guess
                         {"type": "window", "side": "south", "u": 0, "w": 10, "y": 1, "h": 4},
                         {"type": "door", "side": "south", "u": 4, "w": 1, "y": 0, "h": 2}],
            "objects": [{"type": "bookshelf", "x": 2, "z": 0, "w": 3, "d": 1, "h": 2},
                        {"type": "bookshelf", "x": 2, "z": 4, "w": 3, "d": 1, "h": 2}],
            "walls": {s: {"palette": ["minecraft:white_terracotta"], "grid": [[0] * cols for _ in range(rows)]}
                      for s in ("north", "south")},
            "textures": {"walls": {"north": {"image": "n.png", "flip": False, "tiles": [[1] * cols for _ in range(rows)]}}}}
seen_all, none = np.ones((rows, cols)), np.zeros((rows, cols))
glass = none.copy(); glass[2:5, 1:3] = 1; glass[2:5, 6:8] = 1            # two windows with stone between
shelves = none.copy(); shelves[0:2, 2:5] = 1                              # a shelf band at the bottom
p1 = fplan()
apply_wall_features(p1, {"north": {"seen": seen_all, "window": glass, "bookshelf": shelves},
                         "south": {"seen": none, "window": none, "bookshelf": none}}, log=print)
nw = [o for o in p1["openings"] if o["side"] == "north" and o["type"] == "window"]
assert sorted((o["u"], o["w"], o["y"], o["h"]) for o in nw) == [(1, 2, 2, 3), (6, 2, 2, 3)], nw
sw = _grid = np.zeros((rows, cols), bool)
for o in p1["openings"]:
    if o["side"] == "south" and o["type"] == "window":
        sw[o["y"]:o["y"] + o["h"], o["u"]:o["u"] + o["w"]] = True
assert sw.mean() <= 0.6 and not sw[:, 3].any() and not sw[:2, 4].any(), sw.astype(int)   # piers; no glass in the door
g = np.array(p1["walls"]["north"]["grid"]); b = p1["walls"]["north"]["palette"].index("minecraft:bookshelf")
assert (g[0:2, 2:5] == b).all() and (g[2:] != b).all()
assert np.array(p1["textures"]["walls"]["north"]["tiles"])[0:2, 2:5].sum() == 0, "no picture over real shelves"
assert [o["z"] for o in p1["objects"]] == [4], "the bookcase at the shelf wall is a duplicate; the other stays"

p2 = fplan()
apply_wall_features(p2, {"north": {"seen": seen_all, "window": glass, "bookshelf": shelves},
                         "south": {"seen": none, "window": none, "bookshelf": none}}, log=print, mirror=True)
sw2 = sorted((o["u"], o["w"], o["y"], o["h"]) for o in p2["openings"] if o["side"] == "south" and o["type"] == "window")
assert sw2 == [(1, 2, 2, 3), (6, 2, 2, 3)], sw2
assert p2["textures"]["walls"]["south"]["image"] == "n.png" and p2["walls"]["south"]["grid"] == p2["walls"]["north"]["grid"]
print("wall features OK")

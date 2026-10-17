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

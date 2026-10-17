"""Photographic wall textures: projects each wall plane back into the photos (using the reconstructed
cameras) and resamples it into a rectified image, 1 tile = 1 wall block. The mod turns each tile into a
locked map in an invisible item frame.

detail="items" (default): the paint is real blocks - one of the 33 wall blocks, found jointly over all walls
so the same paint is the same block everywhere (unseen walls inferred). Item frames lit per entity can never
blend with smoothly lit blocks, so frames are only used for the things on the wall (posters, boards,
clocks...): snapped to whole blocks, in a few flat colours, filling their frames completely.
detail="photo": the actual photo on every block (paint where unseen).

For every texture pixel the most head-on, closest photo is used, skipping photos where something
(furniture, a person) stands between the camera and the wall (depth test + segmentation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

from palette import WALL_COLOURS, WALL_PALETTE, srgb_to_lab

# Pixels with these labels are never wall surface: things in front of the wall, and windows and doors
# (built as real openings; a window at night would otherwise become black "paint").
NOT_WALL = {"bed", "table", "desk", "coffee_table", "counter", "cabinet", "wardrobe", "bookshelf", "chair",
            "sofa", "lamp", "plant", "fridge", "stove", "sink", "bathtub", "toilet", "storage", "trash",
            "washer", "microwave", "rug", "floor", "ceiling", "ceiling_light", "window", "door"}
# Viewer-left to viewer-right along each wall as seen from inside the room (see roomplan.py axes).
FLIP = {"north": False, "south": True, "west": True, "east": False}


def _bilinear(img, u, v):
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    i = img.astype(np.float32)
    return (i[v0, u0] * (1 - du) * (1 - dv) + i[v0, u0 + 1] * du * (1 - dv)
            + i[v0 + 1, u0] * (1 - du) * dv + i[v0 + 1, u0 + 1] * du * dv)


def wall_pixels(side, room, px):
    """Room-frame (x, y, z) of every texture pixel centre, image row 0 = top of the wall."""
    L = room.wall_len(side)
    cols, rows = L * px, room.H * px
    t = (np.arange(cols) + 0.5) / cols
    if FLIP[side]:
        t = 1 - t
    y = room.hm * (1 - (np.arange(rows) + 0.5) / rows)
    T, Y = np.meshgrid(t, y)
    if side in ("north", "south"):
        X = room.xmin + T * (room.xmax - room.xmin)
        Z = np.full_like(X, room.zmin if side == "north" else room.zmax)
        normal = np.array([0, 0, 1.0 if side == "north" else -1.0])  # pointing into the room
    else:
        Z = room.zmin + T * (room.zmax - room.zmin)
        X = np.full_like(Z, room.xmin if side == "west" else room.xmax)
        normal = np.array([1.0 if side == "west" else -1.0, 0, 0])
    return np.stack([X, Y, Z], -1).reshape(-1, 3), normal, (rows, cols)


PAINT_DE = 22.0     # two paints must differ at least this much (CIELAB, lightness at half weight)
PAINT_SHARE = 0.25  # ... and each cover at least this share of the seen wall area
ITEM_DE = 18.0      # a pixel this far from its paint is part of something on the wall


def _de(lab, ref):
    d = lab - ref
    d[..., 0] *= 0.5  # shading and soft shadows change lightness, not paint
    return np.sqrt((d ** 2).sum(-1))


def project_wall(side, room, to_world, R, views, label_maps, cat_names_of_pixel, px, tol):
    """Resample one wall from the photos. Returns (rgb image, seen mask)."""
    loc, n_room, shape = wall_pixels(side, room, px)
    P = to_world(loc)
    n_world = n_room @ R
    best = np.full(len(P), -np.inf)
    rgb = np.zeros((len(P), 3), np.float32)
    for v, lm in zip(views, label_maps):
        if "K" not in v:
            continue
        c2w = v["cam2world"]
        w2c = np.linalg.inv(c2w)
        Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = Pc[:, 2]
        h, w = v["rgb"].shape[:2]
        ok = z > 1e-3
        K = v["K"]
        u = np.where(ok, K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2], -1)
        vv = np.where(ok, K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2], -1)
        ok &= (u >= 0) & (u < w - 1) & (vv >= 0) & (vv < h - 1)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        ui, vi = np.round(u[idx]).astype(int), np.round(vv[idx]).astype(int)
        C = c2w[:3, 3]
        d_wall = np.linalg.norm(P[idx] - C, axis=1)
        d_obs = np.linalg.norm(v["points"][vi, ui] - C, axis=1)
        # The photo must show a surface *on* the wall here: not something in front of it, and not
        # something further away seen past it (open-plan rooms, an inferred wall).
        visible = v["mask"][vi, ui] & np.isfinite(d_obs) & (np.abs(d_obs - d_wall) < 2 * tol)
        visible &= ~np.isin(cat_names_of_pixel(lm[vi, ui]), list(NOT_WALL))
        cos = np.abs(((P[idx] - C) / d_wall[:, None]) @ n_world)
        score = np.where(visible & (cos > 0.2), cos / d_wall, -np.inf)
        better = score > best[idx]
        sel = idx[better]
        best[sel] = score[better]
        rgb[sel] = _bilinear(v["rgb"], u[sel], vv[sel])
    return rgb.reshape(shape + (3,)), np.isfinite(best).reshape(shape)


def find_paints(walls_seen, fallback_rgb):
    """Paint colours shared by all walls: 1 colour, or more only for big, clearly different areas.

    walls_seen: list of (rgb image, seen mask). Returns (paints rgb (k,3), paints lab (k,3), block ids).
    """
    pix = np.concatenate([img[seen] for img, seen in walls_seen]) if walls_seen else np.zeros((0, 3))
    if len(pix) < 50:
        return _snap([np.array(fallback_rgb, float)])
    pix = pix[:: max(1, len(pix) // 40000)]
    lab = srgb_to_lab(pix)
    best = [np.median(pix, axis=0)]
    for k in (2, 3):
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab)
        share = np.bincount(km.labels_, minlength=k) / len(lab)
        c = km.cluster_centers_
        if share.min() >= PAINT_SHARE and all(_de(c[i], c[j]) >= PAINT_DE for i in range(k) for j in range(i + 1, k)):
            best = [np.median(pix[km.labels_ == i], axis=0) for i in range(k)]
    return _snap(best)


def _snap(colours):
    """Each paint becomes the closest wall block, drawn in exactly that block's colour.

    Returns (draw rgb per paint, measured lab per paint - used to tell paint from items -, block per paint).
    Two paints that snap to the same block are kept apart for detection but drawn identically."""
    blocks = [WALL_PALETTE.nearest(c) for c in colours]
    rgb = np.array([WALL_COLOURS[b] for b in blocks], float)
    return rgb, srgb_to_lab(np.array(colours, float)), blocks


def posterise(pix, max_colours=6):
    """A few flat colours for the things on the wall (near-identical colours and edge slivers merged)."""
    k = int(min(max_colours, max(1, len(pix) // 200)))
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(srgb_to_lab(pix[:: max(1, len(pix) // 20000)]))
    lbl = km.predict(srgb_to_lab(pix))
    counts = np.bincount(lbl, minlength=k)
    keep, remap = [], np.arange(k)
    cc = km.cluster_centers_
    for i in np.argsort(-counts):
        near = [j for j in keep if np.linalg.norm(cc[i] - cc[j]) < 14]
        if near:
            remap[i] = near[0]
        elif keep and counts[i] < 0.05 * len(lbl):  # a sliver (edge blend): fold into the closest colour
            remap[i] = min(keep, key=lambda j: np.linalg.norm(cc[i] - cc[j]))
        else:
            keep.append(i)
    lbl = remap[lbl]
    centres = np.stack([np.median(pix[lbl == i], axis=0) if (lbl == i).any() else pix.mean(0) for i in range(k)])
    return centres[lbl]


ITEM_TILE = 0.35  # an item claims a wall block when it covers at least this much of it


def compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px, detail):
    """Flat paint everywhere (inferred where unseen) with the wall's items, or the photo, on top.

    Returns (image, paint index per pixel - what the paint is underneath the items too -,
    tiles: which wall blocks get a framed map, in image orientation)."""
    h, w = seen.shape
    lab = srgb_to_lab(img)
    dist = np.stack([_de(lab, p) for p in paints_lab], -1)
    label = np.full((h, w), -1)
    label[seen] = dist[seen].argmin(-1)
    item = seen & (dist.min(-1) > ITEM_DE)

    # Paint regions: no small variance - majority over about a block, unseen inferred from nearest seen paint.
    if seen.any():
        lbl = np.where(item | ~seen, -1, label)
        known = lbl >= 0
        if known.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
            lbl = lbl[iy, ix]
        else:
            lbl = np.full((h, w), default_paint)
        if len(paints_rgb) > 1:
            lbl = ndimage.median_filter(lbl, size=max(3, px // 2) | 1, mode="nearest")
    else:
        lbl = np.full((h, w), default_paint)
    out = paints_rgb[lbl].astype(np.float32)

    rows, cols = h // px, w // px
    if detail == "photo":
        out[seen] = img[seen]
        return out, lbl, np.ones((rows, cols), bool)

    # Things on the wall: drop speckle, make them solid, forget anything under 1/16 of a block.
    s = max(1, px // 32)
    item = ndimage.binary_opening(item, structure=np.ones((2 * s + 1, 2 * s + 1)))
    item = ndimage.binary_closing(item, structure=np.ones((4 * s + 1, 4 * s + 1))) & seen
    comp, n = ndimage.label(item)
    item = np.zeros_like(item)
    rects = []
    for i, sl in enumerate(ndimage.find_objects(comp), start=1):
        region = comp[sl] == i
        if region.sum() < (px / 4) ** 2:
            continue
        # Boards, posters, frames and panels are rectangles: snap mostly-rectangular shapes to their box.
        if region.mean() >= 0.6:
            item[sl] = True
            rects.append((sl, region))
        else:
            item[sl] |= region
    if not item.any():
        return out, lbl, np.zeros((rows, cols), bool)
    smooth = np.stack([ndimage.median_filter(img[..., c], size=2 * s + 1) for c in range(3)], -1)
    known = item & seen
    colours = np.zeros_like(out)
    colours[known] = posterise(smooth[known])
    # Items are snapped to whole wall blocks: plain paint stays real blocks (continuous, lit like blocks),
    # and a framed map never shows paint next to a block of paint. A block mostly covered by an item
    # becomes all item - the rest of it takes the nearest item colour.
    tiles = item.reshape(rows, px, cols, px).mean(axis=(1, 3)) >= ITEM_TILE
    area = np.repeat(np.repeat(tiles, px, 0), px, 1)
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    out[area] = colours[iy, ix][area]
    return out, lbl, tiles


def render_walls(views, label_maps, cat_names_of_pixel, ctx, out_dir: Path, px=64, detail="items",
                 fallback_rgb=(225, 222, 215), log=print):
    """Writes tex_<side>.png for every wall and returns the plan's "textures" entry."""
    room, origin, R, y_shift, f = ctx["room"], ctx["origin"], ctx["R"], ctx["y_shift"], ctx["f"]
    to_world = lambda loc: origin + (loc / f + np.array([0, y_shift, 0])) @ R
    tol = 0.12 / f  # 12 cm, in reconstruction units
    sides = ("north", "south", "west", "east")
    proj = {s: project_wall(s, room, to_world, R, views, label_maps, cat_names_of_pixel, px, tol) for s in sides}
    paints_rgb, paints_lab, paint_blocks = find_paints([p for p in proj.values() if p[1].any()], fallback_rgb)
    # The paint unseen walls get: the one covering most of what was seen.
    seen_lab = [_lab_sample(img[seen]) for img, seen in proj.values() if seen.any()]
    if seen_lab:
        allp = np.concatenate(seen_lab)
        default_paint = int(np.bincount(np.stack([_de(allp, p) for p in paints_lab], -1).argmin(-1),
                                        minlength=len(paints_lab)).argmax())
    else:
        default_paint = 0
    unique_blocks = list(dict.fromkeys(paint_blocks))
    block_of_paint = np.array([unique_blocks.index(b) for b in paint_blocks])
    log("[tex] wall paint: " + ", ".join(b.split(":")[1] for b in unique_blocks))

    walls, wall_blocks = {}, {}
    for side in sides:
        img, seen = proj[side]
        out, lbl, tiles = compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px, detail)
        rows, cols = room.H, room.wall_len(side)
        if tiles.any():
            name = f"textures/tex_{side}.png"  # kept out of the photo folder itself
            (out_dir / "textures").mkdir(exist_ok=True)
            Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_dir / name)
            grid = tiles[::-1]  # wall-grid order: [row][u], row 0 = lowest interior row
            if FLIP[side]:
                grid = grid[:, ::-1]
            walls[side] = {"image": name, "flip": FLIP[side], "tiles": grid.astype(int).tolist()}
        # The wall itself is the paint block everywhere (under the items too), so it is continuous.
        blk = block_of_paint[lbl]
        counts = np.stack([(blk == i).reshape(rows, px, cols, px).sum(axis=(1, 3)) for i in range(len(unique_blocks))], -1)
        grid = counts.argmax(-1)[::-1]
        if FLIP[side]:
            grid = grid[:, ::-1]
        wall_blocks[side] = {"palette": unique_blocks, "grid": grid.astype(int).tolist()}
        log(f"[tex] {side} wall: {seen.mean():.0%} seen in photos, {int(tiles.sum())} block(s) with wall items")
    # Flat colours look cleanest without dithering; full photos need it.
    return {"px": px, "dither": detail == "photo", "walls": walls,
            "wall_blocks": wall_blocks, "wall_main": paint_blocks[default_paint]}


def _lab_sample(rgb):
    return srgb_to_lab(rgb[:: max(1, len(rgb) // 20000)])

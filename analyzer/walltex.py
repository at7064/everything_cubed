"""Photographic wall textures: projects each wall plane back into the photos (using the reconstructed
cameras) and resamples it into a rectified image, 1 tile = 1 wall block. The mod turns each tile into a
locked map in an invisible item frame.

detail="items" (default): the paint is real blocks - one of the 33 wall blocks, found jointly over all walls
so the same paint is the same block everywhere (unseen walls inferred). Item frames lit per entity can never
blend with smoothly lit blocks, so frames are only used for the things on the wall (posters, boards,
clocks...): snapped to whole blocks, in a few flat colours, filling their frames completely.
detail="photo": the actual photo on every block (paint where unseen).
detail="pattern": each wall is judged plain or patterned from what the photos show. A plain wall is built as in
"items". A patterned wall (stone, panelling, wallpaper, brick) gets maps on every block: the photo where it was
seen, and everywhere else patches of it grown outward (patch-based hole filling in the spirit of image quilting,
Efros & Freeman, SIGGRAPH 2001). A wall the photos hardly saw borrows patches from the room's other patterned
walls.

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
            "washer", "microwave", "rug", "floor", "ceiling", "ceiling_light", "chandelier", "window", "door"}
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
REL_TOL = 0.08      # photo surface may be this fraction of the camera distance off the fitted wall plane
PATTERN_OFF = 0.25  # a wall is patterned when this share of what was seen is far from its paint ...
PATTERN_GRAIN = 6.0  # ... or its block-scale lightness detail has this standard deviation (CIELAB L)
PATTERN_MIN_SEEN = 0.05  # below this share seen, a wall follows the rest of the room
PATTERN_OWN_POOL = 0.2   # below this share seen, a patterned wall grows from the whole room's patches
PATTERN_HEAD_ON = 0.5    # a patterned wall only keeps and learns from pixels seen within ~60 deg of straight on


def _de(lab, ref):
    d = lab - ref
    d[..., 0] *= 0.5  # shading and soft shadows change lightness, not paint
    return np.sqrt((d ** 2).sum(-1))


def project_wall(side, room, to_world, R, views, label_maps, cat_names_of_pixel, px, tol, rel_tol=REL_TOL, exclude=None):
    """Resample one wall from the photos. `exclude`: optional per-view masks of pixels that are never wall (e.g.
    detected windows). Returns (rgb image, seen mask, cosine between the chosen view ray and the wall normal:
    1 = seen straight on)."""
    loc, n_room, shape = wall_pixels(side, room, px)
    P = to_world(loc)
    n_world = n_room @ R
    best = np.full(len(P), -np.inf)
    rgb = np.zeros((len(P), 3), np.float32)
    head_on = np.zeros(len(P), np.float32)
    for vi_, (v, lm) in enumerate(zip(views, label_maps)):
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
        # The allowed gap grows with distance: reconstruction depth error does, and real walls are not flat
        # (piers, window bays). A fixed 24 cm kept only 2-13% of Suzzallo's walls, seen from 10-20 m.
        visible = v["mask"][vi, ui] & np.isfinite(d_obs) & (np.abs(d_obs - d_wall) < np.maximum(2 * tol, rel_tol * d_wall))
        names = cat_names_of_pixel(lm[vi, ui])
        names = names.astype(str)
        visible &= ~np.isin(names, list(NOT_WALL)) & ~np.char.startswith(names, "generic:") & ~np.char.startswith(names, "vanilla:")
        if exclude is not None and exclude[vi_] is not None:
            visible &= ~exclude[vi_][vi, ui]
        cos = np.abs(((P[idx] - C) / d_wall[:, None]) @ n_world)
        score = np.where(visible & (cos > 0.2), cos / d_wall, -np.inf)
        better = score > best[idx]
        sel = idx[better]
        best[sel] = score[better]
        head_on[sel] = cos[better]
        rgb[sel] = _bilinear(v["rgb"], u[sel], vv[sel])
    return rgb.reshape(shape + (3,)), np.isfinite(best).reshape(shape), head_on.reshape(shape)


FEATURE_FRONT = 1.2   # metres: built-in shelving can stand this far in front of the fitted wall plane


def project_features(side, room, to_world, R, views, label_maps, cat_names_of_pixel, f, px=4,
                     kinds=("window", "bookshelf")):
    """What each wall cell is, from the most head-on photo that sees it: shares (rows x cols, wall-grid order:
    row 0 = lowest interior row) of `kinds`, plus the share of the cell seen at all. Unlike the textures,
    surfaces up to FEATURE_FRONT in front of the wall count (shelves), and anything behind it too (a view
    through a window)."""
    loc, n_room, shape = wall_pixels(side, room, px)
    P = to_world(loc)
    n_world = n_room @ R
    best = np.full(len(P), -np.inf)
    label = np.full(len(P), "", dtype=object)
    front = FEATURE_FRONT / f
    for v, lm in zip(views, label_maps):
        if "K" not in v:
            continue
        c2w = v["cam2world"]
        w2c = np.linalg.inv(c2w)
        Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = Pc[:, 2]
        h, w = v["rgb"].shape[:2]
        K = v["K"]
        ok = z > 1e-3
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
        names = cat_names_of_pixel(lm[vi, ui]).astype(str)
        tol = np.maximum(0.24 / f, REL_TOL * d_wall)
        near = (d_obs > d_wall - np.maximum(front, tol)) & (d_obs < d_wall + tol)
        # Only glass may be seen through: past an inferred wall the photos show the rest of the hall.
        through = (names == "window") & (d_obs >= d_wall + tol)
        visible = v["mask"][vi, ui] & np.isfinite(d_obs) & (near | through)
        cos = np.abs(((P[idx] - C) / d_wall[:, None]) @ n_world)
        score = np.where(visible & (cos > 0.3), cos / d_wall, -np.inf)
        better = score > best[idx]
        sel = idx[better]
        best[sel] = score[better]
        label[sel] = names[better]
    rows, cols = room.H, room.wall_len(side)
    pool = lambda m: m.reshape(shape).reshape(rows, px, cols, px).mean(axis=(1, 3))
    out = {"seen": pool(np.isfinite(best))}
    lab = label.astype(str)
    for k in kinds:  # the template's name, or the vanilla block of the same name ("vanilla:bookshelf:cube:...")
        out[k] = pool((lab == k) | np.char.startswith(lab, f"vanilla:{k}:"))
    for k in out:
        g = out[k][::-1]                       # image row 0 is the top; the wall grid's row 0 is the bottom
        out[k] = g[:, ::-1] if FLIP[side] else g
    return out


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
                 fallback_rgb=(225, 222, 215), log=print, exclude=None):
    """Writes tex_<side>.png for every wall and returns the plan's "textures" entry. `exclude`: optional per-view
    masks of pixels that are never wall surface (detected windows and doors)."""
    room, origin, R, y_shift, f = ctx["room"], ctx["origin"], ctx["R"], ctx["y_shift"], ctx["f"]
    to_world = lambda loc: origin + (loc / f + np.array([0, y_shift, 0])) @ R
    tol = 0.12 / f  # 12 cm, in reconstruction units
    sides = ("north", "south", "west", "east")
    proj3 = {s: project_wall(s, room, to_world, R, views, label_maps, cat_names_of_pixel, px, tol, exclude=exclude)
             for s in sides}
    proj = {s: p[:2] for s, p in proj3.items()}
    if detail == "pattern":
        # Pixels seen at a grazing angle are smeared along the wall: a patterned wall neither keeps them nor
        # learns its pattern from them; the pattern grown from the clean part replaces them.
        proj = {s: (img, seen & (proj3[s][2] >= PATTERN_HEAD_ON)) for s, (img, seen) in proj.items()}
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

    patterned = {}
    if detail == "pattern":
        rng = np.random.default_rng(0)
        stats = {s: wall_pattern_stats(*proj[s], paints_lab, px) for s in sides}
        own = {s: proj[s][1].mean() >= PATTERN_MIN_SEEN for s in sides}
        for s in sides:
            if own[s]:
                patterned[s] = stats[s][0] >= PATTERN_OFF or stats[s][1] >= PATTERN_GRAIN
        # Walls the photos hardly saw follow the room: patterned if most of the wall area seen was.
        seen_area = {s: proj[s][1].sum() for s in patterned}
        room_patterned = sum(seen_area[s] for s in patterned if patterned[s]) > 0.5 * max(1, sum(seen_area.values()))
        for s in sides:
            if not own[s]:
                patterned[s] = room_patterned
            log(f"[tex] {s} wall: {'patterned' if patterned[s] else 'plain'} (off-paint {stats[s][0]:.0%}, "
                f"grain {stats[s][1]:.1f}{'' if own[s] else ', too little seen: follows the room'})")
        B, ds = px, max(1, px // 16)
        pools = {s: _exemplars(*proj[s], B, ds, max(2, px // 4), 1500, rng) for s in sides if patterned.get(s) and own[s]}
        room_pool = (np.concatenate([p[0] for p in pools.values()]) if pools else np.zeros((0, B, B, 3), np.uint8),
                     np.concatenate([p[1] for p in pools.values()]) if pools else np.zeros((0, B // ds, B // ds, 3), np.float32))

    walls, wall_blocks = {}, {}
    for side in sides:
        img, seen = proj[side]
        if patterned.get(side):
            use_own = side in pools and len(pools[side][0]) >= 30 and seen.mean() >= PATTERN_OWN_POOL
            pool = pools[side] if use_own else room_pool
            if len(pool[0]) >= 8:
                out, lbl, tiles = compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px, "photo")
                out = synthesize_wall(np.where(seen[..., None], img, out), seen, pool, px, rng)
            else:
                log(f"[tex] {side} wall: too few clean patches to grow a pattern from; plain")
                patterned[side] = False
        if not patterned.get(side):
            out, lbl, tiles = compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px,
                                           "items" if detail == "pattern" else detail)
        rows, cols = room.H, room.wall_len(side)
        if tiles.any():
            name = f"textures/tex_{side}.png"  # kept out of the photo folder itself
            (out_dir / "textures").mkdir(exist_ok=True)
            Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_dir / name)
            grid = tiles[::-1]  # wall-grid order: [row][u], row 0 = lowest interior row
            if FLIP[side]:
                grid = grid[:, ::-1]
            walls[side] = {"image": name, "flip": FLIP[side], "tiles": grid.astype(int).tolist()}
            if detail == "pattern":
                walls[side]["dither"] = bool(patterned.get(side))  # photo-like walls dither, flat items do not
        # The wall itself is the paint block everywhere (under the items too), so it is continuous.
        blk = block_of_paint[lbl]
        counts = np.stack([(blk == i).reshape(rows, px, cols, px).sum(axis=(1, 3)) for i in range(len(unique_blocks))], -1)
        grid = counts.argmax(-1)[::-1]
        if FLIP[side]:
            grid = grid[:, ::-1]
        wall_blocks[side] = {"palette": unique_blocks, "grid": grid.astype(int).tolist()}
        log(f"[tex] {side} wall: {seen.mean():.0%} seen in photos, {int(tiles.sum())} block(s) with wall items")
    features = {s: project_features(s, room, to_world, R, views, label_maps, cat_names_of_pixel, f) for s in sides}
    # Flat colours look cleanest without dithering; full photos need it.
    return {"px": px, "dither": detail == "photo", "walls": walls, "features": features,
            "wall_blocks": wall_blocks, "wall_main": paint_blocks[default_paint]}


def wall_pattern_stats(img, seen, paints_lab, px):
    """(share of seen pixels far from every paint, std of fine lightness detail) for one wall."""
    if seen.sum() < 4 * px * px:
        return 0.0, 0.0
    lab = srgb_to_lab(img)
    off = float((np.stack([_de(lab[seen], p) for p in paints_lab], -1).min(-1) > ITEM_DE).mean())
    L = np.where(seen, lab[..., 0], np.nan)
    # Detail at about a quarter block and finer; shading and sun bands are slower and do not count.
    fill = np.where(seen, lab[..., 0], np.nanmedian(L))
    fine = fill - ndimage.gaussian_filter(fill, px / 4)
    core = ndimage.binary_erosion(seen, iterations=max(1, px // 8))
    grain = float(np.std(fine[core])) if core.any() else 0.0
    return off, grain


def _exemplars(img, seen, B, ds, stride, cap, rng):
    """Fully seen B x B patches of a wall (uint8), with their ds-pooled versions for matching."""
    h, w = seen.shape
    if h < B or w < B:
        return np.zeros((0, B, B, 3), np.uint8), np.zeros((0, B // ds, B // ds, 3), np.float32)
    frac = ndimage.uniform_filter(seen.astype(np.float32), B, mode="constant")
    ys, xs = np.mgrid[0:h - B + 1:stride, 0:w - B + 1:stride]
    ok = frac[ys + B // 2, xs + B // 2] >= 0.97
    ys, xs = ys[ok], xs[ok]
    if len(ys) > cap:
        pick = rng.choice(len(ys), cap, replace=False)
        ys, xs = ys[pick], xs[pick]
    full = np.stack([img[y:y + B, x:x + B] for y, x in zip(ys, xs)]).clip(0, 255).astype(np.uint8) if len(ys) else \
        np.zeros((0, B, B, 3), np.uint8)
    # Not window glare or sky that leaked in, nor deep shadow: they would spread as white or black patches.
    lum = full.reshape(len(full), -1, 3).mean(1) @ np.array([0.299, 0.587, 0.114]) if len(full) else np.zeros(0)
    full = full[(lum > 25) & (lum < 225)]
    low = full.reshape(len(full), B // ds, ds, B // ds, ds, 3).mean(axis=(2, 4)).astype(np.float32)
    return full, low


def synthesize_wall(img, seen, pool, px, rng, top_k=5, reuse=0.0):
    """Fills the unseen part of a wall image with patches from `pool` (full, low) exemplars, keeping every seen
    pixel. Patches are one block wide; each is chosen among the `top_k` best matches to what is already known
    in its window (photo or earlier patches), so it continues the pattern without copying one patch over and
    over. With `reuse` > 0 every use of a patch makes it that share of the typical match cost more expensive, so
    one striking patch repeats less; off by default, because with few clean patches it turns walls into a busy
    collage (tried on Suzzallo). Returns the filled image."""
    full, low = pool
    B = px
    ds = max(1, B // 16)
    ov = max(2, B // 4)
    step = B - ov
    h, w = seen.shape
    out = img.astype(np.float32).copy()
    known = seen.copy()
    uses = np.zeros(len(full))
    ys = sorted(set(list(range(0, max(1, h - B + 1), step)) + [max(0, h - B)]))
    xs = sorted(set(list(range(0, max(1, w - B + 1), step)) + [max(0, w - B)]))
    for y in ys:
        for x in xs:
            wk = known[y:y + B, x:x + B]
            if wk.all():
                continue
            m = wk.reshape(B // ds, ds, B // ds, ds).mean(axis=(1, 3))
            if m.sum() < 1e-6:
                i = int(rng.integers(len(full)))
            else:
                tgt = (out[y:y + B, x:x + B] * wk[..., None]).reshape(B // ds, ds, B // ds, ds, 3).sum(axis=(1, 3))
                tgt = tgt / np.maximum(m * ds * ds, 1e-6)[..., None]
                d = (((low - tgt) ** 2).sum(-1) * m).sum(axis=(1, 2)) / m.sum()
                d = d + reuse * np.median(d) * uses
                best = np.argpartition(d, min(top_k, len(d) - 1))[:top_k]
                i = int(rng.choice(best))
            uses[i] += 1
            patch = full[i].astype(np.float32)
            win = out[y:y + B, x:x + B]
            if wk.any():
                # Same brightness as what it continues (sun bands and shade would otherwise make a checkerboard).
                gain = np.clip(win[wk].mean() / max(patch[wk].mean(), 1.0), 0.6, 1.6)
                patch = (patch * gain).clip(0, 255)
                # Feather across the overlap with earlier patches; photo pixels are never changed.
                wt = np.clip(ndimage.distance_transform_edt(wk) / ov, 0, 1)
                blend = wk & ~seen[y:y + B, x:x + B]
                win[blend] = win[blend] * wt[blend, None] + patch[blend] * (1 - wt[blend, None])
            win[~wk] = patch[~wk]
            known[y:y + B, x:x + B] = True
    return out


def _lab_sample(rgb):
    return srgb_to_lab(rgb[:: max(1, len(rgb) // 20000)])

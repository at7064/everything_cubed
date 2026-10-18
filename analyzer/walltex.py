"""Photographic wall textures: projects each wall plane back into the photos (using the reconstructed
cameras) and resamples it into a rectified image, 1 tile = 1 wall block. The mod turns each tile into a
locked map in an invisible item frame.

detail="items": the paint is real blocks - one of the 33 wall blocks, found jointly over all walls
so the same paint is the same block everywhere (unseen walls inferred). Item frames lit per entity can never
blend with smoothly lit blocks, so frames are only used for the things on the wall (posters, boards,
clocks...): snapped to whole blocks, in a few flat colours, filling their frames completely.
detail="photo": the actual photo on every block (paint where unseen).
detail="uniform" (the analyzers' default): every wall block is a map in the room's one measured paint, with the
things detected on the wall drawn on top.
detail="style": one wall style for the whole room, so every wall matches. Its vertical profile is the
median colour at each height over all clean wall pixels of all walls (bands such as a wainscot and its trim
line stay at their height; photo lighting averages out), and its fine texture is one typical block-sized
patch made seamless and tiled. Every wall gets that same picture; a room with neither bands nor texture is
built as plain blocks.
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
            "washer", "microwave", "rug", "floor", "ceiling", "ceiling_light", "chandelier", "window", "door",
            "whiteboard", "exit_sign", "fire_extinguisher", "projector", "fan",
            "backpack", "globe", "locker", "drinking_fountain", "umbrella_stand", "shelf"}
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
PATTERN_OFF = 0.4   # a wall is patterned when this share of what was seen is far from its paint ...
PATTERN_GRAIN = 9.0  # ... or its block-scale lightness detail has this standard deviation (CIELAB L)
# (Both measured after even_lighting. Windows, blinds and a few posters are wall items, not a pattern.)
PATTERN_COLOURS = 8      # a patterned wall is drawn in at most this many flat colours
LIGHT_GAIN = (0.6, 1.7)  # how far even_lighting may brighten or darken a pixel
PATTERN_MIN_SEEN = 0.05  # below this share seen, a wall follows the rest of the room
PATTERN_OWN_POOL = 0.2   # below this share seen, a patterned wall grows from the whole room's patches
PATTERN_HEAD_ON = 0.5    # a patterned wall only keeps and learns from pixels seen within ~60 deg of straight on
LIGHT_SCALE = 2.0        # blocks: brightness changes slower than this are the photo's lighting, not the wall
LIGHT_STRENGTH = 0.85    # how much of that lighting is removed (1 = all)
LUMA_GLARE, LUMA_SHADOW = 235, 22  # luminance (0-255) beyond which a photo pixel is glare or deep shadow, not wall


def even_wall_lighting(img, seen, px):
    """Removes the photos' own lighting from a wall image: sun bands, cast shadows and fall-off become an even
    brightness, while colour and detail finer than LIGHT_SCALE blocks stay. Blown-out glare and near-black
    shadow carry no wall at all and become unseen (the pattern fill replaces them). Returns (image, seen)."""
    Y = _luma(img)
    seen = seen & (Y < LUMA_GLARE) & (Y > LUMA_SHADOW)
    if seen.sum() < px * px:
        return img, seen
    sigma = LIGHT_SCALE * px
    m = seen.astype(np.float32)
    low = ndimage.gaussian_filter(Y * m, sigma) / np.maximum(ndimage.gaussian_filter(m, sigma), 1e-3)
    gain = np.clip((np.median(Y[seen]) / np.maximum(low, 1.0)) ** LIGHT_STRENGTH, 0.5, 2.5)
    return np.where(seen[..., None], (img * gain[..., None]).clip(0, 255), img).astype(np.float32), seen


def _de(lab, ref):
    d = lab - ref
    d[..., 0] *= 0.5  # shading and soft shadows change lightness, not paint
    return np.sqrt((d ** 2).sum(-1))


def project_wall(side, room, to_world, R, views, label_maps, cat_names_of_pixel, px, tol, rel_tol=REL_TOL, exclude=None,
                 return_source=False, items=None):
    """Resample one wall from the photos. `exclude`: optional per-view masks of pixels that are never wall (e.g.
    detected windows). Returns (rgb image, seen mask, cosine between the chosen view ray and the wall normal:
    1 = seen straight on), plus the index of the view each pixel came from (-1 unseen) with return_source, plus
    with `items` (per-view masks of detected wall items) the share of the photos seeing each pixel that put it
    inside an item."""
    loc, n_room, shape = wall_pixels(side, room, px)
    P = to_world(loc)
    n_world = n_room @ R
    best = np.full(len(P), -np.inf)
    rgb = np.zeros((len(P), 3), np.float32)
    head_on = np.zeros(len(P), np.float32)
    source = np.full(len(P), -1, np.int32)
    looks = np.zeros(len(P), np.float32)
    hits = np.zeros(len(P), np.float32)
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
        if items is not None and items[vi_] is not None:
            # Detectors miss small things in far, steep shots: those count for less.
            wgt = np.where(np.isfinite(score), cos, 0.0)
            looks[idx] += wgt
            hits[idx] += wgt * items[vi_][vi, ui]
        better = score > best[idx]
        sel = idx[better]
        best[sel] = score[better]
        head_on[sel] = cos[better]
        source[sel] = vi_
        rgb[sel] = _bilinear(v["rgb"], u[sel], vv[sel])
    out = rgb.reshape(shape + (3,)), np.isfinite(best).reshape(shape), head_on.reshape(shape)
    if return_source:
        out += (source.reshape(shape),)
    if items is not None:
        out += ((hits / np.maximum(looks, 1e-6)).reshape(shape),)
    return out


FEATURE_FRONT = 1.2   # metres: built-in shelving can stand this far in front of the fitted wall plane


def project_features(side, room, to_world, R, views, label_maps, cat_names_of_pixel, f, px=4,
                     kinds=("window", "bookshelf")):
    """What each wall cell is, combining visible photos weighted by view angle and distance: shares (rows x cols, wall-grid order:
    row 0 = lowest interior row) of `kinds`, plus the share of the cell seen at all. Unlike the textures,
    surfaces up to FEATURE_FRONT in front of the wall count (shelves), and anything behind it too (a view
    through a window)."""
    loc, n_room, shape = wall_pixels(side, room, px)
    P = to_world(loc)
    n_world = n_room @ R
    best = np.full(len(P), -np.inf)
    votes = {k: np.zeros(len(P), np.float32) for k in kinds}
    total = np.zeros(len(P), np.float32)
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
        weight = np.where(np.isfinite(score), np.maximum(score, 0), 0)
        total[idx] += weight
        for kind in kinds:
            hit = (names == kind) | np.char.startswith(names, f"vanilla:{kind}:")
            votes[kind][idx] += weight * hit
        better = score > best[idx]
        sel = idx[better]
        best[sel] = score[better]
    rows, cols = room.H, room.wall_len(side)
    pool = lambda m: m.reshape(shape).reshape(rows, px, cols, px).mean(axis=(1, 3))
    out = {"seen": pool(np.isfinite(best))}
    for k in kinds:
        # A single noisy view must not override other clear views of the same wall point.
        share = np.divide(votes[k], total, out=np.zeros_like(total), where=total > 0)
        out[k] = pool(share)
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
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(_lab_sample(pix))
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
    comp, _ = ndimage.label(item)
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


def _luma(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def even_lighting(proj, px):
    """Takes the lighting out of the projected walls, so one paint is one colour and only real things on the
    wall stand out. proj: {side: (img, seen, head_on, source view)}; returns the same with corrected images.

    1. Exposure: every photo's paint-coloured pixels are brought to the same median brightness (the phone
       exposes each shot differently, which made hard seams where the chosen photo changes).
    2. Shading: what is left of slow brightness changes across each wall (lamps, sun, glare, shadow) is divided
       out, estimated from the paint-coloured pixels only, blurred over about 1.5 blocks."""
    seen_all = [(img[seen], src[seen]) for img, seen, _, src in proj.values() if seen.any()]
    if not seen_all:
        return proj
    pix = np.concatenate([p for p, _ in seen_all]).astype(np.float32)
    srcs = np.concatenate([s for _, s in seen_all])
    lab = srgb_to_lab(pix[:: max(1, len(pix) // 40000)])
    ref_ab = np.median(lab[:, 1:], axis=0)

    def paintlike(rgb):  # near the walls' main colour in hue; any brightness (that is what we correct)
        return np.linalg.norm(srgb_to_lab(rgb)[..., 1:] - ref_ab, axis=-1) < 10

    pl = paintlike(pix)
    if pl.sum() < 100:
        return proj
    target = float(np.median(_luma(pix[pl])))
    gain_of_view = {}
    for k in np.unique(srcs):
        m = pl & (srcs == k)
        if m.sum() >= 200:
            gain_of_view[int(k)] = float(np.clip(target / max(np.median(_luma(pix[m])), 1.0), *LIGHT_GAIN))

    out = {}
    for side, (img, seen, head_on, src) in proj.items():
        img = img.astype(np.float32).copy()
        g = np.ones(seen.shape, np.float32)
        for k, gk in gain_of_view.items():
            g[seen & (src == k)] = gk
        img *= g[..., None]
        w = (seen & paintlike(img)).astype(np.float32)
        if w.sum() >= px * px:
            Y = ndimage.median_filter(_luma(img), size=max(3, px // 4) | 1)  # small items do not count
            sig = 1.5 * px
            wb = ndimage.gaussian_filter(w, sig, mode="nearest")
            illum = ndimage.gaussian_filter(Y * w, sig, mode="nearest") / np.maximum(wb, 1e-6)
            # Where no paint was seen nearby, keep the exposure-corrected photo as it is.
            k = np.clip(target / np.maximum(illum, 1.0), *LIGHT_GAIN)
            k = np.where(wb > 0.05, k, 1.0)
            img *= k[..., None]
        out[side] = (img.clip(0, 255), seen, head_on, src)
    return out


UNIFORM_MIN_ITEM = 16  # in "uniform" mode, detected things at least (px / this) pixels square are kept
UNIFORM_ITEM_DE = 12.0  # ... and pixels this far from the paint around them (after balancing) are part of one
UNIFORM_BG_DE = 25.0    # pixels this close to the wall's paint tell what the paint looks like locally
UNIFORM_DETECT_SHARE = 0.3  # a pixel is a detected wall item when this share of the photos seeing it say so
UNIFORM_DETECT_MARKS = 0.03  # ... and a detected region is kept when this share of it differs from the paint
UNIFORM_EDGE = 3.0      # CIELAB L per pixel: an edge (a frame, a switch plate's outline)
UNIFORM_BLEND_DE = (4.0, 12.0)  # an item pixel this close to the local paint is paint ... this far, all item
UNIFORM_FEATHER = 2.0   # pixels: an item's outer edge fades into the paint over this width
UNIFORM_HALO = 0.15     # blocks: at most this far in from an item's outline, paint-like pixels fade into the paint
GLARE = 245             # pixels with every channel above this are blown out: never a thing on the wall


def room_paint(proj, paints_lab, default_paint, fallback_rgb):
    """The one colour the room's walls are painted, as measured: the median of every seen wall pixel close to the
    main paint (after even_lighting), so posters, switches and shadows do not shift it."""
    pix = np.concatenate([img[seen] for img, seen, *_ in proj.values() if seen.any()]) if proj else np.zeros((0, 3))
    if len(pix) < 50:
        return np.array(fallback_rgb, float)
    pix = pix[:: max(1, len(pix) // 200000)]
    near = _de(srgb_to_lab(pix), paints_lab[default_paint]) < ITEM_DE
    return np.median(pix[near] if near.sum() >= 50 else pix, axis=0)


def balance_photos(proj, paint_rgb):
    """Every photo's own white balance and exposure: per source photo and colour channel, the gain that turns the
    paint as that photo saw it (over all walls) into the room's paint. Returns {side: balanced image}."""
    gains = {}
    srcs = np.unique(np.concatenate([src[seen] for img, seen, _, src in proj.values()]))
    for k in srcs:
        pix = np.concatenate([img[seen & (src == k)] for img, seen, _, src in proj.values()])
        pix = pix[pix.min(-1) < GLARE]
        if len(pix) < 50:
            continue
        med = np.median(pix, axis=0)
        near = _de(srgb_to_lab(pix), srgb_to_lab(med)) < 25   # the paint in this photo, not what hangs on it
        if near.sum() >= 50:
            med = np.median(pix[near], axis=0)
        gains[int(k)] = np.clip(np.asarray(paint_rgb) / np.maximum(med, 1.0), *LIGHT_GAIN)
    out = {}
    for side, (img, seen, _, src) in proj.items():
        b = img.astype(np.float32).copy()
        for k, g in gains.items():
            m = seen & (src == k)
            b[m] = img[m] * g
        out[side] = b.clip(0, 255)
    return out


def _edges(lab_l, looked, src):
    """Lightness edges on the wall, without the seams where the picture switches from one photo to another."""
    L = ndimage.median_filter(lab_l, 3)
    edge = (np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1)) / 8 > UNIFORM_EDGE) & looked
    if src is not None:
        seam = ndimage.maximum_filter(src, 3) != ndimage.minimum_filter(src, 3)
        edge &= ~ndimage.binary_dilation(seam, iterations=2)
    return edge


def _local_paint(img, looked, paint_rgb, px):
    """What the paint looks like around each pixel in the photo (seams between photos and leftover shading move it
    a little): a blur over the pixels close to the wall's paint. Returns (the lightly smoothed photo's CIELAB -
    sensor grain does not count -, local paint CIELAB)."""
    smooth = np.stack([ndimage.median_filter(img[..., c], size=3) for c in range(3)], -1)
    lab = srgb_to_lab(smooth)
    paint_lab = srgb_to_lab(np.asarray(paint_rgb, float))
    w = (looked & (_de(lab, paint_lab) < UNIFORM_BG_DE)).astype(np.float32)
    sig = px / 2
    wb = ndimage.gaussian_filter(w, sig, mode="nearest")
    blur = lambda ch: ndimage.gaussian_filter(ch * w, sig, mode="nearest") / np.maximum(wb, 1e-6)
    bg_lab = np.where(wb[..., None] > 0.05, np.stack([blur(lab[..., c]) for c in range(3)], -1), paint_lab)
    return lab, bg_lab


def _blend_items(img, keep, paint_rgb, lab, looked, edge, px):
    """The things on the wall, blended into the flat paint so no photo-coloured halo shows around them:
    1. each thing is recoloured by one gain, so the photo's wall in a thin ring just around it becomes exactly the
       flat colour and the thing keeps its look relative to the wall (one gain per thing: no blotches inside);
    2. near its outline (a band a fifth of its size, at most UNIFORM_HALO of a block), what still looks like that
       ring fades into the flat paint - the loose edges of the detector's box -, while the rest of the thing, and
       the inside of a closed shape (a painting's frame, a clock's rim), stays solid: a whiteboard's own shading
       never punches holes into it;
    3. the outer edge is feathered over UNIFORM_FEATHER pixels.
    Returns the blended pixels (h, w, 3) and their opacity."""
    paint = np.asarray(paint_rgb, np.float32)
    paint_lab = srgb_to_lab(paint.astype(float))
    wall = looked & ~keep & (_de(lab, paint_lab) < UNIFORM_BG_DE)
    corrected = img.astype(np.float32).copy()
    alpha = np.zeros(keep.shape, np.float32)
    dist = ndimage.distance_transform_edt(keep)
    comp, _ = ndimage.label(keep)
    r = max(3, px // 16)
    h, w = keep.shape
    for i, sl in enumerate(ndimage.find_objects(comp), start=1):
        win = slice(max(sl[0].start - r, 0), min(sl[0].stop + r, h)), slice(max(sl[1].start - r, 0), min(sl[1].stop + r, w))
        region = comp[win] == i
        ring = ndimage.binary_dilation(region, iterations=r) & wall[win]
        ring_rgb, ring_lab = paint, paint_lab
        if ring.sum() >= 10:
            ring_rgb = np.median(img[win][ring], axis=0)
            ring_lab = srgb_to_lab(ring_rgb.astype(float))
        gain = np.clip(paint / np.maximum(ring_rgb, 1.0), *LIGHT_GAIN)
        corrected[win][region] = (img[win][region] * gain).clip(0, 255)
        a = np.clip((_de(lab[win], ring_lab) - UNIFORM_BLEND_DE[0]) / (UNIFORM_BLEND_DE[1] - UNIFORM_BLEND_DE[0]), 0, 1)
        a = np.maximum(a, edge[win])                          # plate outlines stay
        solid = ndimage.binary_fill_holes(ndimage.binary_closing(a > 0.5, np.ones((3, 3)))) & region
        band = np.clip(0.2 * min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), 1, UNIFORM_HALO * px)
        a = np.where(solid | (dist[win] > band), 1.0, a)
        alpha[win][region] = a[region]
    alpha = ndimage.gaussian_filter(alpha, 0.7)               # no hard single-pixel flecks
    alpha *= np.clip(dist / UNIFORM_FEATHER, 0, 1)
    alpha = np.where(keep, alpha, 0.0)[..., None]
    return paint * (1 - alpha) + corrected * alpha, alpha[..., 0]


def _fit_detections(detected, off, edge, px):
    """Detector boxes, projected from several photos, are loose and sometimes wrong. A region with (almost)
    nothing in it that differs from the paint (colour, or an outline) - a blank wall panel taken for a whiteboard -
    is dropped; a big one is cut down to the box around what does differ."""
    marks = ndimage.binary_opening(detected & (off | edge), np.ones((2, 2)))
    comp, _ = ndimage.label(detected)
    keep = np.zeros_like(detected)
    for i, sl in enumerate(ndimage.find_objects(comp), start=1):
        region = comp[sl] == i
        m = marks[sl] & region
        if m.sum() < max(4, UNIFORM_DETECT_MARKS * region.sum()):
            continue
        if region.sum() < (px / 2) ** 2:   # switches, sockets, signs: their plate is the paint's colour; keep it all
            keep[sl] |= region
            continue
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - 2, 0), ys.max() + 3
        x0, x1 = max(xs.min() - 2, 0), xs.max() + 3
        fit = np.zeros_like(region)
        fit[y0:y1, x0:x1] = True
        keep[sl] |= region & fit
    comp, n = ndimage.label(keep)
    sizes = ndimage.sum(keep, comp, np.arange(1, n + 1))
    return keep & np.isin(comp, 1 + np.flatnonzero(sizes >= (px / UNIFORM_MIN_ITEM) ** 2))


def compose_uniform(img, seen, paint_rgb, px, src=None, detected=None):
    """One flat paint colour over the whole wall, with the things on it (posters, clocks, switches, boards) copied
    straight from the (balanced) photo on top. Nothing is snapped to blocks (every block is a map). Returns
    (image, mask of the things on the wall).

    `detected`: the share of photos whose detector (wallitems) put each pixel inside a wall item; the detector
    alone decides what is an item (fitted to what differs from the paint), so sun patches, glare and chair backs
    leaning on the wall stay paint. Items are blended into the paint (_blend_items). None (no detector): the plain paint only. `src`: which photo each pixel came
    from, so the seams between photos are not taken for outlines."""
    out = np.empty(img.shape, np.float32)
    out[:] = paint_rgb
    looked = seen & (img.min(-1) < GLARE)
    if detected is None or not looked.any():
        return out, np.zeros(seen.shape, bool)
    lab, bg_lab = _local_paint(img, looked, paint_rgb, px)
    off = looked & (_de(lab, bg_lab) > UNIFORM_ITEM_DE)
    edge = _edges(lab[..., 0], looked, src)
    keep = _fit_detections(seen & (detected >= UNIFORM_DETECT_SHARE), off, edge, px)
    out, alpha = _blend_items(img, keep, paint_rgb, lab, looked, edge, px)
    return out, alpha > 0.05


def render_walls(views, label_maps, cat_names_of_pixel, ctx, out_dir: Path, px=64, detail="items",
                 fallback_rgb=(225, 222, 215), log=print, exclude=None, surface_tol=None, item_masks=None):
    """Writes tex_<side>.png for every wall and returns the plan's "textures" entry. `exclude`: optional per-view
    masks of pixels that are never wall surface (detected windows and doors). `surface_tol`: (metres, share of the
    camera distance) a photo's surface may be off the wall plane; measured depth (LiDAR) can be much tighter than
    the default, so furniture and people standing in front of a wall are not painted onto it. `item_masks`:
    optional per-view masks of detected wall items (wallitems.detect_wall_items), used by detail="uniform"
    (without them, uniform walls are plain paint)."""
    room, origin, R, y_shift, f = ctx["room"], ctx["origin"], ctx["R"], ctx["y_shift"], ctx["f"]
    to_world = lambda loc: origin + (loc / f + np.array([0, y_shift, 0])) @ R
    tol = 0.12 / f  # 12 cm, in reconstruction units
    rel_tol = REL_TOL
    if surface_tol:
        tol, rel_tol = surface_tol[0] / (2 * f), surface_tol[1]   # project_wall allows 2 * tol
    sides = ("north", "south", "west", "east")
    items = item_masks if detail == "uniform" else None
    raw = {s: project_wall(s, room, to_world, R, views, label_maps, cat_names_of_pixel, px, tol, rel_tol=rel_tol,
                           exclude=exclude, return_source=True, items=items) for s in sides}
    detected = {s: p[4] for s, p in raw.items()} if items is not None else {s: None for s in sides}
    proj3 = {s: p[:4] for s, p in raw.items()}
    if detail == "uniform":  # the others keep the photo's colours ("pattern" evens each wall on its own below)
        proj3 = even_lighting(proj3, px)
    proj = {s: p[:2] for s, p in proj3.items()}
    if detail in ("pattern", "style"):
        # Pixels seen at a grazing angle are smeared along the wall: a patterned wall neither keeps them nor
        # learns its pattern from them; the pattern grown from the clean part replaces them.
        proj = {s: (img, seen & (proj3[s][2] >= PATTERN_HEAD_ON)) for s, (img, seen) in proj.items()}
    if detail == "pattern":  # "photo" stays the raw photo: evening out cannot tell slow paint changes from light
        proj = {s: even_wall_lighting(img, seen, px) for s, (img, seen) in proj.items()}
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

    if detail == "uniform":
        # Every wall block is a map, things on the wall drawn on top. The paint is exactly the colour of the wall
        # block closest to the room's measured colour (as _snap does for the other modes): where no frame fits,
        # behind a bed or a sofa against the wall, the bare block shows, and it must look the same as the maps.
        paint = room_paint(proj3, paints_lab, default_paint, fallback_rgb)
        balanced = balance_photos(proj3, paint)
        block = WALL_PALETTE.nearest(paint)
        block_rgb = np.array(WALL_COLOURS[block], float)
        log(f"[tex] uniform wall colour rgb{tuple(int(round(c)) for c in paint)} -> {block.split(':')[1]} "
            f"rgb{tuple(int(c) for c in block_rgb)}")
        walls, wall_blocks = {}, {}
        (out_dir / "textures").mkdir(exist_ok=True)
        for side in sides:
            seen = proj[side][1]
            own = block_rgb   # every wall the block's colour, so frames and bare blocks match
            out, items = compose_uniform(balanced[side], seen, own, px, src=proj3[side][3], detected=detected[side])
            rows, cols = room.H, room.wall_len(side)
            name = f"textures/tex_{side}.png"
            Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_dir / name)
            walls[side] = {"image": name, "flip": FLIP[side], "tiles": np.ones((rows, cols), int).tolist(),
                           "dither": False}
            wall_blocks[side] = {"palette": [block], "grid": np.zeros((rows, cols), int).tolist()}
            n_items = int(items.reshape(rows, px, cols, px).any(axis=(1, 3)).sum())
            log(f"[tex] {side} wall: {seen.mean():.0%} seen in photos, colour rgb{tuple(int(round(c)) for c in own)}, "
                f"things on the wall in {n_items} block(s)")
        features = {s: project_features(s, room, to_world, R, views, label_maps, cat_names_of_pixel, f) for s in sides}
        # "lit": the mod puts a full light in front of every frame, so all maps show at the same brightness.
        return {"px": px, "dither": False, "lit": True, "walls": walls, "features": features,
                "wall_blocks": wall_blocks, "wall_main": block}

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

    style = None
    if detail == "style":
        # Glare and deep shadow carry no wall; everything else is averaged over the whole room.
        clean = {s: (img, seen & (_luma(img) < LUMA_GLARE) & (_luma(img) > LUMA_SHADOW))
                 for s, (img, seen) in proj.items()}
        style = room_style([clean[s] for s in sides if clean[s][1].any()], px) if any(c[1].any() for c in clean.values()) else None
        if style is None:
            log("[tex] too little clean wall seen for a room style; plain walls")
        else:
            st = style[2]
            plain = st["height_spread"] < STYLE_PLAIN_DE and st["grain"] < STYLE_PLAIN_GRAIN
            log(f"[tex] room style: {'plain' if plain else 'textured'} (heights differ {st['height_spread']:.1f} dE, "
                f"grain {st['grain']:.1f}, from {st['patches']} clean patches)")
            if plain:
                style = None

    walls, wall_blocks = {}, {}
    for side in sides:
        img, seen = proj[side]
        if style is not None:
            profile, tile, _ = style
            rows_px, cols_px = img.shape[:2]
            reps = (rows_px // px + 1, cols_px // px + 1, 1)
            out = profile[:, None, :] + np.tile(tile, reps)[:rows_px, :cols_px]
            _, lbl, _ = compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px, "photo")  # labels only
            tiles = np.ones((rows_px // px, cols_px // px), bool)
        elif patterned.get(side):
            use_own = side in pools and len(pools[side][0]) >= 30 and seen.mean() >= PATTERN_OWN_POOL
            pool = pools[side] if use_own else room_pool
            if len(pool[0]) >= 8:
                out, lbl, tiles = compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px, "photo")
                out = synthesize_wall(np.where(seen[..., None], img, out), seen, pool, px, rng)
                # A few flat colours, blocky like a Minecraft texture, not a photo.
                flat = np.stack([ndimage.median_filter(out[..., c], size=max(3, px // 16) | 1) for c in range(3)], -1)
                out = posterise(flat.reshape(-1, 3), PATTERN_COLOURS).reshape(out.shape)
            else:
                log(f"[tex] {side} wall: too few clean patches to grow a pattern from; plain")
                patterned[side] = False
        if style is None and not patterned.get(side):
            out, lbl, tiles = compose_wall(img, seen, paints_rgb, paints_lab, default_paint, px,
                                           "items" if detail in ("pattern", "style") else detail)
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
                walls[side]["dither"] = False  # flat map colours: dithering speckles them
            if detail == "style":
                walls[side]["dither"] = style is not None
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


def synthesize_wall(img, seen, pool, px, rng, top_k=5):
    """Fills the unseen part of a wall image with patches from `pool` (full, low) exemplars, keeping every seen
    pixel. Patches are one block wide; each is chosen among the `top_k` best matches to what is already known
    in its window (photo or earlier patches), so it continues the pattern without copying one patch over and
    over. Returns the filled image."""
    full, low = pool
    B = px
    ds = max(1, B // 16)
    ov = max(2, B // 4)
    step = B - ov
    h, w = seen.shape
    out = img.astype(np.float32).copy()
    known = seen.copy()
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
                best = np.argpartition(d, min(top_k, len(d) - 1))[:top_k]
                i = int(rng.choice(best))
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


STYLE_PLAIN_DE = 6.0     # a room style whose heights differ less than this (CIELAB) ...
STYLE_PLAIN_GRAIN = 3.0  # ... and whose texture is weaker than this (std of luminance) is plain
STYLE_SAME_DE = 12.0     # walls this close (CIELAB) to the most-seen wall share the room style; others are accents
BAND_DE = 4.0            # rows within this of their band's colour (CIELAB) belong to it; finer is noise


def _bands(lab, beta):
    """Start indices of the optimal piecewise-constant segmentation of rows `lab` (n, 3): least squared error
    plus `beta` per band (optimal partitioning, Jackson et al., IEEE Signal Processing Letters 2005)."""
    n = len(lab)
    c1 = np.vstack([np.zeros(3), np.cumsum(lab, 0)])
    c2 = np.r_[0.0, np.cumsum((lab ** 2).sum(1))]
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, int)
    for t in range(1, n + 1):
        s0 = np.arange(t)
        cnt = t - s0
        sse = (c2[t] - c2[s0]) - ((c1[t] - c1[s0]) ** 2).sum(1) / cnt
        cost = F[s0] + sse + beta
        j = int(np.argmin(cost))
        F[t], last[t] = cost[j], s0[j]
    out, t = [], n
    while t > 0:
        out.append(int(last[t]))
        t = last[t]
    return out[::-1]


def room_style(walls, px):
    """One wall style from all walls: (profile (rows, 3) colour per texture row, seamless px x px texture tile
    (residual around the profile), stats dict). walls: [(image, clean mask)], all of the same height.
    The style is the room's dominant one: the look shared by the most walls (seen area breaks ties), and every
    wall that looks like it. An accent wall (dark acoustic panels, a feature colour) is left out instead of
    averaged into a muddy mix - even when it is the wall the photos point at most."""
    walls = [w for w in walls if w[1].sum() >= px * px]
    if not walls:
        return None
    lab = [srgb_to_lab(np.median(img[m], axis=0)[None])[0] for img, m in walls]
    near = [[j for j in range(len(walls)) if np.linalg.norm(lab[i] - lab[j]) < STYLE_SAME_DE] for i in range(len(walls))]
    best = max(range(len(walls)), key=lambda i: (len(near[i]), sum(walls[j][1].sum() for j in near[i])))
    walls = [walls[j] for j in near[best]]
    rows = walls[0][0].shape[0]
    med = np.full((rows, 3), np.nan)
    for r in range(rows):
        # Median within each wall (robust to a poster), then every wall an equal vote: walls lit differently in
        # the photos average out instead of the wall seen most deciding the colour.
        per_wall = [np.median(img[r][m[r]], axis=0) for img, m in walls if m[r].sum() >= max(2, px // 4)]
        if per_wall:
            med[r] = np.mean(per_wall, axis=0)
    known = np.isfinite(med[:, 0])
    if known.sum() < max(2, rows // 20):
        return None
    for c in range(3):  # heights nobody saw continue from the nearest heights that were seen
        med[:, c] = np.interp(np.arange(rows), np.flatnonzero(known), med[known, c])
    # Flat bands (white wall, trim, wainscot): the best piecewise-constant fit of the colour over height, where a
    # new band must explain more than BAND_DE over half a block. A trim line (strong, thin) gets its own band;
    # noise between heights (rows seen by different photos) does not, so every wall is smooth.
    med = ndimage.median_filter(med, size=(max(3, px // 8) | 1, 1), mode="nearest")
    ds = max(1, px // 16)
    lab = srgb_to_lab(med)
    n = rows // ds
    starts = _bands(lab[:n * ds].reshape(n, ds, 3).mean(1), BAND_DE ** 2 * (px / 2) / ds)
    bounds = [b * ds for b in starts] + [rows]
    out = np.empty_like(med)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = np.median(med[a:b], axis=0)
    med = out
    profile = ndimage.gaussian_filter1d(med, max(1.0, px / 32), axis=0, mode="nearest").astype(np.float32)
    prof_lab = srgb_to_lab(profile)
    cands = []
    for img, m in walls:
        frac = ndimage.uniform_filter(m.astype(np.float32), px, mode="constant")
        for y in range(0, img.shape[0] - px + 1, max(1, px // 2)):
            for x in range(0, img.shape[1] - px + 1, max(1, px // 2)):
                if frac[y + px // 2, x + px // 2] >= 0.97:
                    patch = img[y:y + px, x:x + px]
                    # Only bare surface: no thermostat, outlet or poster (pixels far from the wall at that height).
                    if (_de(srgb_to_lab(patch), prof_lab[y:y + px, None, :]) > ITEM_DE).mean() > 0.02:
                        continue
                    res = patch - profile[y:y + px, None, :]
                    # Only fine grain: the patch's own brightness and slow gradients (its photo's lighting) would
                    # repeat as stripes once tiled.
                    cands.append(res - ndimage.gaussian_filter(res, (px / 4, px / 4, 0), mode="reflect"))
    if cands:
        energy = np.array([np.std(c @ np.array([0.299, 0.587, 0.114])) for c in cands])
        # The surface's own grain: from the smoother part of the candidates, so edges and shading do not win.
        tile = cands[int(np.argmin(np.abs(energy - np.percentile(energy, 30))))].astype(np.float32)
        # Seamless: cross-fade with itself shifted half a block, weighted towards the original in the middle.
        t = np.sin(np.pi * (np.arange(px) + 0.5) / px) ** 2
        wy, wx = t[:, None, None], t[None, :, None]
        half = px // 2
        tile = wx * tile + (1 - wx) * np.roll(tile, half, axis=1)
        tile = wy * tile + (1 - wy) * np.roll(tile, half, axis=0)
        grain = float(np.percentile(energy, 30))
    else:
        tile, grain = np.zeros((px, px, 3), np.float32), 0.0
    # Full CIELAB difference, as for the bands: a white-over-grey wainscot differs mostly in lightness.
    spread = float(np.percentile(np.linalg.norm(prof_lab - np.median(prof_lab, axis=0), axis=-1), 95))
    return profile, tile, {"height_spread": spread, "grain": grain, "patches": len(cands), "walls": len(walls)}


def _lab_sample(rgb):
    return srgb_to_lab(rgb[:: max(1, len(rgb) // 20000)])

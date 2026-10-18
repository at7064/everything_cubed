"""everything_cubed spectate worker: a live LiDAR stream from the iOS app -> block changes, about 5 times a second.

Usage (started by the mod for `/everything_cubed phone` + Spectate in the app; not meant to be run by hand):
  python spectate_analyzer.py [--scale BLOCKS_PER_METRE] [--mc-jar client.jar]

stdin:  the phone's `frame` messages, forwarded unchanged by the mod (framing as in ios/PROTOCOL.md). Each frame may
        carry a `person` blob: u8 per depth pixel, nonzero where the phone saw a person.
stdout: one JSON object per line:
        {"type": "update", "changes": [[x, y, z, "minecraft:block"], ...], "people": [{"id", "x", "y", "z"}, ...],
         "baseline": bool}  exactly one per frame read, so the mod knows when to send the next one;
        {"type": "status", "level", "text"}  for chat and the app's log.
stderr: log lines.

Coordinates are blocks in the spectate frame: y = 0 is the floor block, +z is where the phone looked when the scan
started (straight along the room's nearest wall direction) and +x is to its left, like Minecraft's east is to the
left of south. The mod puts the origin at the player.

The pipeline is lidar_analyzer's, per frame: frame_geometry() unprojects ARKit depth with the frame's pose; the room's
axes, floor, ceiling, walls and table tops are measured once from the first frames (manhattan_axes, floor_ceiling,
find_walls, find_tables) and label_frame() labels every later frame against them; blocks are coloured with the same
surface palette as build.

Only confident changes reach the world. The first BASELINE_FRAMES frames are the initial scan: everything they show
is committed as it is. After that a block changes only when COMMIT_FRAMES frames in a row show the change clearly
with high-confidence depth (and RECOLOUR_FRAMES in a row for a different colour). A frame that shows a cell both
ways, or a low-confidence view of it, resets that cell's count; unseen cells keep their counts. People are never
built: their pixels (and a margin) are left out, and they are reported as tracked positions instead.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from lidar_analyzer import (L, LABELS, find_tables, find_walls, floor_ceiling, frame_geometry,  # noqa: E402
                            label_frame, manhattan_axes)

BASELINE_FRAMES = 15      # the initial scan: about 3 s at 5 frames a second
COMMIT_FRAMES = 5         # frames in a row that must show a known block appear or disappear
NEW_FRAMES = 3            # ... or a block where nothing was known yet
RECOLOUR_FRAMES = 10      # frames in a row that must agree on a different block
EVIDENCE_DEPTH = 5.0      # metres; live changes only from depth this close (and ARKit confidence high)
MIN_HITS = 2              # sampled depth pixels in a sub-cell for "something is here" this frame
MIN_MISSES = 2            # sampled rays through a sub-cell for "this is empty" this frame
STRIDE = 2                # sample every 2nd depth pixel in both directions
SUB = 3                   # evidence sub-cells per block along each axis
PERSON_MARGIN = 3         # depth pixels around a person mask that are not evidence either
RANGE = 20.0              # metres around the start point that are mirrored
MAX_HEIGHT = 8.0          # metres above the floor that are mirrored
PERSON_CELL = 0.2         # metres: grid for grouping person points into people
PERSON_MIN_PIXELS = 25
PERSON_MATCH = 0.8        # metres a person may move between frames and keep their id
PERSON_SHOW = 2           # frames a person must be seen before they are reported
PERSON_KEEP = 1.0         # seconds a person stays after they were last seen
AUTO_SCALE = 1.25         # blocks per metre, as build uses for LiDAR rooms (measured size)
SPECIAL = {"window": "minecraft:glass", "ceiling_light": "minecraft:sea_lantern"}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def read_message(stream):
    """One message in the phone protocol's framing, as (header, {blob name: bytes}); None at the end."""
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    (hl,) = struct.unpack(">I", raw)
    header = json.loads(stream.read(hl))
    (pl,) = struct.unpack(">I", stream.read(4))
    payload = stream.read(pl)
    blobs, off = {}, 0
    for b in header.get("blobs", []):
        blobs[b["name"]] = payload[off:off + b["length"]]
        off += b["length"]
    return header, blobs


def decode_frame(header, blobs, width=320):
    """The frame as lidar_analyzer.frame_geometry returns it, plus its person mask; None without depth."""
    from PIL import Image

    if "depth" not in blobs or "depthSize" not in header or "image" not in blobs:
        return None
    dw, dh = header["depthSize"]
    fmt = next((b.get("format") for b in header["blobs"] if b["name"] == "depth"), "f16")
    depth = np.frombuffer(blobs["depth"], dtype="<f2" if fmt == "f16" else "<f4").reshape(dh, dw)
    conf = np.frombuffer(blobs["confidence"], np.uint8).reshape(dh, dw) if "confidence" in blobs else None
    img = Image.open(io.BytesIO(blobs["image"]))
    img.draft("RGB", (width, width))   # JPEG decodes at 1/2, 1/4 or 1/8 size: much faster
    fr = frame_geometry(header, depth, conf, img, width=width)
    person = np.zeros((dh, dw), bool)
    if "person" in blobs and len(blobs["person"]) == dw * dh:
        person = np.frombuffer(blobs["person"], np.uint8).reshape(dh, dw) > 0
    fr["person"] = person
    fr["time"] = float(header.get("time", time.time()))
    return fr


class Spectator:
    def __init__(self, palette, scale=None):
        self.palette = palette
        self.s = scale or AUTO_SCALE
        self.baseline = []
        self.ready = False
        self.blocks = ["minecraft:air"]           # block index -> id; index 0 is air
        self.block_of = {"minecraft:air": 0}
        self.tracks, self.next_id = [], 1

    # ------------------------------------------------------------------ room frame (once, from the initial scan)

    def setup(self, frames):
        ex, ez = manhattan_axes(frames, None, log)
        self.floor_y, self.ceil_y = floor_ceiling(frames, None, log)
        self.ex, self.ez = ex, ez
        self.walls = find_walls(frames, ex, ez, self.floor_y, self.ceil_y, log)
        self.tables = find_tables(frames, self.floor_y, log)
        # +z is the room axis nearest to where the phone first looked, +x its left (up x forward).
        look = frames[0]["c2w"][:3, 2]   # OpenCV camera: looks down +Z
        self.fwd = max([ex, -ex, ez, -ez], key=lambda a: float(a @ look))
        self.left = np.cross([0.0, 1.0, 0.0], self.fwd)
        self.origin = np.array([frames[0]["C"][0], self.floor_y, frames[0]["C"][2]])
        self.R = int(np.ceil(RANGE * self.s))
        rows = int(np.ceil(MAX_HEIGHT * self.s)) + 2   # block rows -1 .. rows-2
        self.bshape = (2 * self.R, rows, 2 * self.R)
        self.shape = tuple(SUB * n for n in self.bshape)
        n = int(np.prod(self.shape))
        # Per sub-cell. Evidence is kept SUB x SUB x SUB times finer than blocks: a block that is only partly
        # filled (a 1 m crate in a 0.8 m block row) still has sub-cells that rays pass through, and those must
        # not vote the filled ones away.
        self.committed = np.full(n, -1, np.int32)   # -1 unknown, 0 empty, >0 block index
        self.streak = np.zeros(n, np.int16)         # >0: frames in a row showing a block, <0: showing empty
        self.cand = np.zeros(n, np.int32)           # block a new or recoloured sub-cell would get
        self.cand_streak = np.zeros(n, np.int16)
        self.shown = np.full(int(np.prod(self.bshape)), -1, np.int32)   # per block: what the world has
        o = np.arange(SUB)
        oi, oj, ok = np.meshgrid(o, o, o, indexing="ij")
        self.sub_offsets = ((oi * self.shape[1] + oj) * self.shape[2] + ok).ravel()

    def local(self, P):
        d = P - self.origin
        return np.stack([d @ self.left, d[..., 1], d @ self.fwd], -1)

    def cell_index(self, loc):
        """Flat sub-cell index of local metric points; -1 outside the mirrored region. Block row 0 is centred on
        the floor (like build's floor layer), so the floor surface is in the middle of a sub-cell row."""
        i = np.floor(loc[..., 0] * self.s * SUB).astype(np.int64) + self.R * SUB
        j = np.floor((loc[..., 1] * self.s + 0.5) * SUB).astype(np.int64) + SUB
        k = np.floor(loc[..., 2] * self.s * SUB).astype(np.int64) + self.R * SUB
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1]) & (k >= 0) & (k < self.shape[2])
        return np.where(ok, (i * self.shape[1] + j) * self.shape[2] + k, -1)

    def block_of_cell(self, idx):
        i, rest = np.divmod(idx, self.shape[1] * self.shape[2])
        j, k = np.divmod(rest, self.shape[2])
        return ((i // SUB) * self.bshape[1] + j // SUB) * self.bshape[2] + k // SUB

    def block_coords(self, b):
        i, rest = np.divmod(b, self.bshape[1] * self.bshape[2])
        j, k = np.divmod(rest, self.bshape[2])
        return i - self.R, j - 1, k - self.R

    def block_index(self, ids):
        out = np.empty(len(ids), np.int32)
        for n, b in enumerate(ids):
            if b not in self.block_of:
                self.block_of[b] = len(self.blocks)
                self.blocks.append(b)
            out[n] = self.block_of[b]
        return out

    def blocks_for(self, labels, rgb):
        """Block per cell from its label and mean colour: the build's surface palette, glass for windows."""
        from palette import srgb_to_lab

        lab = srgb_to_lab(rgb)
        d = self.palette.lab[None] - lab[:, None]
        d[..., 0] *= 0.75
        ids = [self.palette.ids[i] for i in np.argmin((d ** 2).sum(-1), 1)]
        for n, l in enumerate(labels):
            ids[n] = SPECIAL.get(LABELS[l], ids[n])
        return self.block_index(ids)

    # ------------------------------------------------------------------ one frame -> evidence per sub-cell

    def observe(self, fr, live):
        """Sub-cells this frame clearly shows filled (with their block), clearly empty, and seen but unclear."""
        from scipy import ndimage

        lab, moved = label_frame(fr, self.ex, self.ez, self.floor_y, self.ceil_y, self.walls, None, self.tables)
        person = ndimage.binary_dilation(fr["person"], iterations=PERSON_MARGIN) if fr["person"].any() else fr["person"]
        ev = fr["valid"] & ~person
        if live:
            ev &= (fr["conf"] >= 2) & (fr["depth"] < EVIDENCE_DEPTH)
        sub = np.zeros_like(ev)
        sub[::STRIDE, ::STRIDE] = True
        ev &= sub
        # Window pixels were moved onto the wall where their ray crossed it: glass there, not the street outside.
        P = moved[ev]
        n = len(self.committed)
        hit = self.cell_index(self.local(P))
        keep = hit >= 0
        hits = np.bincount(hit[keep], minlength=n)

        # Rays from the camera to each sample: the sub-cells they cross before the surface are empty.
        C = fr["C"]
        v = P - C
        length = np.linalg.norm(v, axis=1)
        step = 0.5 / (self.s * SUB)
        margin = 1.0 / (self.s * SUB)
        misses = np.zeros(n, np.int64)
        if len(length):
            ks = np.arange(1, int(np.ceil(length.max() / step)) + 1) * step
            r, k = np.nonzero(ks[None, :] < (length - margin)[:, None])
            miss = self.cell_index(self.local(C + v[r] / length[r, None] * ks[k][:, None]))
            misses = np.bincount(miss[miss >= 0], minlength=n) / 2.0   # about 2 samples per ray and sub-cell

        filled = np.flatnonzero((hits >= MIN_HITS) & (hits >= 2 * misses))
        empty = np.flatnonzero((misses >= MIN_MISSES) & (hits == 0))
        seen = np.flatnonzero((hits > 0) | (misses > 0))
        unclear = np.setdiff1d(seen, np.union1d(filled, empty), assume_unique=True)

        # Majority label and mean colour of the filled sub-cells.
        hc, labs, cols = hit[keep], lab[ev][keep], fr["rgb_d"][ev][keep].astype(np.float64)
        on = np.isin(hc, filled)
        hc, labs, cols = hc[on], labs[on], cols[on]
        cells, inv = np.unique(hc, return_inverse=True)
        votes = np.bincount(inv * len(LABELS) + labs, minlength=len(cells) * len(LABELS)).reshape(len(cells), len(LABELS))
        votes[:, L["other"]] //= 2   # unnamed pixels lose ties to named ones
        cnt = np.bincount(inv, minlength=len(cells))
        rgb = np.stack([np.bincount(inv, cols[:, c], len(cells)) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        return cells, self.blocks_for(votes.argmax(1), rgb), empty, unclear

    # ------------------------------------------------------------------ evidence -> committed sub-cells

    def commit_baseline(self):
        """The initial scan is taken as it is: filled where frames saw something (at least as often as they saw
        through it), empty where they saw through it."""
        n = len(self.committed)
        filled_n = np.zeros(n, np.int32)
        empty_n = np.zeros(n, np.int32)
        last_block = np.zeros(n, np.int32)
        for fr in self.baseline:
            f, b, e, _ = self.observe(fr, live=False)
            filled_n[f] += 1
            empty_n[e] += 1
            last_block[f] = b
        solid = (filled_n >= 1) & (filled_n >= empty_n)
        air = (empty_n >= 1) & ~solid
        self.committed[solid] = last_block[solid]
        self.committed[air] = 0
        self.baseline = []
        return np.flatnonzero(solid | air)

    def update(self, fr):
        filled, blocks, empty, unclear = self.observe(fr, live=True)
        cur = self.committed
        # Unclear this frame: whatever was building up starts over. Unseen sub-cells keep their counts.
        self.streak[unclear] = 0
        self.cand_streak[unclear] = 0

        known = cur[filled] > 0
        # Filled where it was empty (or unknown).
        grow, gb = filled[~known], blocks[~known]
        s = np.maximum(self.streak[grow], 0) + 1
        self.streak[grow] = s
        self.cand[grow] = gb
        appear = grow[s >= np.where(cur[grow] < 0, NEW_FRAMES, COMMIT_FRAMES)]
        # Filled as before: only a different block, agreed on by RECOLOUR_FRAMES frames in a row, changes it.
        stay, nb = filled[known], blocks[known]
        self.streak[stay] = 0
        diff = nb != cur[stay]
        cs = np.where(diff & (self.cand[stay] == nb), self.cand_streak[stay] + 1, np.where(diff, 1, 0))
        self.cand[stay] = np.where(diff, nb, self.cand[stay])
        self.cand_streak[stay] = cs
        recolour = stay[cs >= RECOLOUR_FRAMES]
        # Empty where it was filled (or unknown).
        was_air = cur[empty] == 0
        self.streak[empty[was_air]] = 0
        gone = empty[~was_air]
        s = np.minimum(self.streak[gone], 0) - 1
        self.streak[gone] = s
        vanish = gone[-s >= np.where(cur[gone] < 0, NEW_FRAMES, COMMIT_FRAMES)]

        cur[appear] = self.cand[appear]
        cur[recolour] = self.cand[recolour]
        cur[vanish] = 0
        changed = np.concatenate([appear, recolour, vanish])
        self.streak[changed] = 0
        self.cand_streak[changed] = 0
        return changed

    def changes(self, cells):
        """World changes for the blocks holding `cells`: a block is solid when any of its sub-cells is (the most
        common block among them), empty when none is and some are known empty."""
        out = []
        if not len(cells):
            return out
        bl = np.unique(self.block_of_cell(cells))
        i, rest = np.divmod(bl, self.bshape[1] * self.bshape[2])
        j, k = np.divmod(rest, self.bshape[2])
        base = ((i * SUB) * self.shape[1] + j * SUB) * self.shape[2] + k * SUB
        vals = self.committed[base[:, None] + self.sub_offsets[None]]
        for b, row in zip(bl, vals):
            solid = row[row > 0]
            new = int(np.bincount(solid).argmax()) if len(solid) else 0 if (row == 0).any() else -1
            if new < 0 or new == self.shown[b]:
                continue
            self.shown[b] = new
            x, y, z = self.block_coords(b)
            out.append([int(x), int(y), int(z), self.blocks[new]])
        return out

    # ------------------------------------------------------------------ people

    def people(self, fr):
        from scipy import ndimage

        found = []
        m = fr["person"] & fr["valid"]
        if m.sum() >= PERSON_MIN_PIXELS:
            loc = self.local(fr["P"][m])
            gi = np.floor(loc[:, [0, 2]] / PERSON_CELL).astype(int)
            lo = gi.min(0)
            gi -= lo
            grid = np.zeros(gi.max(0) + 1, np.int32)
            np.add.at(grid, (gi[:, 0], gi[:, 1]), 1)
            comp, nc = ndimage.label(grid > 0, np.ones((3, 3)))
            pid = comp[gi[:, 0], gi[:, 1]]
            for c in range(1, nc + 1):
                sel = pid == c
                if sel.sum() < PERSON_MIN_PIXELS or np.ptp(loc[sel, 1]) < 0.4:
                    continue
                found.append(np.median(loc[sel][:, [0, 2]], axis=0))
        now = fr["time"]
        free = list(range(len(found)))
        for t in sorted(self.tracks, key=lambda t: -t["seen"]):
            if not free:
                break
            d = [np.linalg.norm(found[f] - t["xz"]) for f in free]
            best = int(np.argmin(d))
            if d[best] <= PERSON_MATCH:
                t["xz"] = found[free.pop(best)]
                t["seen"] += 1
                t["last"] = now
        for f in free:
            self.tracks.append({"id": self.next_id, "xz": found[f], "seen": 1, "last": now})
            self.next_id += 1
        self.tracks = [t for t in self.tracks if now - t["last"] <= PERSON_KEEP]
        # Standing on the floor block (y = 0), so their feet are at y = 1.
        return [{"id": t["id"], "x": round(float(t["xz"][0] * self.s), 3), "y": 1.0,
                 "z": round(float(t["xz"][1] * self.s), 3)} for t in self.tracks if t["seen"] >= PERSON_SHOW]

    # ------------------------------------------------------------------ main step

    def step(self, header, blobs):
        if header.get("tracking", "normal") != "normal":
            return {"type": "update", "changes": [], "people": self.people_kept()}
        fr = decode_frame(header, blobs)
        if fr is None:
            return {"type": "update", "changes": [], "people": self.people_kept()}
        if not self.ready:
            self.baseline.append(fr)
            if len(self.baseline) < BASELINE_FRAMES:
                if len(self.baseline) % 5 == 0:
                    emit({"type": "status", "level": "progress",
                          "text": f"Spectate: initial scan {len(self.baseline)}/{BASELINE_FRAMES}"})
                return {"type": "update", "changes": [], "people": []}
            t0 = time.time()
            self.setup(self.baseline)
            changes = self.changes(self.commit_baseline())
            self.ready = True
            solid = sum(c[3] != "minecraft:air" for c in changes)
            log(f"[spectate] initial scan: {solid} blocks, {len(changes) - solid} empty in {time.time() - t0:.1f}s")
            emit({"type": "status", "level": "success",
                  "text": f"Spectate: initial scan built ({solid} blocks). Changes now update live."})
            return {"type": "update", "baseline": True, "changes": changes, "people": self.people(fr)}
        return {"type": "update", "changes": self.changes(self.update(fr)), "people": self.people(fr)}

    def people_kept(self):
        return [{"id": t["id"], "x": round(float(t["xz"][0] * self.s), 3), "y": 1.0,
                 "z": round(float(t["xz"][1] * self.s), 3)} for t in self.tracks if t["seen"] >= PERSON_SHOW]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, help=f"blocks per metre (default {AUTO_SCALE}, the LiDAR build's)")
    ap.add_argument("--mc-jar", help="Minecraft client jar, used to read real block texture colours")
    opts, unknown = ap.parse_known_args(argv)
    if unknown:
        log(f"[spectate] ignoring options for the photo analyzer: {' '.join(unknown)}")
    from palette import surface_palette

    spec = Spectator(surface_palette(opts.mc_jar, Path(__file__).resolve().parent), opts.scale)
    stdin = sys.stdin.buffer
    while True:
        msg = read_message(stdin)
        if msg is None:
            break
        header, blobs = msg
        if header.get("type") != "frame":
            continue
        t0 = time.time()
        try:
            out = spec.step(header, blobs)
        except Exception as ex:   # one bad frame must not end spectating
            import traceback
            traceback.print_exc()
            out = {"type": "update", "changes": [], "people": [], "error": str(ex)}
        out["ms"] = round((time.time() - t0) * 1000)
        emit(out)


if __name__ == "__main__":
    main()

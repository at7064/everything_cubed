"""everything_cubed spectate worker: a live LiDAR stream from the iOS app -> changes to a room already built, about 5
times a second.

Usage (started by the mod when the app spectates a scan it has built; not meant to be run by hand):
  python spectate_analyzer.py --plan <scan>/plan.json --baseline <file> [--mc-jar client.jar]

Spectate starts from a scan: the room has been scanned with LiDAR and built, and that build is the starting state,
taken as certain. The phone relocalises to the scan's ARKit world map, so its live frames are in the scan's
coordinates, and the plan's "arkit" frame (written by lidar_analyzer) maps them onto the plan's blocks.
--baseline is a JSON list of the interior blocks that are solid in the world right now, [[x, y, z], ...] in plan
block coordinates (y = 1 is the first block above the floor), written by the mod.

stdin:  the phone's `frame` messages, forwarded unchanged by the mod (framing as in ios/PROTOCOL.md). Each frame may
        carry a `person` blob: u8 per depth pixel, nonzero where the phone saw a person.
stdout: one JSON object per line:
        {"type": "update", "changes": [[x, y, z, "minecraft:block"], ...], "people": [{"id", "x", "y", "z"}, ...]}
        exactly one per frame read, so the mod knows when to send the next one;
        {"type": "status", "level", "text"}  for chat and the app's log.
stderr: log lines.

Only the inside of the house changes (furniture and things on the floor); the floor, walls and ceiling stay as built.
The pipeline is lidar_analyzer's: frame_geometry() unprojects each frame's ARKit depth with its pose, and new blocks
are coloured with the same surface palette as build.

Only confident changes reach the world. Evidence is kept on 3 x 3 x 3 sub-cells per block. A sub-cell changes only
when COMMIT_FRAMES frames in a row show the change clearly with high-confidence depth closer than EVIDENCE_DEPTH; a
frame that shows it both ways, or only at low confidence, resets its count, and sub-cells out of view keep theirs.
A built block is removed only when most of its sub-cells (EMPTY_SHARE) are clearly empty, so a chair that fills half
its block stays, and anything unclear stays as built. A new block appears where the world has air and a sub-cell is
clearly filled. People are never built: their pixels (and a margin) are left out, and they are reported as tracked
positions instead (the mod shows villagers).
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

from lidar_analyzer import frame_geometry  # noqa: E402

COMMIT_FRAMES = 5         # frames in a row that must clearly show a sub-cell change
RECOLOUR_FRAMES = 10      # frames in a row that must agree on a different block (for blocks spectate placed)
EVIDENCE_DEPTH = 5.0      # metres; changes only from depth this close (and ARKit confidence high)
MIN_HITS = 2              # sampled depth pixels in a sub-cell for "something is here" this frame
MIN_MISSES = 2            # sampled rays through a sub-cell for "this is empty" this frame
STRIDE = 2                # sample every 2nd depth pixel in both directions
SUB = 3                   # evidence sub-cells per block along each axis
EMPTY_SHARE = 2 / 3       # share of a built block's sub-cells that must be clearly empty to remove it
SURFACE_MARGIN = 0.1      # metres: depth this close to the floor, walls or ceiling is that surface, not a new thing
PERSON_MARGIN = 3         # depth pixels around a person mask that are not evidence either
PERSON_CELL = 0.2         # metres: grid for grouping person points into people
PERSON_MIN_PIXELS = 25
PERSON_MATCH = 0.8        # metres a person may move between frames and keep their id
PERSON_SHOW = 2           # frames a person must be seen before they are reported
PERSON_KEEP = 1.0         # seconds a person stays after they were last seen
AIR, BUILT = 0, 1         # sub-cell states; block indices from 2 are blocks spectate places


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


class PlanFrame:
    """The plan's "arkit" frame: ARKit world metres -> the plan's blocks (see lidar_analyzer)."""

    def __init__(self, a):
        self.origin, self.R = np.asarray(a["origin"], float), np.asarray(a["R"], float)
        self.y_shift, self.f = a["y_shift"], a["f"]
        self.xmin, self.xmax, self.zmin, self.zmax, self.hm = a["xmin"], a["xmax"], a["zmin"], a["zmax"], a["hm"]
        self.W, self.D, self.H = a["W"], a["D"], a["H"]

    def room_metres(self, P):
        loc = (np.asarray(P, float) - self.origin) @ self.R.T
        loc[..., 1] -= self.y_shift
        return loc * self.f

    def blocks(self, P):
        """Fractional block coordinates (x, interior row, z); row 0 is the first air layer."""
        loc = self.room_metres(P)
        return np.stack([(loc[..., 0] - self.xmin) / (self.xmax - self.xmin) * self.W, loc[..., 1] / self.hm * self.H,
                         (loc[..., 2] - self.zmin) / (self.zmax - self.zmin) * self.D], -1)


class Spectator:
    def __init__(self, frame: PlanFrame, baseline, palette):
        self.fr = frame
        self.palette = palette
        self.blocks = ["minecraft:air", "built"]   # index -> block id; BUILT is whatever the build put there
        self.block_of = {b: i for i, b in enumerate(self.blocks)}
        self.tracks, self.next_id = [], 1
        self.bshape = (frame.W, frame.H, frame.D)
        self.shape = tuple(SUB * n for n in self.bshape)
        n = int(np.prod(self.shape))
        self.committed = np.full(n, AIR, np.int32)
        self.streak = np.zeros(n, np.int16)          # >0: frames in a row showing a block, <0: showing empty
        self.cand = np.zeros(n, np.int32)            # block a new or recoloured sub-cell would get
        self.cand_streak = np.zeros(n, np.int16)
        self.shown = np.full(int(np.prod(self.bshape)), AIR, np.int32)   # per block: what the world has
        o = np.arange(SUB)
        oi, oj, ok = np.meshgrid(o, o, o, indexing="ij")
        self.sub_offsets = ((oi * self.shape[1] + oj) * self.shape[2] + ok).ravel()
        for x, y, z in baseline:
            if 0 <= x < frame.W and 1 <= y <= frame.H and 0 <= z < frame.D:
                b = (x * self.bshape[1] + (y - 1)) * self.bshape[2] + z
                self.shown[b] = BUILT
                self.committed[self.first_sub(np.array([b]))[0] + self.sub_offsets] = BUILT

    def first_sub(self, b):
        i, rest = np.divmod(b, self.bshape[1] * self.bshape[2])
        j, k = np.divmod(rest, self.bshape[2])
        return ((i * SUB) * self.shape[1] + j * SUB) * self.shape[2] + k * SUB

    def cell_index(self, P):
        """Flat sub-cell index of world points; -1 outside the inside of the house."""
        b = np.floor(self.fr.blocks(P) * SUB).astype(np.int64)
        i, j, k = b[..., 0], b[..., 1], b[..., 2]
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1]) & (k >= 0) & (k < self.shape[2])
        return np.where(ok, (i * self.shape[1] + j) * self.shape[2] + k, -1)

    def block_of_cell(self, idx):
        i, rest = np.divmod(idx, self.shape[1] * self.shape[2])
        j, k = np.divmod(rest, self.shape[2])
        return ((i // SUB) * self.bshape[1] + j // SUB) * self.bshape[2] + k // SUB

    def block_index(self, ids):
        out = np.empty(len(ids), np.int32)
        for n, b in enumerate(ids):
            if b not in self.block_of:
                self.block_of[b] = len(self.blocks)
                self.blocks.append(b)
            out[n] = self.block_of[b]
        return out

    def blocks_for(self, rgb):
        """Block per sub-cell from its mean colour: the build's surface palette."""
        from palette import srgb_to_lab

        d = self.palette.lab[None] - srgb_to_lab(rgb)[:, None]
        d[..., 0] *= 0.75
        return self.block_index([self.palette.ids[i] for i in np.argmin((d ** 2).sum(-1), 1)])

    # ------------------------------------------------------------------ one frame -> evidence per sub-cell

    def observe(self, fr):
        """Sub-cells this frame clearly shows filled (with their block), clearly empty, and seen but unclear."""
        from scipy import ndimage

        person = ndimage.binary_dilation(fr["person"], iterations=PERSON_MARGIN) if fr["person"].any() else fr["person"]
        ev = fr["valid"] & ~person & (fr["conf"] >= 2) & (fr["depth"] < EVIDENCE_DEPTH)
        sub = np.zeros_like(ev)
        sub[::STRIDE, ::STRIDE] = True
        ev &= sub
        P = fr["P"][ev]
        n = len(self.committed)
        hit = self.cell_index(P)
        f, m, loc = self.fr, SURFACE_MARGIN, self.fr.room_metres(P)
        keep = ((hit >= 0) & (loc[:, 0] > f.xmin + m) & (loc[:, 0] < f.xmax - m) & (loc[:, 2] > f.zmin + m)
                & (loc[:, 2] < f.zmax - m) & (loc[:, 1] > m) & (loc[:, 1] < f.hm - m))
        hits = np.bincount(hit[keep], minlength=n)

        # Rays from the camera to each sample: the sub-cells they cross before the surface are empty.
        C = fr["C"]
        v = P - C
        length = np.linalg.norm(v, axis=1)
        sub_m = (self.fr.xmax - self.fr.xmin) / self.fr.W / SUB   # a sub-cell's size in metres
        misses = np.zeros(n)
        if len(length):
            ks = np.arange(1, int(np.ceil(length.max() / (0.5 * sub_m))) + 1) * 0.5 * sub_m
            r, k = np.nonzero(ks[None, :] < (length - sub_m)[:, None])
            miss = self.cell_index(C + v[r] / length[r, None] * ks[k][:, None])
            misses = np.bincount(miss[miss >= 0], minlength=n) / 2.0   # about 2 samples per ray and sub-cell

        filled = np.flatnonzero((hits >= MIN_HITS) & (hits >= 2 * misses))
        empty = np.flatnonzero((misses >= MIN_MISSES) & (hits == 0))
        seen = np.flatnonzero((hits > 0) | (misses > 0))
        unclear = np.setdiff1d(seen, np.union1d(filled, empty), assume_unique=True)

        hc, cols = hit[keep], fr["rgb_d"][ev][keep].astype(np.float64)
        on = np.isin(hc, filled)
        hc, cols = hc[on], cols[on]
        cells, inv = np.unique(hc, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(cells))
        rgb = np.stack([np.bincount(inv, cols[:, c], len(cells)) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        return cells, self.blocks_for(rgb) if len(cells) else np.zeros(0, np.int32), empty, unclear

    # ------------------------------------------------------------------ evidence -> committed sub-cells

    def update(self, fr):
        filled, blocks, empty, unclear = self.observe(fr)
        cur = self.committed
        # Unclear this frame: whatever was building up starts over. Unseen sub-cells keep their counts.
        self.streak[unclear] = 0
        self.cand_streak[unclear] = 0

        solid = cur[filled] != AIR
        # Filled where it was empty.
        grow, gb = filled[~solid], blocks[~solid]
        s = np.maximum(self.streak[grow], 0) + 1
        self.streak[grow] = s
        self.cand[grow] = gb
        appear = grow[s >= COMMIT_FRAMES]
        # Filled as before. Built blocks are never recoloured; blocks spectate placed are, after RECOLOUR_FRAMES
        # frames in a row agree on the new one.
        stay, nb = filled[solid], blocks[solid]
        self.streak[stay] = 0
        diff = (nb != cur[stay]) & (cur[stay] != BUILT)
        cs = np.where(diff & (self.cand[stay] == nb), self.cand_streak[stay] + 1, np.where(diff, 1, 0))
        self.cand[stay] = np.where(diff, nb, self.cand[stay])
        self.cand_streak[stay] = cs
        recolour = stay[cs >= RECOLOUR_FRAMES]
        # Empty where it was filled.
        was_air = cur[empty] == AIR
        self.streak[empty[was_air]] = 0
        gone = empty[~was_air]
        s = np.minimum(self.streak[gone], 0) - 1
        self.streak[gone] = s
        vanish = gone[-s >= COMMIT_FRAMES]

        cur[appear] = self.cand[appear]
        cur[recolour] = self.cand[recolour]
        cur[vanish] = AIR
        changed = np.concatenate([appear, recolour, vanish])
        self.streak[changed] = 0
        self.cand_streak[changed] = 0
        return changed

    def changes(self, cells):
        """World changes for the blocks holding `cells`. A new block appears where sub-cells are newly and clearly
        filled (in place of air, a block spectate placed, or a built block that is now mostly empty); a block goes
        when most of its sub-cells are clearly empty and nothing new is in it; whatever the build put there otherwise
        stays."""
        out = []
        if not len(cells):
            return out
        bl = np.unique(self.block_of_cell(cells))
        vals = self.committed[self.first_sub(bl)[:, None] + self.sub_offsets[None]]
        for b, row in zip(bl, vals):
            new_ones = row[row > BUILT]
            mostly_empty = (row == AIR).sum() >= EMPTY_SHARE * len(row)
            if len(new_ones) and (self.shown[b] != BUILT or mostly_empty):
                # Something new is clearly here. (It fills part of its block, so most of the block can be empty.)
                new = int(np.bincount(new_ones).argmax())
            elif mostly_empty and not len(new_ones):
                new = AIR
            else:
                continue
            if new == self.shown[b]:
                continue
            self.shown[b] = new
            i, rest = divmod(int(b), self.bshape[1] * self.bshape[2])
            j, k = divmod(rest, self.bshape[2])
            out.append([i, j + 1, k, self.blocks[new]])
        return out

    # ------------------------------------------------------------------ people

    def people(self, fr):
        from scipy import ndimage

        found = []
        m = fr["person"] & fr["valid"]
        if m.sum() >= PERSON_MIN_PIXELS:
            loc = self.fr.room_metres(fr["P"][m])
            gi = np.floor(loc[:, [0, 2]] / PERSON_CELL).astype(int)
            gi -= gi.min(0)
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
        return self.people_kept()

    def people_kept(self):
        """Tracked people in plan block coordinates, standing on the floor (y = 1)."""
        f = self.fr
        return [{"id": t["id"], "x": round(float((t["xz"][0] - f.xmin) / (f.xmax - f.xmin) * f.W), 3), "y": 1.0,
                 "z": round(float((t["xz"][1] - f.zmin) / (f.zmax - f.zmin) * f.D), 3)}
                for t in self.tracks if t["seen"] >= PERSON_SHOW]

    # ------------------------------------------------------------------ main step

    def step(self, header, blobs):
        # Before the phone has relocalised to the scan (or while tracking is poor) its frames are not in the
        # scan's coordinates: they are no evidence.
        if header.get("tracking", "normal") != "normal":
            return {"type": "update", "changes": [], "people": self.people_kept()}
        fr = decode_frame(header, blobs)
        if fr is None:
            return {"type": "update", "changes": [], "people": self.people_kept()}
        return {"type": "update", "changes": self.changes(self.update(fr)), "people": self.people(fr)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path, help="plan.json of the built LiDAR scan")
    ap.add_argument("--baseline", required=True, type=Path, help="JSON list of the solid interior blocks now")
    ap.add_argument("--mc-jar", help="Minecraft client jar, used to read real block texture colours")
    opts, unknown = ap.parse_known_args(argv)
    if unknown:
        log(f"[spectate] ignoring options for the photo analyzer: {' '.join(unknown)}")
    from palette import surface_palette

    plan = json.loads(opts.plan.read_text())
    if "arkit" not in plan:
        emit({"type": "status", "level": "error",
              "text": "This scan was built before spectate existed (or is not a LiDAR scan). Build it again first."})
        return 2
    frame = PlanFrame(plan["arkit"])
    baseline = json.loads(opts.baseline.read_text())
    spec = Spectator(frame, baseline, surface_palette(opts.mc_jar, Path(__file__).resolve().parent))
    log(f"[spectate] {frame.W} x {frame.D} x {frame.H} room, {len(baseline)} built block(s) inside")
    emit({"type": "status", "level": "success",
          "text": "Spectate: point the phone at the scanned room. Changes you can clearly see update live."})
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

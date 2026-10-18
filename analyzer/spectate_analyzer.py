"""everything_cubed spectate worker: Apple RoomPlan's live room model from the iOS app -> furniture moves in a room
already scanned and built.

Usage (started by the mod when the app spectates a scan it has built; not meant to be run by hand):
  python spectate_analyzer.py --plan <scan>/plan.json [--mc-jar client.jar]

Spectate starts from a LiDAR scan that has been built: the build is the starting state, taken as certain. The app
keeps the scan's RoomPlan session running, so RoomPlan's live objects (measured with the LiDAR) are in the scan's
coordinates, and the plan's "arkit" frame (written by lidar_analyzer) maps them onto the plan's blocks.

Only whole pieces of furniture change, and only from RoomPlan's objects; no single blocks are ever placed:
- every built piece of furniture is linked to the RoomPlan object of the scan it came from (room.json);
- a piece moves when RoomPlan places that object clearly elsewhere (more than MOVE_M or MOVE_DEG from where it was),
  onto a different footprint, the same one for MOVE_UPDATES room updates in a row. Its new footprint, height and
  facing come from roomplan.place_object, exactly as build places furniture; its material and colour stay;
- a new piece is added when RoomPlan identifies a new object with medium or high confidence, stable for
  MOVE_UPDATES updates;
- anything RoomPlan is unsure of, or stops reporting, stays as built.
People are found in the camera frames (the `person` mask the phone sends with LiDAR depth) and reported as tracked
positions, which the mod shows as villagers.

stdin:  the phone's `room` messages (RoomPlan's live CapturedRoom as JSON in the `room` blob) and `frame` messages,
        forwarded by the mod (framing as in ios/PROTOCOL.md).
stdout: one JSON object per line:
        {"type": "update", "moves": [{"from": object, "to": object}, ...], "adds": [object, ...],
         "people": [{"id", "x", "y", "z"}, ...]}  exactly one per message read, so the mod knows when to send the next;
        objects are plan "objects" entries (plan block coordinates);
        {"type": "status", "level", "text"}  for chat and the app's log.
stderr: log lines.
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

from lidar_analyzer import ROOMPLAN_OBJECTS, _mat, frame_geometry  # noqa: E402
from roomplan import Room, place_object  # noqa: E402

MOVE_M = 0.25             # metres RoomPlan must place an object away from where it was for it to have moved
MOVE_DEG = 25.0           # ... or degrees it must have turned
MOVE_UPDATES = 5          # room updates in a row that must agree on the new footprint (about a second)
MATCH_M = 1.5             # metres: a live RoomPlan object this close to a scanned one of its kind is that one
CONFIDENT = ("medium", "high")
PERSON_CELL = 0.2         # metres: grid for grouping person points into people
PERSON_MIN_PIXELS = 25
PERSON_MATCH = 0.8        # metres a person may move between frames and keep their id
PERSON_SHOW = 2           # frames a person must be seen before they are reported
PERSON_KEEP = 1.0         # seconds a person stays after they were last seen
KEEP_FIELDS = ("material", "color")   # what a moved piece keeps from the built one
SAME_KIND = {"coffee_table": "table"}   # a coffee table and a table are matched as one kind


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
    """The plan's "arkit" frame: ARKit world metres -> the plan's room metres and blocks (see lidar_analyzer)."""

    def __init__(self, a):
        self.origin, self.R = np.asarray(a["origin"], float), np.asarray(a["R"], float)
        self.y_shift, self.f = a["y_shift"], a["f"]
        self.xmin, self.xmax, self.zmin, self.zmax, self.hm = a["xmin"], a["xmax"], a["zmin"], a["zmax"], a["hm"]
        self.W, self.D, self.H = a["W"], a["D"], a["H"]

    def room_metres(self, P):
        loc = (np.asarray(P, float) - self.origin) @ self.R.T
        loc[..., 1] -= self.y_shift
        return loc * self.f

    def block_x(self, x):
        return (x - self.xmin) / (self.xmax - self.xmin) * self.W

    def block_z(self, z):
        return (z - self.zmin) / (self.zmax - self.zmin) * self.D

    def metres_x(self, bx):
        return self.xmin + bx / self.W * (self.xmax - self.xmin)

    def metres_z(self, bz):
        return self.zmin + bz / self.D * (self.zmax - self.zmin)


def rp_category(o):
    """Our furniture type for a RoomPlan object, or None for kinds build doesn't make from RoomPlan."""
    cat = ROOMPLAN_OBJECTS.get(next(iter(o["category"])))
    if cat == "table" and o.get("attributes", {}).get("TableType") == "coffee":
        cat = "coffee_table"
    return cat


def rp_confidence(o):
    return next(iter(o.get("confidence") or {"high": {}}))


def kind(cat):
    return SAME_KIND.get(cat, cat)


class Pose:
    """A RoomPlan box in room metres: centre, yaw (degrees, of its local x axis) and the cluster place_object takes."""

    def __init__(self, frame: PlanFrame, o):
        T, d = _mat(o["transform"]), np.asarray(o["dimensions"], float)
        corners = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * d
        loc = frame.room_metres(corners @ T[:3, :3].T + T[:3, 3])
        self.c = frame.room_metres(T[:3, 3])
        ax = frame.R @ T[:3, 0]
        self.yaw = float(np.degrees(np.arctan2(ax[2], ax[0])))
        self.cluster = {"lo": loc.min(0), "hi": loc.max(0), "c": self.c, "rgb": np.array([128.0, 128.0, 128.0]),
                        "n": 1000}

    def moved_from(self, other: "Pose"):
        turn = abs((self.yaw - other.yaw + 180) % 360 - 180)
        return np.hypot(*(self.c - other.c)[[0, 2]]) > MOVE_M or turn > MOVE_DEG


class Furniture:
    """The built furniture, kept up to date with RoomPlan's live objects."""

    def __init__(self, plan, frame: PlanFrame, scan_room):
        self.fr = frame
        self.objects = [dict(o) for o in plan["objects"]]   # plan objects as they stand now
        self.room = Room(frame.xmin, frame.xmax, frame.zmin, frame.zmax, frame.hm, plan.get("scale", 1.25))
        self.room.W, self.room.D, self.room.H = frame.W, frame.D, frame.H
        # The scan's RoomPlan objects, each linked to the built piece it became (by kind and nearest footprint).
        self.scanned = []          # [(kind, Pose, plan object index or None)]
        taken = set()
        for o in (scan_room or {}).get("objects", []):
            cat = rp_category(o)
            if cat is None:
                continue
            pose = Pose(frame, o)
            bx, bz = frame.block_x(pose.c[0]), frame.block_z(pose.c[2])
            best = None
            for i, p in enumerate(self.objects):
                if i in taken or kind(p["type"]) != kind(cat):
                    continue
                dist = np.hypot(p["x"] + p["w"] / 2 - bx, p["z"] + p["d"] / 2 - bz)
                if dist <= 2 + max(p["w"], p["d"]) and (best is None or dist < best[0]):
                    best = (dist, i)
            if best:
                taken.add(best[1])
            self.scanned.append((kind(cat), pose, best[1] if best else None))
        log(f"[spectate] {sum(s[2] is not None for s in self.scanned)} of {len(self.objects)} built piece(s) "
            f"linked to RoomPlan objects")
        self.live = {}             # live RoomPlan id -> {"scan", "index": plan object index or None, "anchor": Pose}
        self.pending = {}          # live id -> (target key, updates in a row)

    def _occupancy(self, skip=None):
        occ = -np.ones((self.room.W, self.room.D), int)
        for i, o in enumerate(self.objects):
            if i != skip and o["type"] != "rug" and o.get("y", 1) <= 1:
                occ[o["x"]:o["x"] + o["w"], o["z"]:o["z"] + o["d"]] = 1
        return occ

    def _tables(self, skip=None):
        """Tables as place_object's clusters, so chairs face away from them as build does."""
        return [{"c": np.array([self.fr.metres_x(o["x"] + o["w"] / 2), 0.4, self.fr.metres_z(o["z"] + o["d"] / 2)])}
                for i, o in enumerate(self.objects) if i != skip and o["type"] in ("table", "coffee_table", "desk")]

    def _place(self, cat, pose, skip=None):
        def nearest_side(cx, cz):
            d = self.room.wall_dist(cx, cz)
            s = min(d, key=d.get)
            return s, d[s]
        return place_object(cat, pose.cluster, self.room, self._occupancy(skip), self._tables(skip), nearest_side)

    def _link(self, rid, cat, pose):
        """A live object first seen: the scanned object of its kind nearest to it, if close enough and not taken."""
        claimed = {v["scan"] for v in self.live.values() if v["scan"] is not None}
        best = None
        for j, (k, sp, _) in enumerate(self.scanned):
            if j in claimed or k != kind(cat):
                continue
            dist = float(np.hypot(*(pose.c - sp.c)[[0, 2]]))
            if dist <= MATCH_M and (best is None or dist < best[0]):
                best = (dist, j)
        if best:
            j = best[1]
            self.live[rid] = {"scan": j, "index": self.scanned[j][2], "anchor": self.scanned[j][1]}
        else:
            self.live[rid] = {"scan": None, "index": None, "anchor": None}

    def _stable(self, rid, key):
        prev = self.pending.get(rid)
        n = prev[1] + 1 if prev and prev[0] == key else 1
        self.pending[rid] = (key, n)
        return n >= MOVE_UPDATES

    def update(self, room):
        moves, adds = [], []
        for o in room.get("objects", []):
            cat = rp_category(o)
            rid = o.get("identifier")
            if cat is None or rid is None:
                continue
            pose = Pose(self.fr, o)
            if rid not in self.live:
                self._link(rid, cat, pose)
            link = self.live[rid]
            if rp_confidence(o) not in CONFIDENT:
                self.pending.pop(rid, None)   # unsure: whatever was building up starts over
                continue
            idx = link["index"]
            if link["anchor"] is None:
                # Not one of the built pieces: a new piece, once RoomPlan has kept it in the same place a while.
                new = self._place(cat, pose)
                if new is not None and self._stable(rid, ("add", new["x"], new["z"], new["w"], new["d"], new["facing"])):
                    self.objects.append(new)
                    link.update(index=len(self.objects) - 1, anchor=pose)
                    self.pending.pop(rid, None)
                    adds.append(new)
                continue
            if idx is None or not pose.moved_from(link["anchor"]):
                self.pending.pop(rid, None)   # where it was (within RoomPlan's jitter), or never built: nothing to do
                continue
            cur = self.objects[idx]
            new = self._place(cur["type"], pose, skip=idx)
            if new is None:
                continue
            new["type"] = cur["type"]
            for k in KEEP_FIELDS:
                if k in cur:
                    new[k] = cur[k]
            key = (new["x"], new["z"], new["w"], new["d"], new["facing"])
            if key == (cur["x"], cur["z"], cur["w"], cur["d"], cur.get("facing")):
                self.pending.pop(rid, None)   # moved a little, but onto the same blocks
                continue
            if self._stable(rid, ("move",) + key):
                moves.append({"from": cur, "to": new})
                self.objects[idx] = new
                link["anchor"] = pose
                self.pending.pop(rid, None)
        return moves, adds


class People:
    def __init__(self, frame: PlanFrame):
        self.fr = frame
        self.tracks, self.next_id = [], 1

    def update(self, fr):
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
        return self.kept()

    def kept(self):
        """Tracked people in plan block coordinates, standing on the floor (y = 1)."""
        return [{"id": t["id"], "x": round(float(self.fr.block_x(t["xz"][0])), 3), "y": 1.0,
                 "z": round(float(self.fr.block_z(t["xz"][1])), 3)} for t in self.tracks if t["seen"] >= PERSON_SHOW]


def step(furniture, people, header, blobs):
    out = {"type": "update", "moves": [], "adds": [], "people": people.kept()}
    if header.get("type") == "room" and "room" in blobs:
        out["moves"], out["adds"] = furniture.update(json.loads(blobs["room"]))
    elif header.get("type") == "frame" and header.get("tracking", "normal") == "normal":
        fr = decode_frame(header, blobs)
        if fr is not None:
            out["people"] = people.update(fr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path, help="plan.json of the built LiDAR scan")
    ap.add_argument("--mc-jar", help="unused; accepted like the other analyzers")
    opts, unknown = ap.parse_known_args(argv)
    if unknown:
        log(f"[spectate] ignoring options for the photo analyzer: {' '.join(unknown)}")

    plan = json.loads(opts.plan.read_text())
    room_file = opts.plan.parent / "room.json"
    if "arkit" not in plan or not room_file.is_file():
        emit({"type": "status", "level": "error",
              "text": "Spectate needs a LiDAR room scan (RoomPlan) built with this version. Scan and build it first."})
        return 2
    frame = PlanFrame(plan["arkit"])
    furniture = Furniture(plan, frame, json.loads(room_file.read_text()))
    people = People(frame)
    emit({"type": "status", "level": "success",
          "text": "Spectate: furniture you move in the room moves in game once RoomPlan clearly sees it."})
    stdin = sys.stdin.buffer
    while True:
        msg = read_message(stdin)
        if msg is None:
            break
        header, blobs = msg
        if header.get("type") not in ("room", "frame"):
            continue
        t0 = time.time()
        try:
            out = step(furniture, people, header, blobs)
        except Exception as ex:   # one bad message must not end spectating
            import traceback
            traceback.print_exc()
            out = {"type": "update", "moves": [], "adds": [], "people": [], "error": str(ex)}
        out["ms"] = round((time.time() - t0) * 1000)
        emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

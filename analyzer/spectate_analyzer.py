"""everything_cubed spectate worker: Apple RoomPlan's live room model from the iOS app -> furniture moves in a room
already scanned and built.

Usage (started by the mod when the app spectates a scan it has built; not meant to be run by hand):
  python spectate_analyzer.py --plan <scan>/plan.json [--mc-jar client.jar]

Spectate starts from a LiDAR scan that has been built: the build is the starting state, taken as certain. The app
keeps the scan's RoomPlan session running, so RoomPlan's live objects (measured with the LiDAR) are in the scan's
coordinates, and the plan's "arkit" frame (written by lidar_analyzer) maps them onto the plan's blocks.

RoomPlan's live coordinates can still come out shifted from the scan's (once by 1.75 m), so each live room is first
aligned onto the scan: the shift and turn that put its walls and objects onto the scanned ones, used once at least
ALIGN_MIN of them agree (see estimate_alignment); until then nothing is matched.

Only whole pieces of furniture change, and only from RoomPlan's objects; no single blocks are ever placed:
- every built piece of furniture is linked to the RoomPlan object of the scan it came from (room.json);
- a piece moves when RoomPlan places that object clearly elsewhere (less than MOVE_INSIDE of its live floor
  footprint still inside where it was), onto a different footprint, the same one for MOVE_UPDATES room updates in a
  row. Spectate runs RoomPlan afresh, so its first boxes are partial: a box inside the scanned one is the same piece. Its new footprint, height and
  facing come from roomplan.place_object, exactly as build places furniture; its material and colour stay;
- a new piece is added when RoomPlan identifies a new object with medium or high confidence, stable for
  MOVE_UPDATES updates, and only as an extra one: while a scanned piece of its kind has not been seen live, a live
  object of that kind is taken to be that piece (RoomPlan once reported the room's one bed 1.8 m off), not a new one;
- anything RoomPlan is unsure of, or stops reporting, stays as built.
People are found in the camera frames (the `person` mask the phone sends with LiDAR depth) and reported as tracked
positions, which the mod shows as villagers. A person lying on a bed (low and flat over a built bed, LYING_FRAMES
frames in a row) is reported with the bed block they lie on ("sleep": [x, z]); the mod puts the villager to sleep
there.

stdin:  the phone's `room` messages (RoomPlan's live CapturedRoom as JSON in the `room` blob) and `frame` messages,
        forwarded by the mod (framing as in ios/PROTOCOL.md).
stdout: one JSON object per line:
        {"type": "update", "moves": [{"from": object, "to": object}, ...], "adds": [object, ...],
         "people": [{"id", "x", "y", "z", "sleep"?: [x, z]}, ...]}  exactly one per message read, so the mod knows when to send the next;
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

MOVE_INSIDE = 0.5         # an object has moved when less than this share of its live footprint is where it was
MOVE_UPDATES = 5          # room updates in a row that must agree on the new footprint (about a second)
MATCH_M = 1.5             # metres: a live RoomPlan object this close to a scanned one of its kind is that one
CONFIDENT = ("medium", "high")
PERSON_CELL = 0.2         # metres: grid for grouping person points into people
PERSON_MIN_PIXELS = 25
PERSON_MATCH = 0.8        # metres a person may move between frames and keep their id
PERSON_SHOW = 2           # frames a person must be seen before they are reported
PERSON_KEEP = 1.0         # seconds a person stays after they were last seen
LYING_TOP = 1.2          # metres: a person whose highest point is below this, and
LYING_SPAN = 0.9         # ... whose height from lowest to highest point is below this, over a bed, is lying on it
LYING_FRAMES = 3         # frames in a row that must agree before a villager lies down or gets up
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


# ----------------------------------------------------------------------------------- live -> scan alignment
# Spectate runs RoomPlan again, and its live coordinates have come out shifted from the scan's: in one session every
# live object sat 1.75 m off (tracking normal), so the room's one bed and its chair matched nothing. RoomPlan's live
# walls and objects are matched to the scanned ones to find that shift (and any turn), on the floor plane in the
# plan's room metres; every live box is moved by it before matching.

ALIGN_OBJECT_M = 0.5      # a shifted live object agrees when its centre is this close to a scanned one of its kind
ALIGN_WALL_M = 0.3        # a shifted live wall agrees when it lies along a scanned wall, this close to its line
ALIGN_PARALLEL = np.sin(np.radians(5))
ALIGN_MIN = 2             # walls and objects that must agree before an alignment is used
ALIGN_LOG_M, ALIGN_LOG_DEG = 0.1, 2.0   # an alignment that changes more than this is logged again


def _rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _room_walls(frame: PlanFrame, room):
    """Walls as (centre, unit direction, length) on the floor plane in room metres."""
    out = []
    for w in (room or {}).get("walls", []):
        T = _mat(w["transform"])
        u = (frame.R @ T[:3, 0])[[0, 2]]
        if np.linalg.norm(u) < 1e-6:
            continue
        out.append((frame.room_metres(T[:3, 3])[[0, 2]], u / np.linalg.norm(u), float(w["dimensions"][0]) * frame.f))
    return out


def _room_objects(frame: PlanFrame, room):
    """Objects build makes as (kind, centre) on the floor plane in room metres."""
    out = []
    for o in (room or {}).get("objects", []):
        cat = rp_category(o)
        if cat is not None:
            out.append((kind(cat), frame.room_metres(_mat(o["transform"])[:3, 3])[[0, 2]]))
    return out


def _corners(walls):
    """Where two roughly perpendicular walls' lines cross: (point, index i, index j)."""
    out = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            (ci, ui, _), (cj, uj, _) = walls[i], walls[j]
            if abs(ui @ uj) > 0.3:
                continue
            a = np.linalg.solve(np.stack([ui, -uj], 1), cj - ci)
            out.append((ci + a[0] * ui, i, j))
    return out


class Alignment:
    """live room metres (floor plane) -> scan room metres: p -> rot(turn) p + shift."""

    def __init__(self, turn=0.0, shift=(0.0, 0.0), walls=0, objects=0):
        self.turn, self.shift, self.walls, self.objects = float(turn), np.asarray(shift, float), walls, objects

    @property
    def support(self):
        return self.walls + self.objects

    def differs(self, other):
        return (other is None or np.linalg.norm(self.shift - other.shift) > ALIGN_LOG_M
                or abs(np.degrees(self.turn - other.turn)) > ALIGN_LOG_DEG)

    def world(self, frame: PlanFrame):
        """The same move as a 4x4 on ARKit world coordinates (RoomPlan transforms are multiplied by it)."""
        r2 = _rot2(self.turn)
        rot = np.eye(3)
        rot[0, 0], rot[0, 2], rot[2, 0], rot[2, 2] = r2[0, 0], r2[0, 1], r2[1, 0], r2[1, 1]
        M = np.eye(4)
        M[:3, :3] = frame.R.T @ rot @ frame.R
        t_room = np.array([self.shift[0], 0.0, self.shift[1]]) / frame.f
        M[:3, 3] = frame.origin - M[:3, :3] @ frame.origin + frame.R.T @ t_room
        return M

    def describe(self):
        return (f"shift ({self.shift[0]:+.2f}, {self.shift[1]:+.2f}) m, turn {np.degrees(self.turn):+.1f} deg, "
                f"agreed by {self.walls} wall(s) and {self.objects} object(s)")


def _agreement(turn, shift, live_w, live_o, scan_w, scan_o):
    """How many live walls and objects land on scanned ones under (turn, shift), with what refines the shift:
    object pairs (live, scanned) and wall pairs (live centre, scanned centre, scanned normal)."""
    r = _rot2(turn)
    walls, objs = [], []
    for c, u, length in live_w:
        c2, u2 = r @ c + shift, r @ u
        for cs, us, ls in scan_w:
            n = np.array([-us[1], us[0]])
            if abs(u2[0] * us[1] - u2[1] * us[0]) <= ALIGN_PARALLEL and abs(n @ (c2 - cs)) <= ALIGN_WALL_M \
                    and abs(us @ (c2 - cs)) <= (length + ls) / 2:
                walls.append((c, cs, n))
                break
    used = set()
    for k, c in live_o:
        c2 = r @ c + shift
        best = None
        for j, (ks, cs) in enumerate(scan_o):
            if ks == k and j not in used and np.linalg.norm(c2 - cs) <= ALIGN_OBJECT_M:
                if best is None or np.linalg.norm(c2 - cs) < best[0]:
                    best = (np.linalg.norm(c2 - cs), j)
        if best:
            used.add(best[1])
            objs.append((c, scan_o[best[1]][1]))
    return walls, objs


def estimate_alignment(live_w, live_o, scan_w, scan_o):
    """The turn and shift that put the most live walls and objects onto scanned ones (at least ALIGN_MIN), its shift
    refined by least squares over those that agree; ties go to the smallest move. None when nothing reaches it."""
    turns = [0.0]
    for _, u, _ in live_w:
        for _, us, _ in scan_w:
            d = np.arctan2(us[1], us[0]) - np.arctan2(u[1], u[0])
            for t in (d, d + np.pi):     # a wall's direction is only known up to a half turn
                t = (t + np.pi) % (2 * np.pi) - np.pi
                if all(abs((t - q + np.pi) % (2 * np.pi) - np.pi) > np.radians(1) for q in turns):
                    turns.append(t)
    live_c, scan_c = _corners(live_w), _corners(scan_w)
    best = None
    for turn in turns:
        r = _rot2(turn)
        shifts = [np.zeros(2)]
        shifts += [cs - r @ c for k, c in live_o for ks, cs in scan_o if ks == k]
        shifts += [cs - r @ c for c, _, _ in live_c for cs, _, _ in scan_c]
        for shift in shifts:
            walls, objs = _agreement(turn, shift, live_w, live_o, scan_w, scan_o)
            if len(walls) + len(objs) < ALIGN_MIN:
                continue
            # Least squares: objects pin both coordinates, walls only across themselves.
            A, b = np.zeros((2, 2)), np.zeros(2)
            for c, cs in objs:
                A += np.eye(2); b += cs - r @ c
            for c, cs, n in walls:
                A += np.outer(n, n); b += n * (n @ (cs - r @ c))
            if np.linalg.matrix_rank(A) == 2:
                refined = np.linalg.solve(A, b)
                w2, o2 = _agreement(turn, refined, live_w, live_o, scan_w, scan_o)
                if len(w2) + len(o2) >= len(walls) + len(objs):
                    shift, walls, objs = refined, w2, o2
            key = (-(len(walls) + len(objs)), float(np.linalg.norm(shift)) + abs(turn))
            if best is None or key < best[0]:
                best = (key, Alignment(turn, shift, len(walls), len(objs)))
    return best[1] if best else None


class Pose:
    """A RoomPlan box in room metres: centre, floor footprint and the cluster place_object takes."""

    def __init__(self, frame: PlanFrame, o):
        T, d = _mat(o["transform"]), np.asarray(o["dimensions"], float)
        corners = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * d
        loc = frame.room_metres(corners @ T[:3, :3].T + T[:3, 3])
        self.c = frame.room_metres(T[:3, 3])
        self.cluster = {"lo": loc.min(0), "hi": loc.max(0), "c": self.c, "rgb": np.array([128.0, 128.0, 128.0]),
                        "n": 1000}
        # Floor footprint: the box's two horizontal axes (unit, room frame) and its half sizes along them.
        self.axes = np.stack([frame.R @ T[:3, 0], frame.R @ T[:3, 2]])[:, [0, 2]]
        self.half = d[[0, 2]] / 2 * frame.f

    def inside_share(self, other: "Pose", n=9):
        """Share of this box's floor footprint that lies inside `other`'s (sampled on an n x n grid)."""
        g = (np.arange(n) + 0.5) / n * 2 - 1
        local = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) * self.half
        pts = self.c[[0, 2]] + local @ self.axes
        rel = (pts - other.c[[0, 2]]) @ other.axes.T
        return float(np.all(np.abs(rel) <= other.half + 1e-6, axis=1).mean())

    def describe(self, frame: PlanFrame):
        """For the log: centre in blocks and footprint size in metres."""
        return (f"centre block ({frame.block_x(self.c[0]):.1f}, {frame.block_z(self.c[2]):.1f}), "
                f"{2 * self.half[0]:.2f} x {2 * self.half[1]:.2f} m")

    def moved_from(self, other: "Pose"):
        """Mostly outside where it was. A partial box RoomPlan has only begun to see, inside the scanned one, has not
        moved, nor has the same box turned in place or described with its axes the other way round."""
        return self.inside_share(other) < MOVE_INSIDE


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
        self.scan_walls, self.scan_objects = _room_walls(frame, scan_room), _room_objects(frame, scan_room)
        self.align = None          # Alignment of RoomPlan's live coordinates onto the scan's, once agreed
        self.align_logged = None   # the alignment last logged
        self.waiting_logged = False
        self.held = set()          # live ids not added because a scanned piece of their kind is unseen (logged once)

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

    def _link(self, rid, cat, pose, o=None):
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
        # First sighting, matched or not: how far it is from the nearest scanned piece of its kind. The same offset
        # on every object means RoomPlan's coordinates moved, not the furniture.
        same = [(float(np.hypot(*(pose.c - sp.c)[[0, 2]])), j, sp) for j, (k, sp, _) in enumerate(self.scanned)
                if k == kind(cat)]
        if same:
            dist, j, sp = min(same, key=lambda t: t[0])
            off = pose.c - sp.c
            verdict = ("matched" if best else f"already matched to another live {cat}" if j in claimed
                       else f"not matched (more than {MATCH_M} m)")
            near = (f"nearest scanned {kind(cat)} {sp.describe(self.fr)}: offset ({off[0]:+.2f}, {off[2]:+.2f}) m, "
                    f"{dist:.2f} m, {pose.inside_share(sp):.0%} inside it; {verdict}")
        else:
            near = f"no scanned {kind(cat)}"
        log(f"[spectate] first sight: {cat} {rid[:8]} {pose.describe(self.fr)}, {rp_confidence(o) if o else '?'} "
            f"confidence; {near}")

    def _realign(self, room):
        """Brings self.align up to date from this live room. False while no alignment has been agreed yet."""
        live_w, live_o = _room_walls(self.fr, room), _room_objects(self.fr, room)
        best = estimate_alignment(live_w, live_o, self.scan_walls, self.scan_objects)
        if self.align is not None:
            w, o = _agreement(self.align.turn, self.align.shift, live_w, live_o, self.scan_walls, self.scan_objects)
            if best is None or best.support <= len(w) + len(o):
                return True        # the current alignment still explains this update as well as anything
        if best is None:
            if self.align is None and not self.waiting_logged:
                self.waiting_logged = True
                log(f"[spectate] waiting to align RoomPlan's live coordinates with the scan: fewer than {ALIGN_MIN} "
                    f"of {len(live_w)} live wall(s) and {len(live_o)} object(s) agree on a shift yet")
            return self.align is not None
        if best.differs(self.align_logged):
            log(f"[spectate] RoomPlan's live coordinates aligned to the scan: {best.describe()}")
            self.align_logged = best
        self.align = best
        return True

    def _unseen_scanned(self, k):
        """How many scanned pieces of kind k no live object has been matched to."""
        claimed = {v["scan"] for v in self.live.values() if v["scan"] is not None}
        return sum(1 for j, (kk, _, _) in enumerate(self.scanned) if kk == k and j not in claimed)

    def _stable(self, rid, key):
        prev = self.pending.get(rid)
        n = prev[1] + 1 if prev and prev[0] == key else 1
        self.pending[rid] = (key, n)
        return n >= MOVE_UPDATES

    def update(self, room):
        moves, adds = [], []
        if not self._realign(room):
            return moves, adds
        M = self.align.world(self.fr)
        for o in room.get("objects", []):
            o = dict(o, transform=(M @ _mat(o["transform"])).T.ravel().tolist())   # in the scan's coordinates
            cat = rp_category(o)
            rid = o.get("identifier")
            if cat is None or rid is None:
                continue
            pose = Pose(self.fr, o)
            if rid not in self.live:
                self._link(rid, cat, pose, o)
            link = self.live[rid]
            if rp_confidence(o) not in CONFIDENT:
                self.pending.pop(rid, None)   # unsure: whatever was building up starts over
                continue
            idx = link["index"]
            if link["anchor"] is None:
                # Not one of the built pieces: a new piece, once RoomPlan has kept it in the same place a while, and
                # only when every scanned piece of its kind is accounted for live.
                unseen = self._unseen_scanned(kind(cat))
                if unseen:
                    if rid not in self.held:
                        self.held.add(rid)
                        log(f"[spectate] live {cat} {rid[:8]} not added: {unseen} scanned {kind(cat)}(s) not seen live "
                            f"yet, so it is taken to be one of them")
                    self.pending.pop(rid, None)
                    continue
                new = self._place(cat, pose)
                if new is not None and self._stable(rid, ("add", new["x"], new["z"], new["w"], new["d"], new["facing"])):
                    log(f"[spectate] add {cat} at block ({new['x']}, {new['z']}) {new['w']}x{new['d']} facing "
                        f"{new['facing']}: live {rid[:8]} {pose.describe(self.fr)}, {rp_confidence(o)} confidence")
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
                log(f"[spectate] move {cur['type']} #{idx} from block ({cur['x']}, {cur['z']}) {cur['w']}x{cur['d']} facing "
                    f"{cur.get('facing')} to ({new['x']}, {new['z']}) {new['w']}x{new['d']} facing {new['facing']}: "
                    f"live {rid[:8]} {pose.describe(self.fr)}, {rp_confidence(o)} confidence, "
                    f"{pose.inside_share(link['anchor']):.0%} inside where it was ({link['anchor'].describe(self.fr)})")
                moves.append({"from": cur, "to": new})
                self.objects[idx] = new
                link["anchor"] = pose
                self.pending.pop(rid, None)
        return moves, adds


class People:
    def __init__(self, frame: PlanFrame, furniture=None):
        self.fr = frame
        self.furniture = furniture      # its objects are the beds as they stand now (they can move)
        self.tracks, self.next_id = [], 1

    def bed_under(self, xz):
        """The bed block (x, z) under a point (room metres), if a built bed is there (half a block of slack)."""
        bx, bz = self.fr.block_x(xz[0]), self.fr.block_z(xz[1])
        for o in (self.furniture.objects if self.furniture else []):
            if o["type"] != "bed":
                continue
            if o["x"] - 0.5 <= bx < o["x"] + o["w"] + 0.5 and o["z"] - 0.5 <= bz < o["z"] + o["d"] + 0.5:
                return (int(np.clip(np.floor(bx), o["x"], o["x"] + o["w"] - 1)),
                        int(np.clip(np.floor(bz), o["z"], o["z"] + o["d"] - 1)))
        return None

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
                if sel.sum() < PERSON_MIN_PIXELS:
                    continue
                y = loc[sel, 1]
                top, span = np.percentile(y, 95), np.percentile(y, 95) - np.percentile(y, 5)
                xz = np.median(loc[sel][:, [0, 2]], axis=0)
                bed = self.bed_under(xz) if top < LYING_TOP and span < LYING_SPAN else None
                if span < 0.4 and bed is None:
                    continue   # too flat for someone standing or sitting, and not on a bed: not a person
                found.append((xz, bed))
        now = fr["time"]
        free = list(range(len(found)))
        for t in sorted(self.tracks, key=lambda t: -t["seen"]):
            if not free:
                break
            d = [np.linalg.norm(found[f][0] - t["xz"]) for f in free]
            best = int(np.argmin(d))
            if d[best] <= PERSON_MATCH:
                t["xz"], bed = found[free.pop(best)]
                t["seen"] += 1
                t["last"] = now
                # Lying down or getting up only after LYING_FRAMES frames in a row agree.
                lying = bed is not None
                t["streak"] = t["streak"] + 1 if lying != bool(t["bed"]) else 0
                if lying and t["bed"]:
                    t["bed"] = bed
                elif t["streak"] >= LYING_FRAMES:
                    t["bed"], t["streak"] = bed, 0
        for f in free:
            # A first sighting on a bed counts towards lying down.
            self.tracks.append({"id": self.next_id, "xz": found[f][0], "seen": 1, "last": now, "bed": None,
                                "streak": int(found[f][1] is not None)})
            self.next_id += 1
        self.tracks = [t for t in self.tracks if now - t["last"] <= PERSON_KEEP]
        return self.kept()

    def kept(self):
        """Tracked people in plan block coordinates, standing on the floor (y = 1)."""
        out = []
        for t in self.tracks:
            if t["seen"] < PERSON_SHOW:
                continue
            p = {"id": t["id"], "x": round(float(self.fr.block_x(t["xz"][0])), 3), "y": 1.0,
                 "z": round(float(self.fr.block_z(t["xz"][1])), 3)}
            if t["bed"]:
                p["sleep"] = list(t["bed"])
            out.append(p)
        return out


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
    people = People(frame, furniture)
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

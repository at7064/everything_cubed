"""Sanity test for spectate_analyzer.py: RoomPlan's live room model (and camera frames) through its stdin / stdout,
as the mod runs it, over a room already scanned and built.

Room: 6.0 m (x) by 5.0 m (z), 2.6 m high, in ARKit's world frame turned 20 degrees about gravity, built at 1.25
blocks/m (8 x 6 x 3 blocks). The scan's RoomPlan model has a chair and a table, built as a chair and a table.
Live RoomPlan updates (with their own object ids, as a new capture run gives) then:
1. show the room as scanned, and with a few cm of jitter: nothing changes;
2. show the chair 1.2 m away for 3 updates, then back: nothing changes;
3. show the chair 1.2 m away for good: it moves once, as a whole chair (same material), and nothing else does;
4. show a new sofa at low confidence: nothing; at medium confidence: one sofa is added;
5. leave the table out: it stays.
A camera frame with a person (masked by the phone) reports a person; no blocks are ever sent. Someone lying on the
bed is reported sleeping in it after 3 frames (not 2), and awake again after getting up for 3 frames.
Run: python test_spectate.py
"""
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from perception import ARKIT_TO_CV

W, D, H, S = 6.0, 5.0, 2.6, 1.25
TURN = np.radians(20)
IW, IH, DW, DH, FX = 1920, 1440, 256, 192, 1450.0


def rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


WORLD = np.eye(4); WORLD[:3, :3] = rot_y(TURN)
ARKIT = {"origin": [0.0, 0.0, 0.0], "R": [(rot_y(TURN) @ [1.0, 0, 0]).tolist(), [0.0, 1.0, 0.0],
                                          (rot_y(TURN) @ [0, 0, 1.0]).tolist()],
         "y_shift": 0.0, "f": 1.0, "xmin": 0.0, "xmax": W, "zmin": 0.0, "zmax": D, "hm": H,
         "W": round(W * S), "D": round(D * S), "H": round(H * S)}


def rp(ident, cat, centre, dims, conf="high", yaw=0.0, **attrs):
    """A RoomPlan object at a room-local centre (x, z), standing on the floor, as CapturedRoom encodes it."""
    T = np.eye(4)
    T[:3, :3] = rot_y(TURN) @ rot_y(np.radians(yaw))
    T[:3, 3] = rot_y(TURN) @ [centre[0], dims[1] / 2, centre[1]]
    return {"identifier": ident, "category": {cat: {}}, "transform": T.T.ravel().tolist(), "dimensions": list(dims),
            "confidence": {conf: {}}, "attributes": attrs}


CHAIR, TABLE, SOFA, BED = (0.5, 0.86, 0.55), (1.4, 0.75, 0.8), (1.8, 0.85, 0.9), (1.1, 0.6, 2.0)


def message(type_, header=None, blobs=()):
    h = dict(header or {}, type=type_, blobs=[{"name": n, "format": f, "length": len(b)} for n, f, b in blobs])
    hb = json.dumps(h).encode()
    payload = b"".join(b for _, _, b in blobs)
    return struct.pack(">I", len(hb)) + hb + struct.pack(">I", len(payload)) + payload


def room_message(objects):
    return message("room", blobs=[("room", "roomplan.json", json.dumps({"objects": objects}).encode())])


def look_at(eye, target):
    """ARKit cam2world (room-local) at `eye` looking at `target` (camera looks down -Z, +Y up)."""
    fwd = np.asarray(target, float) - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1.0, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -fwd, eye
    return c2w


def body_frame(i, lo, hi, eye, target, live=np.eye(4)):
    """A frame of a person (a box lo..hi in room metres, masked) over the floor, from `eye` looking at `target`.
    live: how RoomPlan's live coordinates sit against the scan's (the camera pose is reported in them)."""
    K = np.array([[FX, 0, IW / 2], [0, FX, IH / 2], [0, 0, 1]])
    Kd = K.copy(); Kd[:2] *= DW / IW
    cv = look_at(np.asarray(eye, float), target) @ ARKIT_TO_CV
    u, v = np.meshgrid(np.arange(DW) + 0.5, np.arange(DH) + 0.5)
    dirs = np.stack([(u - Kd[0, 2]) / Kd[0, 0], (v - Kd[1, 2]) / Kd[1, 1], np.ones_like(u)], -1) @ cv[:3, :3].T
    o = cv[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (np.asarray(lo) - o) / dirs, (np.asarray(hi) - o) / dirs
        tn, tf = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
        floor = np.where(dirs[..., 1] < 0, -o[1] / dirs[..., 1], 9.0)
    body = np.where((tn <= tf) & (tn > 0), tn, np.inf)
    mask = body < floor
    depth = np.minimum(np.minimum(body, floor), 9.0)
    img = io.BytesIO(); Image.fromarray(np.full((IH // 8, IW // 8, 3), 180, np.uint8)).resize((IW, IH)).save(img, "JPEG")
    header = {"seq": i, "time": i * 0.2, "tracking": "normal", "imageSize": [IW, IH], "depthSize": [DW, DH],
              "K": K.ravel().tolist(), "cam2world": (live @ WORLD @ look_at(np.asarray(eye, float), target)).ravel().tolist()}
    return message("frame", header, [("image", "jpeg", img.getvalue()),
                                     ("depth", "f16", depth.astype("<f2").tobytes()),
                                     ("confidence", "u8", np.full((DH, DW), 2, np.uint8).tobytes()),
                                     ("person", "u8", mask.astype(np.uint8).tobytes())])


def person_frame(i):
    """A camera frame looking at a person (masked) standing at room (3.4, 1.15)."""
    K = np.array([[FX, 0, IW / 2], [0, FX, IH / 2], [0, 0, 1]])
    Kd = K.copy(); Kd[:2] *= DW / IW
    c2w = np.eye(4); c2w[:3, 3] = [3.4, 1.4, 3.5]              # looking down -z (ARKit), at the person
    u, v = np.meshgrid(np.arange(DW) + 0.5, np.arange(DH) + 0.5)
    cv = c2w @ ARKIT_TO_CV
    dirs = np.stack([(u - Kd[0, 2]) / Kd[0, 0], (v - Kd[1, 2]) / Kd[1, 1], np.ones_like(u)], -1) @ cv[:3, :3].T
    depth = np.full((DH, DW), 2.3)                              # the person's front at z = 1.2
    hit = cv[:3, 3] + depth[..., None] * dirs
    mask = (np.abs(hit[..., 0] - 3.4) < 0.25) & (hit[..., 1] > 0) & (hit[..., 1] < 1.75)
    depth = np.where(mask, depth, 4.0)
    img = io.BytesIO(); Image.fromarray(np.full((IH // 8, IW // 8, 3), 180, np.uint8)).resize((IW, IH)).save(img, "JPEG")
    header = {"seq": i, "time": i * 0.2, "tracking": "normal", "imageSize": [IW, IH], "depthSize": [DW, DH],
              "K": K.ravel().tolist(), "cam2world": (WORLD @ c2w).ravel().tolist()}
    return message("frame", header, [("image", "jpeg", img.getvalue()),
                                     ("depth", "f16", depth.astype("<f2").tobytes()),
                                     ("confidence", "u8", np.full((DH, DW), 2, np.uint8).tobytes()),
                                     ("person", "u8", mask.astype(np.uint8).tobytes())])


# The scan and its build: the worker's own placement rules give the built objects (as build placed them).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spectate_analyzer as sa  # noqa: E402

scan = {"objects": [rp("scan-chair", "chair", (1.5, 1.5), CHAIR), rp("scan-table", "table", (4.0, 3.0), TABLE),
                    rp("scan-bed", "bed", (5.4, 3.9), BED, yaw=90)]}
frame = sa.PlanFrame(ARKIT)
builder = sa.Furniture({"objects": [], "scale": S}, frame, {"objects": []})
built = []
for o in scan["objects"]:
    obj = builder._place(sa.rp_category(o), sa.Pose(frame, o))
    obj["material"] = "spruce"
    builder.objects.append(obj)
    built.append(obj)
tmp = Path(tempfile.mkdtemp())
(tmp / "plan.json").write_text(json.dumps({"arkit": ARKIT, "scale": S, "objects": built}))
(tmp / "room.json").write_text(json.dumps(scan))

p = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / "spectate_analyzer.py"),
                      "--plan", str(tmp / "plan.json")], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
moves, adds, people, ms = [], [], [], []


def send(data):
    global people
    p.stdin.write(data); p.stdin.flush()
    while True:
        out = json.loads(p.stdout.readline())
        if out["type"] == "update":
            break
    assert "error" not in out, out
    assert set(out) <= {"type", "moves", "adds", "people", "ms"}, out   # furniture only, never single blocks
    moves.extend(out["moves"]); adds.extend(out["adds"]); people = out["people"]; ms.append(out["ms"])


def live(chair=(1.5, 1.5), table=True, sofa=None, jitter=0.0, bed=None, bed_id="live-bed"):
    rng = np.random.default_rng(len(ms))
    j = lambda c: (c[0] + rng.uniform(-jitter, jitter), c[1] + rng.uniform(-jitter, jitter))
    objs = [rp("live-chair", "chair", j(chair), CHAIR)]
    if table:
        objs.append(rp("live-table", "table", j((4.0, 3.0)), TABLE))
    if sofa:
        objs.append(rp("live-sofa", "sofa", (1.5, 4.0), SOFA, conf=sofa))
    if bed:
        objs.append(rp(bed_id, "bed", bed[0], bed[1], yaw=90))
    send(room_message(objs))


for _ in range(5):
    live()
for _ in range(10):
    live(jitter=0.05)
assert not moves and not adds, (moves, adds)
print("room as scanned, and with 5 cm jitter: nothing changes")

# RoomPlan reports the room's one bed, full size, 1.9 m from where it was scanned (a real session put it at the far
# wall). The scanned bed hasn't been seen live, so this is taken to be it: no second bed is added.
for _ in range(8):
    live(bed=((3.5, 3.9), BED), bed_id="ghost-bed")
assert not moves and not adds, (moves, adds)
print("one bed reported far from where it was scanned: no second bed")

# Spectate runs RoomPlan afresh: its first bed box covers only the half it has seen, so its centre is 45 cm off the
# scanned one. Inside the scanned box, it is the same bed: nothing moves.
for _ in range(8):
    live(bed=((5.85, 3.9), (1.1, 0.6, 1.0)))
assert not moves and not adds, (moves, adds)
print("partial bed box inside the scanned one: nothing changes")

for _ in range(3):
    live(chair=(1.5, 2.7))
for _ in range(5):
    live()
assert not moves, moves
print("chair away for 3 updates, then back: nothing changes")

for _ in range(8):
    live(chair=(1.5, 2.7))
assert len(moves) == 1, moves
mv = moves[0]
assert mv["from"] == built[0] and mv["to"]["type"] == "chair" and mv["to"]["material"] == "spruce", mv
assert mv["to"]["z"] > mv["from"]["z"], mv
print(f"chair moved for good: one move, ({mv['from']['x']}, {mv['from']['z']}) -> ({mv['to']['x']}, {mv['to']['z']})")

for _ in range(8):
    live(chair=(1.5, 2.7), sofa="low")
assert not adds, adds
for _ in range(8):
    live(chair=(1.5, 2.7), sofa="medium")
assert len(adds) == 1 and adds[0]["type"] == "sofa", adds
print(f"new sofa: nothing at low confidence, added once at medium ({adds[0]['x']}, {adds[0]['z']})")

for _ in range(10):
    live(chair=(1.5, 2.7), table=False, sofa="medium")
assert len(moves) == 1 and len(adds) == 1, (moves, adds)
print("table left out of RoomPlan's updates: it stays")

for i in range(4):
    send(person_frame(i))
assert people and abs(people[0]["x"] - 3.4 / W * 8) < 1 and abs(people[0]["z"] - 1.15 / D * 6) < 1, people
print(f"person at block ({people[0]['x']:.1f}, {people[0]['z']:.1f})")

# Someone lies down on the bed: low and flat over it. Asleep after 3 frames, awake 3 frames after getting up.
bed = next(o for o in built if o["type"] == "bed")
bx, bz = frame.metres_x(bed["x"] + bed["w"] / 2), frame.metres_z(bed["z"] + bed["d"] / 2)
lying = ((bx - 0.25, 0.55, bz - 0.8), (bx + 0.25, 0.85, bz + 0.8))
eye, look = (bx - 2.0, 1.6, bz), (bx, 0.6, bz)
for i in range(2):
    send(body_frame(10 + i, *lying, eye, look))
assert people and not any("sleep" in q for q in people), people
send(body_frame(12, *lying, eye, look))
sleeper = [q for q in people if "sleep" in q]
assert sleeper, people
sx, sz = sleeper[0]["sleep"]
assert bed["x"] <= sx < bed["x"] + bed["w"] and bed["z"] <= sz < bed["z"] + bed["d"], (sleeper, bed)
print(f"lying on the bed: asleep in bed block ({sx}, {sz}) after 3 frames")
standing = ((bx - 0.9, 0.0, bz - 0.2), (bx - 0.5, 1.75, bz + 0.2))   # got up, standing beside the bed
for i in range(2):
    send(body_frame(13 + i, *standing, eye, look))
assert any("sleep" in q for q in people), "woke up after only 2 frames"
send(body_frame(15, *standing, eye, look))
assert people and not any("sleep" in q for q in people), people
print("got up: awake after 3 frames")
p.stdin.close()
p.wait(timeout=10)
print(f"median {np.median(ms):.0f} ms per message")
# RoomPlan's live coordinates off from the scan's (real sessions: every object 1.75 m off, and 0.65 m high), here
# also turned 8 deg.
# The worker aligns the live walls and objects onto the scanned ones first, so nothing moves or is added; a chair that
# really moves still moves, to where it really went.
def wall(x0, z0, x1, z1, h=2.5):
    d = rot_y(TURN) @ [x1 - x0, 0, z1 - z0]
    T = np.eye(4)
    T[:3, 0] = d / np.linalg.norm(d); T[:3, 1] = [0, 1, 0]; T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
    T[:3, 3] = rot_y(TURN) @ [(x0 + x1) / 2, h / 2, (z0 + z1) / 2]
    return {"transform": T.T.ravel().tolist(), "dimensions": [float(np.hypot(x1 - x0, z1 - z0)), h, 0]}


walls = [wall(0, 0, W, 0), wall(W, 0, W, D), wall(0, D, 0, 0)]          # the south wall was never scanned
scan_w = dict(scan, walls=walls)
G = np.eye(4); G[:3, :3] = rot_y(np.radians(8)); G[:3, 3] = rot_y(TURN) @ [-1.75, 0.65, -0.17]   # live = G @ scan


def shifted(o):
    return dict(o, transform=(G @ np.array(o["transform"], float).reshape(4, 4).T).T.ravel().tolist())


fw = sa.Furniture({"objects": [dict(o) for o in built], "scale": S}, frame, scan_w)
live_objs = [rp("al-chair", "chair", (1.5, 1.5), CHAIR), rp("al-table", "table", (4.0, 3.0), TABLE),
             rp("al-bed", "bed", (5.4, 3.9), BED, yaw=90)]
got = []
for _ in range(8):
    m, a = fw.update({"walls": [shifted(w) for w in walls], "objects": [shifted(o) for o in live_objs]})
    got += m + a
assert not got, got
back = fw.align.world(frame) @ G
assert np.allclose(back, np.eye(4), atol=0.02), np.round(back, 3)
assert {v["index"] for v in fw.live.values()} == {0, 1, 2}, fw.live
live_objs[0] = rp("al-chair", "chair", (1.5, 2.7), CHAIR)                 # the chair really moves 1.2 m
for _ in range(8):
    m, a = fw.update({"walls": [shifted(w) for w in walls], "objects": [shifted(o) for o in live_objs]})
    got += m + a
true_move = builder._place("chair", sa.Pose(frame, live_objs[0]), skip=0)
assert len(got) == 1 and (got[0]["to"]["x"], got[0]["to"]["z"]) == (true_move["x"], true_move["z"]), (got, true_move)
print(f"live coordinates 1.75 m off and turned 8 deg: aligned, nothing moved; real chair move -> "
      f"({got[0]['to']['x']}, {got[0]['to']['z']})")

# People come in the same live coordinates (the frame's camera pose): someone lying on the bed, seen through the
# shifted and turned frame, sleeps in the built bed. Before the furniture is aligned nobody is reported.
def frame_of(msg):
    header, blobs = sa.read_message(io.BytesIO(msg))
    return sa.decode_frame(header, blobs)


wbed = next(o for o in fw.objects if o["type"] == "bed")
wx, wz = frame.metres_x(wbed["x"] + wbed["w"] / 2), frame.metres_z(wbed["z"] + wbed["d"] / 2)
w_lying = ((wx - 0.25, 0.55, wz - 0.8), (wx + 0.25, 0.85, wz + 0.8))
w_eye, w_look = (wx - 2.0, 1.6, wz), (wx, 0.6, wz)
unaligned = sa.People(frame, sa.Furniture({"objects": [dict(o) for o in built], "scale": S}, frame, scan_w))
assert unaligned.update(frame_of(body_frame(30, *w_lying, w_eye, w_look, live=G))) == []
crowd = sa.People(frame, fw)
for i in range(3):
    seen = crowd.update(frame_of(body_frame(31 + i, *w_lying, w_eye, w_look, live=G)))
asleep = [q for q in seen if "sleep" in q]
assert asleep and wbed["x"] <= asleep[0]["sleep"][0] < wbed["x"] + wbed["w"] \
    and wbed["z"] <= asleep[0]["sleep"][1] < wbed["z"] + wbed["d"], (seen, wbed)
print(f"person on the bed through the shifted frame: asleep in bed block {asleep[0]['sleep']}")

# A speck of stray mask pixels (36 px, 1.8 m up at the far wall) beside a person is not a second person.
specked = frame_of(body_frame(35, *w_lying, w_eye, w_look, live=G))
specked["person"][:6, :6] = True
specked["valid"][:6, :6] = True
specked["P"][:6, :6] = (G @ WORLD @ [W - 0.1, 1.8, D - 0.1, 1])[:3]
crowd2 = sa.People(frame, fw)
for i in range(3):
    specked["time"] = 36 + i
    seen2 = crowd2.update(specked)
assert len(seen2) == 1, seen2
print("a speck beside the person: still one person")

# At 2.5 blocks/m the built bed (2 x 1 blocks, 0.8 x 0.4 m) is far smaller than the real 2 x 1.1 m one, so someone
# lying at the real bed's foot end is off the built blocks; they still lie on the bed (and sleep in it). A real
# session's villager vanished instead: flat and "not on a bed", the person was dropped.
S25 = 2.5
ARKIT25 = dict(ARKIT, W=round(W * S25), D=round(D * S25), H=round(H * S25))
frame25 = sa.PlanFrame(ARKIT25)
build25 = sa.Furniture({"objects": [], "scale": S25}, frame25, {"objects": []})
bed25 = build25._place("bed", sa.Pose(frame25, scan["objects"][2]))
f25 = sa.Furniture({"objects": [bed25], "scale": S25}, frame25, {"objects": [scan["objects"][2]]})
f25.align = sa.Alignment()                  # live coordinates as the scan's (people wait for an alignment)
p25 = sa.People(frame25, f25)
foot = (5.4 - 0.75, 3.9)                    # the bed runs along x (yaw 90): 0.75 m from its centre
assert not (bed25["x"] - 0.5 <= frame25.block_x(foot[0]) < bed25["x"] + bed25["w"] + 0.5), (bed25, "test needs a spot off the built bed")
assert p25.bed_under(np.array(foot)) is not None and p25.bed_under(np.array((5.4, 3.9 + 1.6))) is None
lying25 = ((foot[0] - 0.8, 0.55, foot[1] - 0.25), (foot[0] + 0.3, 0.85, foot[1] + 0.25))
for i in range(3):
    seen25 = p25.update(frame_of(body_frame(40 + i, *lying25, (foot[0], 1.6, foot[1] - 2.0), (foot[0], 0.6, foot[1]))))
assert seen25 and "sleep" in seen25[0], seen25
print(f"lying on the real bed off the built blocks: asleep in bed block {seen25[0]['sleep']}")
print("OK")

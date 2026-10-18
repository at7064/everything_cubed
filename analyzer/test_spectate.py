"""Sanity test for spectate_analyzer.py: a synthetic live stream through its stdin / stdout, as the mod runs it, over
a room already scanned and built.

Room: 6.0 m (x) by 5.0 m (z), 2.6 m high, in ARKit's world frame turned 20 degrees about gravity, built at 1.25
blocks/m (8 x 6 x 3 blocks). The build has a 1 m crate and a low 0.4 m box (half its block) in it. The phone stands
in the middle and keeps turning (30 degrees a frame, slower to see a new box), 5 frames a second, in the
scan's coordinates (it has relocalised).
1. Seeing the room as built changes nothing; the low box stays though most of its block is empty.
2. A box seen in one frame only, and the crate gone but seen only at low confidence, change nothing.
3. The crate carried away: its blocks go once the empty floor is clearly seen; nothing else goes.
4. A new box put down appears; the floor, walls and ceiling never become blocks.
5. A person (masked by the phone) walks in: reported as a person, never built.
6. Frames while the phone is still relocalising are no evidence.
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

W, D, H = 6.0, 5.0, 2.6
S = 1.25
IW, IH, DW, DH = 1920, 1440, 256, 192
FX = 1450.0
TURN = np.radians(20)
CRATE = (np.array([3.8, 0.0, 3.4]), np.array([4.8, 1.0, 4.2]))
LOW = (np.array([1.55, 0.0, 1.75]), np.array([2.2, 0.4, 2.45]))      # inside block (2, 1, 2), 0.4 of 0.87 m high
GHOST = (np.array([1.0, 0.0, 0.8]), np.array([1.8, 1.2, 1.5]))
NEW = (np.array([0.9, 0.0, 3.5]), np.array([1.6, 0.7, 4.1]))
PERSON = (np.array([3.2, 0.0, 1.0]), np.array([3.6, 1.75, 1.3]))
CAM = np.array([3.0, 1.4, 2.5])


def rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def blocks_of(box):
    """The plan blocks a box fills, as the build would have placed it."""
    (x0, y0, z0), (x1, y1, z1) = box
    xs = range(int(x0 / W * 8), int(np.ceil(x1 / W * 8)))
    zs = range(int(z0 / D * 6), int(np.ceil(z1 / D * 6)))
    ys = range(1, int(np.ceil(y1 / H * 3)) + 1)
    return {(x, y, z) for x in xs for y in ys for z in zs}


def slab(o, dirs, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / dirs, (hi - o) / dirs
    tn = np.nanmax(np.minimum(t1, t2), -1)
    tf = np.nanmin(np.maximum(t1, t2), -1)
    return np.where((tn <= tf) & (tn > 0), tn, np.inf)


def render(c2w_cv, K, boxes, person):
    u, v = np.meshgrid(np.arange(DW) + 0.5, np.arange(DH) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dirs = rays @ c2w_cv[:3, :3].T
    o = c2w_cv[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs > 0, (np.array([W, H, D]) - o) / dirs, (0 - o) / dirs)
    t[~np.isfinite(t)] = np.inf
    depth = t.min(-1)
    for lo, hi in boxes:
        depth = np.minimum(depth, slab(o, dirs, lo, hi))
    mask = np.zeros(depth.shape, bool)
    if person:
        tp = slab(o, dirs, *PERSON)
        mask = tp < depth
        depth = np.minimum(depth, tp)
    return depth, mask


WORLD = np.eye(4); WORLD[:3, :3] = rot_y(TURN)


def frame_message(i, boxes, conf=2, person=False, tracking="normal", turn=np.pi / 6):
    R = rot_y(i * turn) @ rot_x(np.radians(-35 if i % 2 else -15))
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = CAM
    K = np.array([[FX, 0, IW / 2], [0, FX, IH / 2], [0, 0, 1]])
    Kd = K.copy(); Kd[:2] *= DW / IW
    depth, mask = render(c2w @ ARKIT_TO_CV, Kd, boxes, person)
    img = io.BytesIO()
    Image.fromarray(np.full((IH // 8, IW // 8, 3), 180, np.uint8)).resize((IW, IH)).save(img, "JPEG")
    blobs = [("image", "jpeg", img.getvalue()),
             ("depth", "f16", np.minimum(depth, 60).astype("<f2").tobytes()),
             ("confidence", "u8", np.full((DH, DW), conf, np.uint8).tobytes()),
             ("person", "u8", mask.astype(np.uint8).tobytes())]
    header = {"type": "frame", "seq": i, "time": i * 0.2, "tracking": tracking, "imageSize": [IW, IH],
              "depthSize": [DW, DH], "K": K.ravel().tolist(), "cam2world": (WORLD @ c2w).ravel().tolist(),
              "blobs": [{"name": n, "format": f, "length": len(b)} for n, f, b in blobs]}
    h = json.dumps(header).encode()
    payload = b"".join(b for _, _, b in blobs)
    return struct.pack(">I", len(h)) + h + struct.pack(">I", len(payload)) + payload


class Worker:
    def __init__(self, built):
        tmp = Path(tempfile.mkdtemp())
        ex, ez = rot_y(TURN) @ [1.0, 0, 0], rot_y(TURN) @ [0, 0, 1.0]
        plan = {"arkit": {"origin": [0.0, 0.0, 0.0], "R": [ex.tolist(), [0.0, 1.0, 0.0], ez.tolist()], "y_shift": 0.0,
                          "f": 1.0, "xmin": 0.0, "xmax": W, "zmin": 0.0, "zmax": D, "hm": H,
                          "W": round(W * S), "D": round(D * S), "H": round(H * S)}}
        (tmp / "plan.json").write_text(json.dumps(plan))
        (tmp / "baseline.json").write_text(json.dumps(sorted(built)))
        script = Path(__file__).resolve().parent / "spectate_analyzer.py"
        self.p = subprocess.Popen([sys.executable, str(script), "--plan", str(tmp / "plan.json"),
                                   "--baseline", str(tmp / "baseline.json")], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.solid = set(built)
        self.people, self.i, self.ms = [], 0, []

    def send(self, boxes, **kw):
        self.p.stdin.write(frame_message(self.i, boxes, **kw)); self.p.stdin.flush()
        self.i += 1
        while True:
            out = json.loads(self.p.stdout.readline())
            if out["type"] == "update":
                break
        assert "error" not in out, out
        self.ms.append(out["ms"])
        for x, y, z, b in out["changes"]:
            (self.solid.discard if b == "minecraft:air" else self.solid.add)((x, y, z))
        self.people = out["people"]
        return out


crate, low = blocks_of(CRATE), blocks_of(LOW)
assert low == {(2, 1, 2)}, low
built = crate | low
w = Worker(built)

for _ in range(15):
    w.send([CRATE, LOW])
assert w.solid == built, f"the room as built changed: {w.solid ^ built}"
print(f"as built: {len(built)} block(s) inside, unchanged; the low box stays")

w.send([CRATE, LOW, GHOST])
for _ in range(3):
    w.send([CRATE, LOW])
for _ in range(15):
    w.send([LOW], conf=1)
assert w.solid == built, f"a one-frame box or low-confidence depth changed the world: {w.solid ^ built}"
print("one-frame box and low-confidence views: no change")

for _ in range(15):
    w.send([LOW], tracking="limited")
assert w.solid == built, "frames while relocalising changed the world"
print("relocalising frames: no change")

for _ in range(48):   # the crate is in view 2-3 frames per turn; 5 clear views in a row are needed
    w.send([LOW])
assert w.solid == low, f"after the crate went: {w.solid}"
print(f"crate removed: {len(crate)} block(s) cleared, nothing else")

for _ in range(48):   # turning slowly, as someone looking around does: a new thing is in view for ~9 frames
    w.send([LOW, NEW], turn=np.pi / 24)
added = w.solid - low
assert added and all(1 <= y <= 2 for _, y, _ in added) and added <= blocks_of(NEW) | {(x + dx, y, z + dz) for x, y, z in blocks_of(NEW) for dx in (-1, 0, 1) for dz in (-1, 0, 1)}, added
print(f"new box: {len(added)} block(s) added where it stands, no floor, wall or ceiling blocks")

before = set(w.solid)
for _ in range(12):
    w.send([LOW, NEW], person=True)
assert w.people, "no person reported"
assert w.solid == before, f"the person was built: {w.solid ^ before}"
p = w.people[0]
print(f"person at block ({p['x']:.1f}, {p['z']:.1f}), {len(w.people)} reported")
w.p.stdin.close()
w.p.wait(timeout=10)
print(f"median {np.median(w.ms):.0f} ms per frame")
assert np.median(w.ms) < 200, "too slow for 5 updates a second"
print("OK")

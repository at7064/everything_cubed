"""Sanity test for spectate_analyzer.py: a synthetic live stream through its stdin / stdout, as the mod runs it.

Room: 6.0 m (x) by 5.0 m (z), 2.6 m high, in ARKit's world frame turned 20 degrees about gravity, with a 1 m crate
on the floor. The phone stands in the middle and keeps turning, 5 frames a second.
1. The initial scan (15 frames) builds the floor, walls and the crate.
2. The crate is carried away: once the phone has clearly seen the empty floor for a second, its blocks go.
3. A box that shows up in one frame only, and one that only low-confidence depth sees, change nothing.
4. A person (masked by the phone) walks in: reported as a person, never built.
Run: python test_spectate.py
"""
import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from perception import ARKIT_TO_CV

W, D, H = 6.0, 5.0, 2.6
IW, IH, DW, DH = 1920, 1440, 256, 192
FX = 1450.0
TURN = np.radians(20)
CRATE = (np.array([3.8, 0.0, 3.2]), np.array([4.8, 1.0, 4.2]))
GHOST = (np.array([1.0, 0.0, 1.0]), np.array([2.0, 1.2, 2.0]))
PERSON = (np.array([3.2, 0.0, 1.0]), np.array([3.6, 1.75, 1.3]))
CAM = np.array([2.5, 1.4, 2.5])


def rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


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


def frame_message(i, boxes, conf=2, person=False):
    world = np.eye(4); world[:3, :3] = rot_y(TURN)
    R = rot_y(i * np.pi / 6) @ rot_x(np.radians(-30 if i % 2 else -15))
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
    header = {"type": "frame", "seq": i, "time": i * 0.2, "tracking": "normal", "imageSize": [IW, IH],
              "depthSize": [DW, DH], "K": K.ravel().tolist(), "cam2world": (world @ c2w).ravel().tolist(),
              "blobs": [{"name": n, "format": f, "length": len(b)} for n, f, b in blobs]}
    h = json.dumps(header).encode()
    payload = b"".join(b for _, _, b in blobs)
    return struct.pack(">I", len(h)) + h + struct.pack(">I", len(payload)) + payload


class Worker:
    def __init__(self):
        script = Path(__file__).resolve().parent / "spectate_analyzer.py"
        self.p = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.world = {}
        self.people = []
        self.i = 0
        self.ms = []

    def send(self, boxes, conf=2, person=False):
        self.p.stdin.write(frame_message(self.i, boxes, conf, person)); self.p.stdin.flush()
        self.i += 1
        while True:
            out = json.loads(self.p.stdout.readline())
            if out["type"] == "update":
                break
        assert "error" not in out, out
        self.ms.append(out["ms"])
        for x, y, z, b in out["changes"]:
            self.world[(x, y, z)] = b
        self.people = out["people"]
        return out

    def solid(self):
        return {k for k, b in self.world.items() if b != "minecraft:air"}


# The same room scanned without the crate, for reference.
ref = Worker()
for _ in range(15):
    ref.send([])
empty_room = ref.solid()
ref.p.stdin.close()

w = Worker()
baseline = None
for _ in range(15):
    out = w.send([CRATE])
    if out.get("baseline"):
        baseline = out
assert baseline, "no baseline after 15 frames"
floor = {c for c in w.solid() if c[1] == 0}
crate = {c for c in w.solid() - empty_room if c[1] >= 1}
assert len(floor) > 20, len(floor)
assert crate, "the crate was not built"
print(f"baseline: {len(w.solid())} blocks, {len(floor)} floor, crate {len(crate)} block(s)")

# 3a. A box in one frame only: nothing changes.
before = dict(w.world)
w.send([CRATE, GHOST])
for _ in range(2):
    w.send([CRATE])
assert w.world == before, "a one-frame box changed the world"
# 3b. The crate gone, but only low-confidence depth sees it: nothing changes.
for _ in range(12):
    w.send([], conf=1)
assert w.world == before, "low-confidence depth changed the world"
print("one-frame box and low-confidence views: no change")

# 2. The crate gone for real.
before_solid = w.solid()
for _ in range(48):   # the crate is in view 2-3 frames per turn; 5 clear views are needed
    w.send([])
assert not crate & w.solid(), f"crate blocks stayed after it was taken away: {crate & w.solid()}"
lost = (before_solid - crate) - w.solid()
assert not lost, f"blocks other than the crate went missing: {lost}"
print(f"crate removed: {len(crate)} block(s) cleared, nothing else")

# 4. A person walks in.
before = w.solid()
for _ in range(12):
    w.send([], person=True)
assert w.people, "no person reported"
assert w.solid() <= before | set(), f"the person was built: {w.solid() - before}"
p = w.people[0]
print(f"person at ({p['x']:.1f}, {p['z']:.1f}) blocks, {len(w.people)} reported")
w.p.stdin.close()
w.p.wait(timeout=10)
print(f"median {np.median(w.ms[16:]):.0f} ms per live frame")
assert np.median(w.ms[16:]) < 200, "too slow for 5 updates a second"
print("OK")

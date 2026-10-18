"""Sanity test for lidar_analyzer.py on a synthetic LiDAR scan, written the way the mod saves one.

Room: 6.0 m (x) by 5.0 m (z), 2.6 m high, in ARKit's world frame (+Y up), turned 20 degrees about gravity.
The east wall (x = 6) has a 1.6 m x 1.2 m window that the LiDAR sees through (to a wall 3 m outside), and a
1.2 m x 0.8 m table with its top at 0.75 m stands in the room. Two pillars stand 1.5 m in front of the
west wall; the floor is seen behind them, so they must not be taken for the wall.
Checks the room size, that only the window wall gets glass, and that the table is found from the depth alone
(no room.json) and from RoomPlan's box, and that "uniform" walls are maps on every block in one flat colour.
Run: python test_lidar.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from perception import ARKIT_TO_CV

W, D, H = 6.0, 5.0, 2.6
IW, IH, DW, DH = 1920, 1440, 256, 192
FX = 1450.0
TURN = np.radians(20)
WINDOW = dict(y=(0.9, 2.1), z=(1.6, 3.2))
TABLE = (np.array([3.6, 0.0, 2.6]), np.array([4.8, 0.75, 3.4]))
PILLARS = [(np.array([1.3, 0.0, z0]), np.array([1.6, H, z0 + 0.3])) for z0 in (1.0, 3.5)]


def rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def slab(o, dirs, lo, hi):
    """Distance to an axis-aligned box seen from outside (inf when missed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / dirs, (hi - o) / dirs
    tn = np.nanmax(np.minimum(t1, t2), -1)
    tf = np.nanmin(np.maximum(t1, t2), -1)
    return np.where((tn <= tf) & (tn > 0), tn, np.inf)


def depth_map(c2w_cv, K, w, h):
    """Room-local ray casting; t along rays with z_cam = 1 is the depth."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dirs = rays @ c2w_cv[:3, :3].T
    o = c2w_cv[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs > 0, (np.array([W, H, D]) - o) / dirs, (0 - o) / dirs)
    t[~np.isfinite(t)] = np.inf
    tw = t.min(-1)
    hit = o + tw[..., None] * dirs
    glass = (np.abs(hit[..., 0] - W) < 1e-6) & (hit[..., 1] > WINDOW["y"][0]) & (hit[..., 1] < WINDOW["y"][1]) \
        & (hit[..., 2] > WINDOW["z"][0]) & (hit[..., 2] < WINDOW["z"][1])
    with np.errstate(divide="ignore", invalid="ignore"):
        outside = (W + 3 - o[0]) / dirs[..., 0]
    tw = np.where(glass, outside, tw)
    for lo, hi in [TABLE] + PILLARS:
        tw = np.minimum(tw, slab(o, dirs, lo, hi))
    return tw


def write_scan(folder: Path, with_room: bool):
    (folder / "depth").mkdir()
    world = np.eye(4); world[:3, :3] = rot_y(TURN)      # room-local -> ARKit world
    frames = []
    for i in range(16):
        R = rot_y(i * np.pi / 8) @ rot_x(np.radians(-25 if i % 2 else 12))
        c2w_local = np.eye(4); c2w_local[:3, :3] = R; c2w_local[:3, 3] = [2.5, 1.4, 2.0]
        K = np.array([[FX, 0, IW / 2], [0, FX, IH / 2], [0, 0, 1]])
        Kd = K.copy(); Kd[:2] *= DW / IW
        depth = depth_map(c2w_local @ ARKIT_TO_CV, Kd, DW, DH)
        name = f"frame_{i + 1:04d}"
        rng = np.random.default_rng(i)
        img = (np.full((IH // 8, IW // 8, 3), 200) + rng.integers(-6, 6, (IH // 8, IW // 8, 1))).astype(np.uint8)
        Image.fromarray(img).resize((IW, IH)).save(folder / f"{name}.jpg")
        np.minimum(depth, 60).astype("<f2").tofile(folder / f"depth/{name}.depth")
        np.full((DH, DW), 2, np.uint8).tofile(folder / f"depth/{name}.conf")
        frames.append({"seq": i, "kind": "keyframe", "tracking": "normal", "image": f"{name}.jpg",
                       "imageSize": [IW, IH], "depthSize": [DW, DH], "K": K.ravel().tolist(),
                       "cam2world": (world @ c2w_local).ravel().tolist(),
                       "depth": {"file": f"depth/{name}.depth", "format": "f16"},
                       "confidence": {"file": f"depth/{name}.conf", "format": "u8"}})
    cap = {"format": "everything_cubed.arkit", "version": 1, "device": "test", "frames": frames}
    if with_room:
        T = np.eye(4); T[:3, 3] = (TABLE[0] + TABLE[1]) / 2
        T = world @ T
        cap["room"] = "room.json"
        (folder / "room.json").write_text(json.dumps({
            "walls": [], "doors": [], "windows": [], "openings": [], "floors": [],
            "objects": [{"category": {"table": {}}, "attributes": {}, "transform": T.T.ravel().tolist(),
                         "dimensions": (TABLE[1] - TABLE[0]).tolist()}]}))
    (folder / "capture.json").write_text(json.dumps(cap))


def run(with_room, detail):
    import lidar_analyzer
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_scan(folder, with_room)
        lidar_analyzer.main(["--images", str(folder), "--wall-detail", detail])
        plan = json.loads((folder / "plan.json").read_text())
        if detail == "uniform":
            # Every wall block is a map - around doors too, painted plain (roomplan.clear_around_doors) - and the
            # plain walls are one colour, with nothing drawn on them.
            for side, w in plan["textures"]["walls"].items():
                assert np.array(w["tiles"], bool).all(), side
                img = np.asarray(Image.open(folder / w["image"]))
                assert len(np.unique(img.reshape(-1, 3), axis=0)) == 1, side
        return plan


for with_room, detail in ((False, "items"), (True, "uniform")):
    plan = run(with_room, detail)
    m = plan["metres"]
    dims = sorted([m["x"], m["z"]])
    assert abs(dims[0] - D) < 0.15 and abs(dims[1] - W) < 0.15, m
    assert abs(m["y"] - H) < 0.1, m
    glass = {o["side"] for o in plan["openings"] if o["type"] == "window"}
    assert len(glass) == 1, plan["openings"]
    tables = [o for o in plan["objects"] if o["type"] == "table"]
    assert len(tables) == 1, plan["objects"]
    # LiDAR rooms are built at their measured size: 1.25 blocks/m for this room, not enlarged like photo rooms.
    assert abs(plan["scale"] - 1.25) < 0.01, plan["scale"]
    print(f"room.json={with_room}, walls {detail}: room {m}, window on {glass}, table {tables[0]}")
print("OK")

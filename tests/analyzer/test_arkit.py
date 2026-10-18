"""Sanity test for the LiDAR (arkit) backend on a synthetic scan, written the way the mod saves one.

Room: 4.0 m (x) by 3.2 m (z), 2.6 m high, in ARKit's world frame (+Y up). Cameras stand in the middle and
look around; half of them hold the phone in portrait, so their images are sideways in sensor orientation.
Checks that every unprojected point lies on the room's box, that every image ends up upright, and that
K + cam2world project each point back onto its own pixel (walltex relies on this).
Run: python tests/analyzer/test_arkit.py
"""
import json
import sys
import tempfile
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2] / "analyzer"   # the modules under test
sys.path.insert(0, str(ANALYZER))

import numpy as np
from PIL import Image

from perception import ARKIT_TO_CV, reconstruct_arkit

W, D, H = 4.0, 3.2, 2.6
IW, IH, DW, DH = 1920, 1440, 256, 192
FX = 1450.0


def rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def box_depth(c2w_cv, K, w, h):
    """Distance along the optical axis to the inside of the room box, per pixel."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1
    dirs = rays @ c2w_cv[:3, :3].T
    o = c2w_cv[:3, 3]
    lo, hi = np.array([0, 0, 0]), np.array([W, H, D])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs > 0, (hi - o) / dirs, (lo - o) / dirs)
    t[~np.isfinite(t)] = np.inf
    return t.min(-1)  # rays have z_cam = 1, so t is the depth


def write_scan(folder: Path):
    (folder / "depth").mkdir()
    frames = []
    for i in range(8):
        portrait = i % 2 == 1
        # ARKit camera: looks down -Z, +Y up in the sensor's landscape orientation. A phone held upright in
        # portrait has the sensor turned a quarter turn about the viewing axis.
        R = rot_y(i * np.pi / 4) @ rot_z(np.pi / 2 if portrait else 0.0)
        c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = [W / 2, 1.4, D / 2]
        K = np.array([[FX, 0, IW / 2 - 3], [0, FX, IH / 2 + 4], [0, 0, 1]])
        Kd = K.copy(); Kd[:2] *= DW / IW
        depth = box_depth(c2w @ ARKIT_TO_CV, Kd, DW, DH)
        name = f"frame_{i + 1:04d}"
        # Image: a vertical gradient in world height, so "upright" is checkable (bright = high).
        Ki = K.copy()
        dimg = box_depth(c2w @ ARKIT_TO_CV, Ki / np.array([[8], [8], [1]]), IW // 8, IH // 8)
        u, v = np.meshgrid(np.arange(IW // 8) + 0.5, np.arange(IH // 8) + 0.5)
        Kq = Ki / np.array([[8], [8], [1]])
        cam = np.stack([(u - Kq[0, 2]) / Kq[0, 0] * dimg, (v - Kq[1, 2]) / Kq[1, 1] * dimg, dimg], -1)
        y = (cam @ (c2w @ ARKIT_TO_CV)[:3, :3].T + c2w[:3, 3])[..., 1]
        g = (np.clip(y / H, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(np.stack([g] * 3, -1)).resize((IW, IH)).save(folder / f"{name}.jpg")
        depth.astype("<f2").tofile(folder / f"depth/{name}.depth")
        np.full((DH, DW), 2, np.uint8).tofile(folder / f"depth/{name}.conf")
        frames.append({"seq": i, "kind": "keyframe", "image": f"{name}.jpg", "imageSize": [IW, IH],
                       "depthSize": [DW, DH], "K": K.ravel().tolist(), "cam2world": c2w.ravel().tolist(),
                       "depth": {"file": f"depth/{name}.depth", "format": "f16"},
                       "confidence": {"file": f"depth/{name}.conf", "format": "u8"}})
    (folder / "capture.json").write_text(json.dumps({"format": "everything_cubed.arkit", "version": 1,
                                                     "device": "test", "frames": frames}))


with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    write_scan(folder)
    views, metric = reconstruct_arkit(folder, print)

assert metric and len(views) == 8
for i, v in enumerate(views):
    P = v["points"][v["mask"]].astype(np.float64)
    # On the box: one coordinate at a face (f16 depth and nearest upsampling leave a few cm).
    gap = np.min(np.abs(np.concatenate([P, [W, H, D] - P], 1)), 1)
    assert np.percentile(gap, 99) < 0.05, (i, np.percentile(gap, 99))
    # Upright: world height falls from the top row to the bottom row, both in points and in the photo.
    h, w = v["rgb"].shape[:2]
    assert (h > w) == (i % 2 == 1), f"view {i}: portrait frames must be turned upright"
    ys = v["points"][..., 1]
    assert ys[: h // 5].mean() > ys[-h // 5:].mean() + 0.3, f"view {i} is not upright"
    g = v["rgb"][..., 0].astype(float)
    assert g[: h // 5].mean() > g[-h // 5:].mean() + 20, f"photo {i} is not upright"
    # K + cam2world project every point back onto its own pixel.
    w2c = np.linalg.inv(v["cam2world"])
    c = v["points"].reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3]
    uv = c @ v["K"].T
    uv = uv[:, :2] / uv[:, 2:]
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    err = np.abs(uv - np.stack([jj.ravel(), ii.ravel()], 1))[v["mask"].ravel()]
    assert err.max() < 1.0, (i, err.max())
print("views:", [v["rgb"].shape[:2] for v in views])
print("OK")

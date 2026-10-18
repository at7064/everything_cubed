"""Checks photos taken in the iOS app under ARKit world tracking without LiDAR (perception.prepare_tracked):
a synthetic scan written the way the mod saves one, with one photo held in portrait (stored sideways, in the
sensor's landscape orientation) and one in landscape, and two marked world points in each.
Checks that such a scan is told apart from a LiDAR scan, that every photo is turned upright, that its K and pose
project the marked points onto their marks, and that crop_intrinsics follows MapAnything's resize and crop.
Run: python test_tracked.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from perception import ARKIT_TO_CV, capture_has_depth, crop_intrinsics, prepare_tracked, view_crop

IW, IH, F = 1920, 1440, 1450.0
K = np.array([[F, 0, IW / 2], [0, F, IH / 2], [0, 0, 1.0]])
C = np.array([0.0, 1.5, 0.0])
P_LOW, P_HIGH = np.array([0.3, 1.2, -3.0]), np.array([-0.2, 2.1, -3.0])   # world +Y is up


def camera(x, y, z):
    """cam2world in OpenCV axes from the camera's x (right), y (down), z (forward) in world coordinates."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = C
    return c2w


def project(K_, c2w, P):
    pc = c2w[:3, :3].T @ (P - c2w[:3, 3])
    return K_[0, 0] * pc[0] / pc[2] + K_[0, 2], K_[1, 1] * pc[1] / pc[2] + K_[1, 2]


# Looking along -Z. Landscape: image right = +X, image down = -Y. Portrait: the sensor's x points down.
landscape = camera(np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0]))
portrait = camera(np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0]))

tmp = Path(tempfile.mkdtemp())
frames = []
for i, c2w in enumerate((portrait, landscape), start=1):
    img = Image.new("RGB", (IW, IH), (128, 128, 128))
    d = ImageDraw.Draw(img)
    for P, col in ((P_LOW, (255, 0, 0)), (P_HIGH, (0, 255, 0))):
        u, v = project(K, c2w, P)
        d.ellipse([u - 6, v - 6, u + 6, v + 6], fill=col)
    img.save(tmp / f"frame_{i:04d}.jpg", quality=95)
    frames.append({"image": f"frame_{i:04d}.jpg", "kind": "photo", "tracking": "normal", "imageSize": [IW, IH],
                   "K": K.ravel().tolist(), "cam2world": (c2w @ ARKIT_TO_CV).ravel().tolist()})   # ARKit axes
(tmp / "capture.json").write_text(json.dumps({"mode": "capture", "source": "tracked", "frames": frames}))

assert not capture_has_depth(tmp), "photos without depth are a tracked scan, not LiDAR"
lidar = tmp / "lidar"
lidar.mkdir()
(lidar / "capture.json").write_text(json.dumps({"frames": [dict(frames[0], depth={"file": "depth/x.depth"})]}))
assert capture_has_depth(lidar)

shots = prepare_tracked(tmp)
assert len(shots) == 2
for (path, K2, c2w2, size), name in zip(shots, ("portrait", "landscape")):
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    assert img.shape[1::-1] == size
    assert (size[1] > size[0]) == (name == "portrait"), (name, size)   # upright: portrait is taller than wide
    for P, ch in ((P_LOW, 0), (P_HIGH, 1)):
        others = [c for c in range(3) if c != ch]
        mask = (img[..., ch] > 200) & (img[..., others[0]] < 80) & (img[..., others[1]] < 80)
        ys, xs = np.nonzero(mask)
        u, v = project(K2, c2w2, P)
        assert len(xs) and abs(xs.mean() - u) < 2 and abs(ys.mean() - v) < 2, (name, ch, xs.mean(), ys.mean(), u, v)
    v_low, v_high = project(K2, c2w2, P_LOW)[1], project(K2, c2w2, P_HIGH)[1]
    assert v_high < v_low, (name, "the higher point must be nearer the top of the upright photo")
    print(f"{name}: upright {size}, marks where K and pose put them")

# crop_intrinsics: a photo pixel maps to the same view pixel as MapAnything's scale-then-centred-crop.
photo, view = (1440, 1920), (392, 518)
Kv = crop_intrinsics(K, photo, view)
l, t, r, b = view_crop(photo, view)
s = view[0] / (r - l)
for u, v in ((100.0, 200.0), (720.0, 960.0), (1300.0, 1800.0)):
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]     # the same ray
    uv = (Kv[0, 0] * x + Kv[0, 2], Kv[1, 1] * y + Kv[1, 2])
    assert abs(uv[0] - (u - l) * s) < 1e-6 and abs(uv[1] - (v - t) * s) < 1e-6, (uv, ((u - l) * s, (v - t) * s))
print("OK")

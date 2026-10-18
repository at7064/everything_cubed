"""Checks photos taken in the iOS app under ARKit world tracking without LiDAR (perception.prepare_tracked):
a synthetic scan written the way the mod saves one, with one photo held in portrait (stored sideways, in the
sensor's landscape orientation) and one in landscape, and two marked world points in each.
Checks that such a scan is told apart from a LiDAR scan, that every photo is turned upright, that its K and pose
project the marked points onto their marks, and that crop_intrinsics follows MapAnything's resize and crop.
Run: python tests/analyzer/test_tracked.py
"""
import json
import sys
import tempfile
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2] / "analyzer"   # the modules under test
sys.path.insert(0, str(ANALYZER))

import numpy as np
from PIL import Image, ImageDraw

from perception import (ARKIT_TO_CV, capture_has_depth, crop_intrinsics, depth_scale_from_points, prepare_tracked,
                        rigid_fit, view_crop)

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
for (path, K2, c2w2, size, _frame), name in zip(shots, ("portrait", "landscape")):
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
# The photo cap is hard (memory): 50 frames, 30 of them shutter photos, capped at 24 -> 24, shutter photos first.
many = tmp / "many"
many.mkdir()
fr = []
for i in range(50):
    Image.new("RGB", (64, 48)).save(many / f"f{i:02d}.jpg")
    fr.append(dict(frames[1], image=f"f{i:02d}.jpg", kind="photo" if i % 5 else "keyframe", imageSize=[64, 48]))
(many / "capture.json").write_text(json.dumps({"frames": fr}))
capped = prepare_tracked(many, max_views=24)
assert len(capped) == 24, len(capped)
print("photo cap OK")

# rigid_fit: MapAnything's output frame = ARKit's turned and moved; the fit undoes it from the camera centres.
rng = np.random.default_rng(0)
cams = []
for k in range(6):
    c = np.eye(4); c[:3, 3] = rng.uniform(-2, 2, 3); cams.append(c)
cams = np.stack(cams)
a = np.radians(164)
Tm = np.eye(4); Tm[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]; Tm[:3, 3] = [0.5, -0.2, 1.0]
out = np.stack([Tm @ c for c in cams])
fit = rigid_fit(out, cams)
assert np.allclose(np.stack([fit @ o for o in out]), cams, atol=1e-9), "rigid_fit must take the output back"

# depth_scale_from_points: a wall 3 m in front, predicted at 2.3 m; ARKit's points on it say 3 m -> scale 1.3.
Hv, Wv = 120, 160
Kv = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
vv, uu = np.mgrid[0:Hv, 0:Wv]
ray = np.stack([(uu - 80) / 100.0, (vv - 60) / 100.0, np.ones_like(uu, float)], -1)
view = {"cam2world": np.eye(4), "K": Kv, "points": ray * (3.0 / 1.3), "mask": np.ones((Hv, Wv), bool)}
true_pts = (ray * 3.0).reshape(-1, 3)[rng.choice(Hv * Wv, 300, replace=False)]
s = depth_scale_from_points(view, true_pts)
assert abs(s - 1.3) < 0.02, s
assert depth_scale_from_points(view, true_pts[:5]) is None, "too few points: no correction"
print("rigid fit + depth correction OK", round(s, 3))

# Batches (reconstruct_tracked batch_views): a stand-in for MapAnything answers every batch in its own frame (a
# different turn and shift each time, as the real one answers in its first camera's frame). Every photo must still
# land in ARKit's world, batches must be even and in the order the photos were taken, and no batch may be tiny.
import sys
import types
import torch
from perception import reconstruct_tracked

ring = tmp / "ring"
ring.mkdir()
fr = []
for i in range(10):
    a = 2 * np.pi * i / 10   # walking a circle of radius 1.5 m, looking outwards (ARKit axes: camera looks along -Z)
    c2w_ar = np.eye(4)
    c2w_ar[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    c2w_ar[:3, 3] = [1.5 * np.sin(-a), 1.4, -1.5 * np.cos(a)]
    Image.new("RGB", (64, 48), (90, 90, 90)).save(ring / f"r{i:02d}.jpg")
    fr.append({"image": f"r{i:02d}.jpg", "kind": "keyframe", "tracking": "normal", "imageSize": [64, 48],
               "K": [50.0, 0, 32, 0, 50.0, 24, 0, 0, 1], "cam2world": c2w_ar.ravel().tolist()})
(ring / "capture.json").write_text(json.dumps({"frames": fr}))

calls = []
gen = np.random.default_rng(1)


class FakeModel:
    def to(self, device):
        return self

    def infer(self, views, **kw):
        ang, shift = gen.uniform(0, 2 * np.pi), gen.uniform(-3, 3, 3)
        G = np.eye(4)
        G[:3, :3] = [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
        G[:3, 3] = shift
        calls.append([float(v["camera_poses"][0, 0, 3]) for v in views])
        preds = []
        for v in views:
            c2w = v["camera_poses"][0].numpy().astype(np.float64)
            H, W = v["img"].shape[-2:]
            yy, xx = np.mgrid[0:H, 0:W]
            K_ = v["intrinsics"][0].numpy()
            ray = np.stack([(xx - K_[0, 2]) / K_[0, 0], (yy - K_[1, 2]) / K_[1, 1], np.ones((H, W))], -1)
            world = (ray * 2.0) @ c2w[:3, :3].T + c2w[:3, 3]           # a surface 2 m in front of every camera
            preds.append({"pts3d": torch.from_numpy(world @ G[:3, :3].T + G[:3, 3]).float()[None],
                          "camera_poses": torch.from_numpy(G @ c2w).float()[None],
                          "intrinsics": v["intrinsics"], "conf": torch.ones(1, H, W),
                          "mask": torch.ones(1, H, W, 1), "img_no_norm": torch.zeros(1, H, W, 3)})
        return preds


def fake_load_images(paths):
    return [{"img": torch.zeros(1, 3, 28, 42)} for _ in paths]


mods = {"mapanything": types.ModuleType("mapanything"), "mapanything.models": types.ModuleType("mapanything.models"),
        "mapanything.utils": types.ModuleType("mapanything.utils"),
        "mapanything.utils.image": types.ModuleType("mapanything.utils.image")}
mods["mapanything.models"].MapAnything = types.SimpleNamespace(from_pretrained=lambda name: FakeModel())
mods["mapanything.utils.image"].load_images = fake_load_images
saved = {k: sys.modules.get(k) for k in mods}
sys.modules.update(mods)
try:
    for batch, sizes in ((0, [10]), (4, [4, 3, 3]), (3, [3, 3, 2, 2]), (20, [10])):
        calls.clear()
        views, metric, paths = reconstruct_tracked(ring, "cpu", "fake", lambda m: None, max_views=24,
                                                   work_dir=tmp / f"up{batch}", batch_views=batch)
        assert [len(c) for c in calls] == sizes, (batch, [len(c) for c in calls])
        order = [x for c in calls for x in c]
        shots = prepare_tracked(ring, 24, tmp / "chk")
        assert np.allclose(order, [s[2][0, 3] for s in shots], atol=1e-5), "batches keep the photos in order"
        for v, (_, _K, c2w, _size, _f) in zip(views, shots):
            Kv = v["K"]
            H, W = v["points"].shape[:2]
            yy, xx = np.mgrid[0:H, 0:W]
            ray = np.stack([(xx - Kv[0, 2]) / Kv[0, 0], (yy - Kv[1, 2]) / Kv[1, 1], np.ones((H, W))], -1)
            want = (ray * 2.0) @ c2w[:3, :3].T + c2w[:3, 3]
            assert np.abs(v["points"] - want).max() < 1e-3, (batch, np.abs(v["points"] - want).max())
            assert np.allclose(v["cam2world"], c2w)
finally:
    for k, m in saved.items():
        if m is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = m
print("batched reconstruction OK")
print("OK")

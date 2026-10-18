"""Photos -> labelled, coloured 3D point cloud, using established open-source models.

* 3D reconstruction from any number of unposed photos:
    - MapAnything (Meta, https://github.com/facebookresearch/map-anything) - metric scale, preferred
    - VGGT (Meta, https://github.com/facebookresearch/vggt) - fallback; its up-to-scale output is made
      metric with Depth Anything V2 Metric-Indoor (via Hugging Face transformers)
* LiDAR scans streamed from the iOS scanner app (capture.json): ARKit already gives metric depth and
  camera poses per photo, so no reconstruction model is needed
* Semantic segmentation: SegFormer fine-tuned on ADE20K (150 indoor/outdoor classes incl. wall, floor,
  ceiling, bed, window, door, table, sofa, ...) via Hugging Face transformers.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


def list_images(folder: Path) -> list[Path]:
    # Only the photos themselves: never our own outputs (textures/, or tex_*.png left by older versions).
    imgs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT
                  and not p.name.startswith((".", "tex_")))
    if not imgs:
        raise SystemExit(f"no photos found in {folder}")
    return imgs


def pick_device(requested: str | None):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _np(t):
    return t.detach().float().cpu().numpy()


# ----------------------------------------------------------------------------------- reconstruction

FULL_FRAME_DIAGONAL_MM = 43.2666  # a 36 x 24 mm frame: "35 mm equivalent" focal lengths are relative to it


def exif_focal_px(path):
    """Focal length in pixels of the photo as stored (after EXIF orientation), from its 35 mm-equivalent focal
    length (the diagonal definition), or None when the photo does not say."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)  # Exif FocalLengthIn35mmFilm
        w, h = ImageOps.exif_transpose(im).size
    if not f35:
        return None
    return float(f35) / FULL_FRAME_DIAGONAL_MM * float(np.hypot(w, h)), (w, h)


def crop_intrinsics(K, photo_size, view_size):
    """A photo's pinhole K after MapAnything's resize: uniform scale, then a centred crop (see view_crop)."""
    (W0, H0), (Wv, Hv) = photo_size, view_size
    s = max(Wv / W0, Hv / H0)
    left, top = (W0 * s - Wv) / 2, (H0 * s - Hv) / 2
    return np.array([[K[0, 0] * s, 0, K[0, 2] * s - left], [0, K[1, 1] * s, K[1, 2] * s - top], [0, 0, 1.0]])


def view_intrinsics(focal, photo_size, view_size):
    """Pinhole K of a photo after MapAnything's resize: uniform scale, then a centred crop (see view_crop)."""
    (W0, H0), (Wv, Hv) = photo_size, view_size
    s = max(Wv / W0, Hv / H0)
    return np.array([[focal * s, 0, Wv / 2], [0, focal * s, Hv / 2], [0, 0, 1.0]])


def reconstruct_mapanything(paths, device, model_name, log, use_exif=True):
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    log(f"[recon] MapAnything ({model_name}) on {len(paths)} photo(s), device={device}")
    model = MapAnything.from_pretrained(model_name).to(device)
    views = load_images([str(p) for p in paths])
    # The camera geometry, when every photo states its lens: MapAnything then does not have to guess the field of
    # view, which otherwise feeds straight into the scale.
    focals = [exif_focal_px(p) for p in paths] if use_exif and len(paths) == len(views) else [None]
    if all(focals):
        for v, (f, size) in zip(views, focals):
            Hv, Wv = v["img"].shape[-2:]
            v["intrinsics"] = torch.from_numpy(view_intrinsics(f, size, (Wv, Hv))).float()[None]
        fov = np.degrees(2 * np.arctan(focals[0][1][0] / 2 / focals[0][0]))
        log(f"[recon] camera geometry from the photos' lens data (horizontal field of view {fov:.0f} deg)")
    elif use_exif:
        log("[recon] some photos have no lens data; MapAnything estimates the camera geometry itself")
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=device == "cuda",
                            amp_dtype="bf16", apply_mask=True, mask_edges=True,
                            apply_confidence_mask=False, confidence_percentile=10)
    return _mapanything_views(preds), True


def _mapanything_views(preds):
    out = []
    for p in preds:
        img = _np(p["img_no_norm"][0])
        out.append({"points": _np(p["pts3d"][0]),
                    "K": _np(p["intrinsics"][0]).astype(np.float64),
                    "cam2world": _np(p["camera_poses"][0]).astype(np.float64),
                    "conf": _np(p["conf"][0]),
                    "mask": _np(p["mask"][0])[..., 0] > 0.5,
                    "rgb": (img * 255 if img.max() <= 1.5 else img).clip(0, 255).astype(np.uint8)})
    return out


def capture_has_depth(folder: Path) -> bool:
    """True for a LiDAR scan (frames with depth), False for photos tracked by ARKit without depth."""
    import json
    cap = json.loads((folder / "capture.json").read_text())
    return any("depth" in f for f in cap.get("frames", []))


TRACKED_MAX_VIEWS = 24   # MapAnything holds every photo at once: 60 used up 24 GB (and the swap) next to Minecraft


def sharpness(path, size=320):
    """Variance of the Laplacian of a downscaled grey image: low for a photo smeared by motion blur."""
    from PIL import Image
    from scipy import ndimage
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((size, size))
        return float(ndimage.laplace(np.asarray(im, np.float32)).var())


def prepare_tracked(folder: Path, max_views=TRACKED_MAX_VIEWS, out_dir=None):
    """Photos from the iOS app taken under ARKit world tracking (any iPhone, no LiDAR needed): each turned upright
    into `out_dir` (default <folder>/.upright/), with its K and camera pose (OpenCV axes, metres) to match.
    Returns [(path, K, cam2world, (w, h), frame record)]."""
    import json
    from PIL import Image
    cap = json.loads((folder / "capture.json").read_text())
    frames = [f for f in cap["frames"] if f.get("image") and f.get("tracking", "normal") == "normal"]
    if not frames:
        raise SystemExit("the scan has no photos taken with normal tracking")
    if len(frames) > max_views:
        # A hard cap (memory): shutter photos first; the rest of the places go to keyframes spread over the walk,
        # the sharpest of each stretch (frames kept while walking are often smeared by motion blur).
        photos = [i for i, f in enumerate(frames) if f.get("kind") == "photo"][:max_views]
        rest = [i for i in range(len(frames)) if i not in set(photos)]
        n = max_views - len(photos)
        picked = []
        if n > 0 and rest:
            for stretch in np.array_split(np.array(rest), min(n, len(rest))):
                picked.append(int(max(stretch, key=lambda i: sharpness(folder / frames[i]["image"]))))
        frames = [frames[i] for i in sorted(set(photos) | set(picked))]
    out_dir = Path(out_dir) if out_dir else folder / ".upright"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in frames:
        img = Image.open(folder / f["image"]).convert("RGB")
        w, h = img.size
        iw, ih = f.get("imageSize", (w, h))
        K = np.array(f["K"], dtype=np.float64).reshape(3, 3)
        K[0] *= w / iw
        K[1] *= h / ih
        c2w = np.array(f["cam2world"], dtype=np.float64).reshape(4, 4) @ ARKIT_TO_CV
        turns, K, c2w = upright_camera(K, c2w, w, h)
        if turns % 4:
            img = img.rotate(90 * (turns % 4), expand=True)   # PIL turns counterclockwise, like np.rot90
        p = out_dir / Path(f["image"]).name
        img.save(p, quality=95)
        out.append((p, K, c2w, img.size, f))
    return out


def reconstruct_tracked(folder: Path, device, model_name, log, max_views=TRACKED_MAX_VIEWS, work_dir=None):
    """MapAnything given each photo's real camera pose and intrinsics from ARKit, marked metric: it no longer
    guesses the scale or where the photos were taken, only the depth. Returns (views, metric, upright paths)."""
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    shots = prepare_tracked(folder, max_views, work_dir)
    log(f"[recon] {len(shots)} photo(s) with ARKit camera positions -> MapAnything ({model_name}), device={device}")
    model = MapAnything.from_pretrained(model_name).to(device)
    views = load_images([str(p) for p, *_ in shots])
    for v, (_, K, c2w, size, _f) in zip(views, shots):
        Hv, Wv = v["img"].shape[-2:]
        v["intrinsics"] = torch.from_numpy(crop_intrinsics(K, size, (Wv, Hv))).float()[None]
        v["camera_poses"] = torch.from_numpy(c2w).float()[None]
        v["is_metric_scale"] = torch.tensor([True])
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=device == "cuda",
                            amp_dtype="bf16", apply_mask=True, mask_edges=True,
                            apply_confidence_mask=False, confidence_percentile=10)
    out = _mapanything_views(preds)
    # MapAnything answers in its first camera's frame. It honours ARKit's poses (within ~1 cm on a real scan), so one
    # rigid map takes everything back to ARKit's world, where the feature points and the planes are.
    to_arkit = rigid_fit(np.stack([v["cam2world"] for v in out]), np.stack([s[2] for s in shots]))
    scales = []
    for v, (_, _K, c2w, _size, frame) in zip(out, shots):
        v["points"] = v["points"] @ to_arkit[:3, :3].T + to_arkit[:3, 3]
        v["cam2world"] = c2w
        pts = load_feature_points(folder, frame)
        if pts is not None:
            s = depth_scale_from_points(v, pts)
            if s is not None:
                C = c2w[:3, 3]
                v["points"] = C + s * (v["points"] - C)   # this photo's depth, rescaled to agree with ARKit's points
                scales.append(s)
    if scales:
        log(f"[recon] depth of {len(scales)}/{len(out)} photo(s) corrected with ARKit's feature points "
            f"(median change {np.median(np.abs(np.array(scales) - 1)) * 100:.0f}%)")
    return out, True, [p for p, *_ in shots]


def rigid_fit(src_c2w, dst_c2w):
    """4x4 rigid transform taking poses src -> dst (no scale): from the camera centres when they are spread out,
    else from the first camera alone."""
    a, b = src_c2w[:, :3, 3], dst_c2w[:, :3, 3]
    T = np.eye(4)
    if len(a) >= 3 and np.linalg.norm(a - a.mean(0), axis=1).mean() > 0.2:
        Ma, Mb = a - a.mean(0), b - b.mean(0)
        U, _, Vt = np.linalg.svd(Ma.T @ Mb)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
        T[:3, :3], T[:3, 3] = Rm, b.mean(0) - Rm @ a.mean(0)
        return T
    return dst_c2w[0] @ np.linalg.inv(src_c2w[0])


def load_planes(folder: Path):
    """The flat surfaces ARKit found during a photo scan (planes.json), as build_plan's `planes`: world normal
    (the plane's local +Y), centre point, area and label, in ARKit's world frame. [] when there are none."""
    import json
    f = folder / "planes.json"
    if not f.is_file():
        return []
    out = []
    for p in json.loads(f.read_text()).get("planes", []):
        T = np.array(p["transform"], dtype=np.float64).reshape(4, 4)
        centre = T @ np.r_[np.array(p.get("center", [0, 0, 0]), float), 1.0]
        w, h = p.get("extent", [0, 0])
        out.append({"normal": T[:3, 1], "point": centre[:3], "area": float(w) * float(h),
                    "label": p.get("classification", "none")})
    return out


def load_feature_points(folder: Path, frame):
    """ARKit's tracked feature points of one frame (world, metres, ARKit axes), or None."""
    ref = frame.get("points")
    if not ref or not (folder / ref["file"]).is_file():
        return None
    pts = np.fromfile(folder / ref["file"], dtype="<f4")
    return pts.reshape(-1, 3).astype(np.float64) if len(pts) >= 3 else None


def depth_scale_from_points(view, pts_world, min_points=15):
    """How much this view's depth must be scaled to agree with ARKit's measured feature points: the median ratio
    of their true depth to the predicted depth at the pixels they project to (None: too few points)."""
    c2w, K = view["cam2world"], view["K"]
    w2c = np.linalg.inv(c2w)
    pc = pts_world @ w2c[:3, :3].T + w2c[:3, 3]
    pc = pc[pc[:, 2] > 0.1]
    if not len(pc):
        return None
    h, w = view["points"].shape[:2]
    u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z_true = u[inside], v[inside], pc[inside, 2]
    P = view["points"][v, u]
    z_pred = (P - c2w[:3, 3]) @ c2w[:3, 2]          # depth along this camera's viewing axis
    ok = view["mask"][v, u] & np.isfinite(z_pred) & (z_pred > 0.05)
    if ok.sum() < min_points:
        return None
    return float(np.clip(np.median(z_true[ok] / z_pred[ok]), 0.5, 2.0))


def reconstruct_vggt(paths, device, model_name, depth_model, log):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    log(f"[recon] VGGT ({model_name}) on {len(paths)} photo(s), device={device}")
    model = VGGT.from_pretrained(model_name).to(device).eval()
    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
        pred = model(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = _np(pred["depth"][0])                       # (S,H,W,1)
    pts = unproject_depth_map_to_point_map(depth, _np(extr[0]), _np(intr[0]))
    conf = _np(pred["depth_conf"][0])
    rgb = (_np(images).transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    out = []
    for i in range(len(rgb)):
        w2c = np.eye(4); w2c[:3] = _np(extr[0, i])
        c2w = np.linalg.inv(w2c)
        out.append({"points": pts[i], "conf": conf[i], "mask": depth[i, ..., 0] > 0, "rgb": rgb[i],
                    "K": _np(intr[0, i]).astype(np.float64), "cam2world": c2w})

    metric = False
    if depth_model:
        try:
            scale = metric_scale(out, depth[..., 0], device, depth_model, log)
            for v in out:
                v["points"] = v["points"] * scale
                v["cam2world"][:3, 3] *= scale
            metric = True
        except Exception as ex:
            log(f"[recon] metric depth unavailable ({ex}); falling back to size priors")
    return out, metric


def metric_scale(views, rel_depth, device, model_name, log):
    """Median ratio between Depth Anything V2 metric depth and the reconstruction's depth."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    ratios = []
    for v, d in zip(views, rel_depth):
        inp = proc(images=Image.fromarray(v["rgb"]), return_tensors="pt").to(device)
        with torch.no_grad():
            pd = model(**inp).predicted_depth
        pd = torch.nn.functional.interpolate(pd[:, None], size=d.shape, mode="bilinear", align_corners=False)
        m = _np(pd[0, 0])
        good = (d > 0) & (v["conf"] > np.percentile(v["conf"], 30)) & np.isfinite(m) & (m > 0.1)
        if good.sum() > 100:
            ratios.append(np.median(m[good] / d[good]))
    if not ratios:
        raise RuntimeError("no valid depth overlap")
    s = float(np.median(ratios))
    log(f"[recon] metric scale factor {s:.3f} (Depth Anything V2)")
    return s


ARKIT_WIDTH = 640           # working resolution for LiDAR views (depth itself is 256x192)
ARKIT_MAX_DEPTH = 5.0       # metres; LiDAR is unreliable beyond this
ARKIT_MAX_VIEWS = 60
# ARKit cameras look down -Z with +Y up; OpenCV-style cameras (used everywhere here) look down +Z with +Y down.
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def upright_camera(K, cam2world, w, h):
    """Quarter turns (counterclockwise) that bring the world's up (+Y in ARKit) to the top of an image of size
    (w, h), and K and cam2world (OpenCV axes) after them. ARKit images are always in the sensor's landscape
    orientation, whatever way the phone was held. Returns (turns, K, cam2world)."""
    K, cam2world = np.array(K, float), np.array(cam2world, float)
    rot = np.eye(4); rot[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]   # new camera axes from old
    for turns in range(4):
        up = cam2world[:3, :3].T @ np.array([0.0, 1.0, 0.0])   # world up in camera axes
        if -up[1] >= abs(up[0]):                               # already points to -v (image top)
            return turns, K, cam2world
        # A quarter turn counterclockwise: new (u, v) = (v, W-1-u).
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], w - 1 - K[0, 2]], [0, 0, 1]])
        cam2world = cam2world @ rot.T
        w, h = h, w
    return 4, K, cam2world


def _upright(view):
    """Rotates a view by quarter turns so the world's up points to the top of the image (see upright_camera)."""
    h, w = view["rgb"].shape[:2]
    turns, view["K"], view["cam2world"] = upright_camera(view["K"], view["cam2world"], w, h)
    for k in ("points", "rgb", "conf", "mask"):
        view[k] = np.ascontiguousarray(np.rot90(view[k], turns))
    return view


def reconstruct_arkit(folder: Path, log, max_views=ARKIT_MAX_VIEWS):
    """Views from a capture.json written by the mod for a scan from the iOS app: per photo, the LiDAR depth
    is unprojected with ARKit's intrinsics and camera pose. Metric, gravity-aligned, no ML needed."""
    import json
    from PIL import Image

    cap = json.loads((folder / "capture.json").read_text())
    frames = [f for f in cap["frames"] if "depth" in f]
    if not frames:
        raise ImportError("the scan has no LiDAR depth")
    if len(frames) > max_views:  # evenly thinned; explicit photos are always kept
        keep = set(np.linspace(0, len(frames) - 1, max_views).round().astype(int))
        frames = [f for i, f in enumerate(frames) if i in keep or f.get("kind") == "photo"]
    log(f"[recon] LiDAR scan from {cap.get('device', 'iPhone')}: {len(frames)} posed photo(s) with depth")
    out = []
    for f in frames:
        img = Image.open(folder / f["image"]).convert("RGB")
        iw, ih = f.get("imageSize", img.size)
        s = ARKIT_WIDTH / iw
        W, H = ARKIT_WIDTH, round(ih * s)
        rgb = np.asarray(img.resize((W, H), Image.BILINEAR))
        K = np.array(f["K"], dtype=np.float64).reshape(3, 3)
        K[:2] *= s

        dw, dh = f["depthSize"]
        fmt = f["depth"].get("format", "f16")
        depth = np.fromfile(folder / f["depth"]["file"], dtype="<f2" if fmt == "f16" else "<f4").reshape(dh, dw)
        conf = (np.fromfile(folder / f["confidence"]["file"], dtype=np.uint8).reshape(dh, dw)
                if "confidence" in f else np.full((dh, dw), 2, np.uint8))
        # Nearest-neighbour upsampling: blending depth across an object edge would make points in mid-air.
        yi = np.minimum((np.arange(H) + 0.5) * dh / H, dh - 1).astype(int)
        xi = np.minimum((np.arange(W) + 0.5) * dw / W, dw - 1).astype(int)
        d = depth[yi][:, xi].astype(np.float64)
        c = conf[yi][:, xi].astype(np.float32)

        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
        c2w = np.array(f["cam2world"], dtype=np.float64).reshape(4, 4) @ ARKIT_TO_CV
        pts = cam @ c2w[:3, :3].T + c2w[:3, 3]
        mask = np.isfinite(d) & (d > 0.1) & (d < ARKIT_MAX_DEPTH) & (c >= 1)
        out.append(_upright({"points": pts.astype(np.float32), "K": K, "cam2world": c2w, "conf": c,
                             "mask": mask, "rgb": rgb}))
    return out, True


def reconstruct(paths, backend, device, opts, log):
    errors = []
    order = [backend] if backend != "auto" else ["arkit", "mapanything", "vggt"]
    for b in order:
        try:
            if b == "arkit":
                folder = paths[0].parent
                if not (folder / "capture.json").is_file():
                    raise ImportError(f"no capture.json in {folder}")
                return reconstruct_arkit(folder, log)
            if b == "mapanything":
                return reconstruct_mapanything(paths, device, opts.mapanything_model, log,
                                               use_exif=not getattr(opts, "no_exif_camera", False))
            if b == "vggt":
                return reconstruct_vggt(paths, device, opts.vggt_model, opts.depth_model, log)
        except ImportError as ex:
            errors.append(f"{b}: not installed ({ex})")
    raise SystemExit("no reconstruction backend available:\n  " + "\n  ".join(errors) +
                     "\nInstall one, see analyzer/README.md")


# ----------------------------------------------------------------------------------- segmentation

def segment(views, device, model_name, log):
    """ADE20K semantic labels per pixel. Returns (label maps, id2label)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    log(f"[seg] {model_name}")
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    maps = []
    for v in views:
        img = Image.fromarray(v["rgb"])
        inp = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[v["rgb"].shape[:2]])[0]
        maps.append(seg.cpu().numpy().astype(np.int32))
    return maps, {int(k): v.strip().lower() for k, v in model.config.id2label.items()}


# ----------------------------------------------------------------------------------- fusion

def category_lut(id2label, ade_to_category):
    """Maps segmentation class ids to our category names: returns f(ids) -> names."""
    names = np.array([ade_to_category.get(id2label.get(i, ""), "other") for i in range(max(id2label) + 1)])
    return lambda ids: names[np.clip(ids, 0, len(names) - 1)]


def fuse(views, label_maps, id2label, ade_to_category, voxel=0.03, conf_drop=20):
    """Concatenate all views, drop low-confidence pixels and voxel-downsample (majority label per voxel)."""
    cat_names = sorted(set(ade_to_category.values()) | {"other"})
    lut = np.array([cat_names.index(ade_to_category.get(id2label.get(i, ""), "other"))
                    for i in range(max(id2label) + 1)])
    P, C, L, V = [], [], [], []
    for view_index, (v, lm) in enumerate(zip(views, label_maps)):
        valid = v["mask"] & np.isfinite(v["points"]).all(-1)
        if valid.any():
            valid &= v["conf"] >= np.percentile(v["conf"][valid], conf_drop)
        P.append(v["points"][valid]); C.append(v["rgb"][valid]); L.append(lut[np.clip(lm[valid], 0, len(lut) - 1)])
        V.append(np.full(valid.sum(), view_index, np.int32))
    P, C, L, V = (np.concatenate(P).astype(np.float64), np.concatenate(C).astype(np.float64),
                  np.concatenate(L), np.concatenate(V))
    if not len(P):
        raise SystemExit("the reconstruction has no usable 3D points (photos too dark, too few or not overlapping)")

    key = np.floor(P / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)
    pts = np.stack([np.bincount(inv, P[:, i], n) for i in range(3)], 1) / cnt[:, None]
    rgb = np.stack([np.bincount(inv, C[:, i], n) for i in range(3)], 1) / cnt[:, None]
    # First choose a label within each view, then give each view one vote per 3D voxel.
    # This keeps a large or close image from overwhelming several agreeing viewpoints.
    nc = len(cat_names)
    combo = (inv * len(views) + V) * nc + L
    uc, ucnt = np.unique(combo, return_counts=True)
    view_vox, lab = uc // nc, uc % nc
    order = np.lexsort((-ucnt, view_vox))
    first = np.r_[True, view_vox[order][1:] != view_vox[order][:-1]]
    chosen = order[first]
    votes = (view_vox[chosen] // len(views)) * nc + lab[chosen]
    vc, vote_index, vcount = np.unique(votes, return_inverse=True, return_counts=True)
    pixel_support = np.bincount(vote_index, weights=ucnt[chosen], minlength=len(vc))
    vox, lab = vc // nc, vc % nc
    order = np.lexsort((-pixel_support, -vcount, vox))
    first = np.r_[True, vox[order][1:] != vox[order][:-1]]
    labels = np.empty(n, int)
    labels[vox[order][first]] = lab[order][first]
    return pts, rgb, np.array(cat_names, dtype=object)[labels].astype(str)


def detection_masks(report, views, templates=("window", "door")):
    """Per-view masks of the detector's boxes for `templates` (from perceptron_pass's report), or None."""
    if not report:
        return None
    masks = []
    for v, photo in zip(views, report["photos"]):
        m = np.zeros(v["rgb"].shape[:2], bool)
        for d in photo["detections"]:
            if d["template"] in templates:
                x0, y0, x1, y1 = d["box"]
                m[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] = True
        masks.append(m)
    return masks


# ----------------------------------------------------------------------------------- open-vocabulary detection
# Optional, opt-in (--perceptron): Perceptron Mk1.5 /v1/detect with no categories ("detect all objects").
# Off by default because the photos leave the machine. Before upload each photo is cropped to the part the
# reconstruction kept and pixels SegFormer calls "person" are blurred (best effort, not a guarantee).
# API: https://docs.perceptron.inc/capabilities/detect.md (boxes in pixels of the uploaded image, no scores).

PERCEPTRON_API = "https://api.perceptron.inc/v1"
PERCEPTRON_MODEL = "perceptron-mk1.5"
DET_PREFIX = "det:"           # id2label names of detector classes, kept apart from ADE20K names
KEEP_SEGFORMER = {"wall", "floor", "ceiling"}
# ...except where the detector found something that SegFormer usually calls wall or floor.
# Not windows and doors: their boxes enclose a lot of real wall (a Gothic window's stone tracery took 28% of the
# wall points in Suzzallo and stretched the room), and SegFormer finds them well itself.
MAY_TAKE = {"painting": {"wall"}, "sconce": {"wall"}, "curtain": {"wall"}, "mirror": {"wall"}, "clock": {"wall"},
            "bulletin_board": {"wall"}, "projection_screen": {"wall"}, "radiator": {"wall"},
            "range_hood": {"wall"}, "wall_hooks": {"wall"},
            **{c: {"wall"} for c in ("light_switch", "outlet", "thermostat", "fire_alarm",
                "soap_dispenser", "sanitizer_dispenser", "paper_towel_dispenser")},
            "whiteboard": {"wall"}, "exit_sign": {"wall"}, "fire_extinguisher": {"wall"}, "rug": {"floor"},
            "drinking_fountain": {"wall"},
            # Hanging things are seen against the ceiling (the depth gate keeps the real ceiling behind them).
            "projector": {"ceiling"}, "chandelier": {"ceiling"}, "ceiling_fan": {"ceiling"},
            **{c: {"ceiling"} for c in ("led_panel", "downlight", "linear_light", "pendant_lamp", "smoke_detector")}}
# Detector names that only map by these synonyms when the model's mapping is unavailable. ADE_TO_CATEGORY
# is tried first; IGNORE_NAMES are people and parts of the building, never furniture.
SYNONYMS = {"office chair": "chair", "dining chair": "chair", "couch": "sofa", "loveseat": "sofa",
            "bookshelf": "bookshelf", "bookshelves": "bookshelf", "book shelf": "bookshelf",
            "nightstand": "nightstand", "bedside table": "nightstand", "dresser": "dresser", "drawer": "dresser",
            "filing cabinet": "cabinet",
            "closet": "wardrobe", "end table": "table", "side table": "table", "dining table": "table",
            "television": "tv", "tv": "tv", "laptop": "laptop", "computer monitor": "monitor",
            "desktop monitor": "monitor", "desk lamp": "desk_lamp", "floor lamp": "lamp",
            "table lamp": "desk_lamp", "pendant light": "pendant_lamp", "ceiling light": "ceiling_light",
            "led panel": "led_panel", "troffer": "led_panel", "recessed light": "downlight",
            "downlight": "downlight", "linear light": "linear_light", "suspended light": "linear_light",
            "pendant lamp": "pendant_lamp", "stack of books": "book_stack", "books on desk": "book_stack",
            "mug": "mug", "coffee cup": "mug", "cup": "mug", "small potted plant": "small_plant",
            "desktop printer": "printer", "printer": "printer", "light switch": "light_switch",
            "power outlet": "outlet", "electrical outlet": "outlet", "socket": "outlet",
            "thermostat": "thermostat", "fire alarm": "fire_alarm", "fire alarm box": "fire_alarm",
            "fire alarm bell": "fire_alarm", "fire alarm horn strobe": "fire_alarm",
            "fire alarm pull station": "fire_alarm",
            "alarm pull station": "fire_alarm", "smoke detector": "smoke_detector",
            "soap dispenser": "soap_dispenser", "sanitizer dispenser": "sanitizer_dispenser",
            "hand sanitizer dispenser": "sanitizer_dispenser", "paper towel dispenser": "paper_towel_dispenser",
            "potted plant": "plant", "houseplant": "plant", "picture": "painting", "picture frame": "painting",
            "artwork": "painting", "bulletin board": "bulletin_board", "cork board": "bulletin_board",
            "corkboard": "bulletin_board", "whiteboard": "whiteboard", "dry erase board": "whiteboard",
            "chalkboard": "whiteboard", "blackboard": "whiteboard", "projector": "projector",
            "exit sign": "exit_sign", "fire extinguisher": "fire_extinguisher", "extinguisher": "fire_extinguisher",
            "fire extinguisher cabinet": "fire_extinguisher", "floor fan": "fan", "tower fan": "fan",
            "pedestal fan": "fan", "ceiling fan": "ceiling_fan", "stack of chairs": "chair_stack",
            "stacked chairs": "chair_stack", "student chair desk": "student_desk", "writing arm chair": "student_desk",
            "projection screen": "projection_screen", "pull down screen": "projection_screen",
            "wall heater": "radiator", "range hood": "range_hood", "extractor hood": "range_hood",
            "coat rack": "coat_rack", "wall hooks": "wall_hooks", "coat hooks": "wall_hooks",
            "lectern": "podium", "podium": "podium", "piano": "piano", "carpet": "rug", "trash can": "trash",
            "garbage can": "trash", "wastebasket": "trash", "bin": "trash", "stained glass window": "window",
            "fridge": "fridge", "bathroom sink": "sink", "shower": "shower", "doorway": "door"}
PERSON_WORDS = {"person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "student",
                "students", "human", "face", "head", "hand", "arm", "leg"}
IGNORE_NAMES = {"person", "people", "man", "woman", "boy", "girl", "child", "student", "human", "face", "head",
                "hand", "arm", "leg", "wall", "floor", "ceiling", "column", "pillar", "arch", "beam", "roof",
                "window frame", "door frame", "molding", "railing", "stairs", "staircase", "shadow", "reflection"}


SYNONYMS.update({
    "backpack": "backpack", "bag": "backpack", "school bag": "backpack", "handbag": "backpack", "tote bag": "backpack",
    "globe": "globe", "desk globe": "globe", "locker": "locker", "lockers": "locker", "drinking fountain":
    "drinking_fountain", "water fountain": "drinking_fountain", "bottle filler": "drinking_fountain",
    "bottle filling station": "drinking_fountain", "umbrella stand": "umbrella_stand", "umbrella holder": "umbrella_stand",
})


def _singular(name):
    return name[:-1] if name.endswith("s") and not name.endswith("ss") and len(name) > 3 else name


def synonym_template(name, templates, ade_to_category):
    """Dictionary fallback: detector name -> template id, "ignore" or "generic"."""
    n = " ".join(name.lower().replace("_", " ").split())
    if PERSON_WORDS & set(n.split()):
        return "ignore"
    for k in (n, _singular(n)):
        if k in IGNORE_NAMES:
            return "ignore"
        t = SYNONYMS.get(k) or ade_to_category.get(k) or (k.replace(" ", "_") if k.replace(" ", "_") in templates else None)
        if t in templates:
            return t
    return "generic"


def _post(path, body, key, timeout=300, tries=3):
    import json
    import time
    import urllib.error
    import urllib.request
    key = key.strip()
    if not key or any(c.isspace() or not c.isprintable() for c in key):
        # Never let the key reach an error message (http.client would echo the whole header).
        raise SystemExit("PERCEPTRON_API_KEY is malformed (whitespace or extra text in it); check how it is set")
    req = urllib.request.Request(PERCEPTRON_API + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                          # Cloudflare in front of the API rejects the default Python-urllib agent (error 1010)
                                          "User-Agent": "everything-cubed-analyzer/1.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as ex:  # the API explains itself in {"error": {"message": ...}}
            err = RuntimeError(f"Perceptron {path} HTTP {ex.code}: {ex.read()[:400].decode(errors='replace')}")
            if ex.code != 429 and ex.code < 500:
                raise err from None
        except (TimeoutError, urllib.error.URLError) as ex:  # slow reasoning or a dropped connection
            err = RuntimeError(f"Perceptron {path}: {ex}")
        if attempt + 1 < tries:
            time.sleep(5 * 2 ** attempt)
    raise err from None


def map_names_with_model(names, templates, key, log):
    """Ask Mk1.5 (text only, no image) to pick a template, "unknown" or "ignore" for each name. `templates`
    maps template id -> what it is (a bare collection of ids also works). Returns {name: choice} for the
    names it answered validly."""
    import json
    choices = sorted(templates) + ["unknown", "ignore"]
    schema = {"type": "object", "additionalProperties": False, "required": ["mappings"],
              "properties": {"mappings": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False, "required": ["name", "template"],
                  "properties": {"name": {"type": "string"}, "template": {"type": "string", "enum": choices}}}}}}
    prompt = ("An object detector found these things in photos of a room. Each will be rebuilt in Minecraft "
              "from one furniture template. For every name, choose the template that the object is, or would "
              "be built as. Choose \"unknown\" for a real piece of furniture or a standalone object that no "
              "template fits. Choose \"ignore\" for people or their body parts, parts of the building itself "
              "(walls, floor, ceiling, columns, arches, beams, railings), and things that are only a part of "
              "another object. Answer for every name, spelled exactly as given.\n\nTemplates:"
              + "".join(f"\n- {t}: {templates[t]}" if isinstance(templates, dict) else f"\n- {t}"
                        for t in sorted(templates)) + "\n\nNames:\n" + "\n".join(names))
    resp = _post("/chat/completions", {
        "model": PERCEPTRON_MODEL, "messages": [{"role": "user", "content": prompt}], "n": 1,
        "max_completion_tokens": 64 + 24 * len(names),
        "response_format": {"type": "json_schema", "json_schema": {"name": "template_map", "strict": True, "schema": schema}},
    }, key)
    choice = resp["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise RuntimeError(f"mapping incomplete (finish_reason={choice.get('finish_reason')})")
    out = {}
    for m in json.loads(choice["message"]["content"] or "")["mappings"]:
        if m.get("name") in names and m.get("template") in choices:
            out[m["name"]] = m["template"]
    log(f"[det] model mapped {len(out)}/{len(names)} name(s)")
    return out


def map_names(names, templates, ade_to_category, key, log, use_model=True, vanilla_cat=None):
    """{name: (template id | "vanilla:<id>:<cube|model>" | "generic" | "ignore", source)}. A name that is (nearly)
    a vanilla block's name is that block; the model maps the rest to our templates, the synonym dict if it fails."""
    import vanilla
    out, rest = {}, []
    for n in names:
        hit = vanilla.match(n, vanilla_cat) if vanilla_cat else None
        if hit and synonym_template(n, templates, ade_to_category) != "ignore":
            out[n] = (f"vanilla:{hit[0]}:{'cube' if hit[1] else 'model'}", "vanilla")
        else:
            rest.append(n)
    if out:
        log("[det] vanilla blocks: " + ", ".join(f"{n} -> {t.split(':')[1]}" for n, (t, _) in sorted(out.items())))
    names = rest
    model = {}
    if use_model and names:
        try:
            model = map_names_with_model(names, templates, key, log)
        except Exception as ex:
            log(f"[det] model mapping failed ({ex}); using the synonym dictionary")
    for n in names:
        if n in model:
            out[n] = ("generic" if model[n] == "unknown" else model[n], "model")
        else:
            out[n] = (synonym_template(n, templates, ade_to_category), "synonyms")
    return out


def view_crop(photo_size, view_size):
    """Box (l, t, r, b) in photo pixels that MapAnything / VGGT keep: uniform scale, then a centred crop."""
    W0, H0 = photo_size
    Wv, Hv = view_size
    s = max(Wv / W0, Hv / H0)
    cw, ch = Wv / s, Hv / s
    l, t = (W0 - cw) / 2, (H0 - ch) / 2
    return l, t, l + cw, t + ch


def upload_image(path, view, person_mask, max_side=1536, log=print, find_people=None):
    """The photo region seen by `view`, at up to `max_side` px, people blurred. Falls back to the view's own
    (low-res) pixels when the photo does not line up with the view. `find_people(img)` -> boxes to blur too.
    Returns (PIL image, view px per upload px)."""
    from PIL import Image, ImageOps
    Hv, Wv = view["rgb"].shape[:2]
    img = None
    try:
        photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        crop = photo.crop(tuple(round(c) for c in view_crop(photo.size, (Wv, Hv))))
        small = np.asarray(crop.resize((Wv, Hv), Image.BILINEAR), float)
        err = np.abs(small - view["rgb"]).mean()
        if err < 12:  # mean abs error per channel, 0..255; misaligned crops are typically > 30
            img = crop
            if max(img.size) > max_side:
                img = img.resize((round(img.width * max_side / max(img.size)), round(img.height * max_side / max(img.size))), Image.LANCZOS)
        else:
            log(f"[det] {Path(path).name} does not line up with its view (error {err:.0f}); sending the view pixels")
    except Exception as ex:
        log(f"[det] {Path(path).name}: {ex}; sending the view pixels")
    if img is None:
        img = Image.fromarray(view["rgb"]).resize((Wv * 2, Hv * 2), Image.LANCZOS)
    return blur_people(img, person_mask, find_people(img) if find_people else ()), Wv / img.width


def jpeg_bytes(img):
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def detect_objects(img, key, reasoning="none", cache_dir=None):
    """All objects in one image (PIL or JPEG bytes), as [(name, (x0, y0, x1, y1), mention)] in the image's pixels.
    A mention can be a description ("wall, stone, with arched windows"); the name is its first clause.
    With `cache_dir`, the API's answer is kept per exact upload and reasoning setting, so a re-run sends nothing."""
    import base64
    import hashlib
    import json
    data = img if isinstance(img, bytes) else jpeg_bytes(img)
    cache = Path(cache_dir) / (hashlib.sha256(data + reasoning.encode()).hexdigest()[:32] + ".json") if cache_dir else None
    if cache and cache.exists():
        resp = json.loads(cache.read_text())
    else:
        url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        resp = _post("/detect", {"media": {"type": "image", "image_url": url},
                                 "config": {"output": "box", "reasoning_effort": reasoning}}, key)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(resp))
    out = []
    for d in resp.get("detections", []):
        b = d.get("point_box")
        if not b or not d.get("mention"):
            continue
        x0, y0, x1, y1 = b["top_left"]["x"], b["top_left"]["y"], b["bottom_right"]["x"], b["bottom_right"]["y"]
        name = " ".join(d["mention"].split(",")[0].strip().lower().split())
        if name:
            out.append((name, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), d["mention"].strip()))
    return out, resp.get("finish_reason")


def camera_depth(view):
    """Per-pixel distance along the camera axis; NaN where the reconstruction has no point (MapAnything
    zeroes masked points, so those must not count as real depths)."""
    w2c = np.linalg.inv(view["cam2world"])
    d = view["points"] @ w2c[2, :3] + w2c[2, 3]
    return np.where(view["mask"] & (d > 0), d, np.nan)


def paint_detections(label_map, boxes, id2label, depth=None, band=(0.35, 0.15), pct=30):
    """Paints boxes [(class id, (x0, y0, x1, y1) in label-map pixels[, labels it may take anyway])] into a copy
    of `label_map`. Larger boxes go first so smaller ones stay on top; pixels SegFormer calls wall / floor /
    ceiling are kept, unless the box lists that label (a painting takes wall pixels, a rug floor pixels).
    With `depth` (metres, NaN = unknown), each box only takes pixels within band[0] + band[1] * d of the
    object's depth d: the `pct` percentile depth of the box's central half, over pixels that are not wall /
    floor / ceiling (the object stands in front of what shows through it). A pixel a smaller box rejects
    stays with the larger box below it."""
    from PIL import Image, ImageDraw
    H, W = label_map.shape
    structural = {name: label_map == k for k, name in id2label.items() if name in KEEP_SEGFORMER}
    out = label_map.copy()
    for b in sorted(boxes, key=lambda b: -(b[1][2] - b[1][0]) * (b[1][3] - b[1][1])):
        cid, (x0, y0, x1, y1), may_take = b[0], b[1], (b[2] if len(b) > 2 else set())
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        keep = np.zeros(label_map.shape, bool)
        for name, m in structural.items():
            if name not in may_take:
                keep |= m
        rect = Image.new("1", (W, H), 0)
        ImageDraw.Draw(rect).rectangle([round(x0), round(y0), round(x1) - 1, round(y1) - 1], fill=1)
        take = np.asarray(rect, dtype=bool) & ~keep
        if depth is not None:
            cx0, cx1 = max(0, int((3 * x0 + x1) / 4)), int(np.ceil((x0 + 3 * x1) / 4))
            cy0, cy1 = max(0, int((3 * y0 + y1) / 4)), int(np.ceil((y0 + 3 * y1) / 4))
            core = depth[cy0:cy1, cx0:cx1][~keep[cy0:cy1, cx0:cx1]]
            core = core[np.isfinite(core)]
            if len(core):  # no usable depth: paint the whole box
                d = np.percentile(core, pct)
                with np.errstate(invalid="ignore"):
                    take &= np.abs(depth - d) < band[0] + band[1] * d
        out[take] = cid
    return out


def match_object_observations(observations, max_distance=0.45):
    """Group same-category detections whose reconstructed 3D footprints overlap across views.
    A group needs two distinct views; uncertain single-view detections keep the ordinary cloud path."""
    groups = []
    for obs in sorted(observations, key=lambda o: -len(o["points"])):
        center = np.median(obs["points"], axis=0)
        lo, hi = np.percentile(obs["points"], [10, 90], axis=0)
        best, best_dist = None, np.inf
        for group in groups:
            if group["category"] != obs["category"] or obs["view"] in group["views"]:
                continue
            dist = np.linalg.norm(center - group["center"])
            overlap = np.minimum(hi, group["hi"]) - np.maximum(lo, group["lo"])
            # A close centre alone can merge neighbouring chairs; require footprint overlap too.
            if dist < max_distance and (overlap[[0, 2]] > -0.08).all() and dist < best_dist:
                best, best_dist = group, dist
        if best is None:
            groups.append({"category": obs["category"], "views": {obs["view"]}, "parts": [obs],
                           "center": center, "lo": lo, "hi": hi})
        else:
            best["views"].add(obs["view"])
            best["parts"].append(obs)
            best["center"] = np.median(np.stack([np.median(p["points"], axis=0) for p in best["parts"]]), axis=0)
            best["lo"] = np.min(np.stack([np.percentile(p["points"], 10, axis=0) for p in best["parts"]]), axis=0)
            best["hi"] = np.max(np.stack([np.percentile(p["points"], 90, axis=0) for p in best["parts"]]), axis=0)
    return [{"category": g["category"], "views": sorted(g["views"]),
             "center": g["center"],
             "points": np.concatenate([p["points"] for p in g["parts"]]),
             "colors": np.concatenate([p["colors"] for p in g["parts"]])}
            for g in groups if len(g["views"]) >= 2]


_PERSON_MODEL = {}


def person_boxes(img, device="cpu", score=0.15, pad=0.15):
    """People in `img` (PIL) found locally by torchvision's COCO Faster R-CNN v2 at full upload resolution,
    as padded (x0, y0, x1, y1). The threshold is low on purpose: a false alarm only blurs a chair."""
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    if device not in _PERSON_MODEL:
        w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _PERSON_MODEL[device] = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=score,
                                                           min_size=min(img.size), max_size=max(img.size)).to(device).eval()
    x = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255).permute(2, 0, 1).to(device)
    with torch.no_grad():
        r = _PERSON_MODEL[device]([x])[0]
    out = []
    for (x0, y0, x1, y1), lab, sc in zip(_np(r["boxes"]), _np(r["labels"]), _np(r["scores"])):
        if int(lab) == 1 and sc >= score:  # COCO 1 = person
            px, py = pad * (x1 - x0), pad * (y1 - y0)
            out.append((max(0, x0 - px), max(0, y0 - py), min(img.width, x1 + px), min(img.height, y1 + py)))
    return out


def blur_people(img, person_mask, boxes=()):
    """Blurs `img` (PIL) where the label-map-sized `person_mask` is set (grown a little past the segment edge)
    and inside every box in `boxes` (img pixels)."""
    from PIL import Image, ImageDraw, ImageFilter
    m = Image.new("L", img.size, 0)
    if person_mask is not None and person_mask.any():
        m = Image.fromarray(person_mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST).filter(ImageFilter.MaxFilter(9))
    draw = ImageDraw.Draw(m)
    for b in boxes:
        draw.rectangle([round(c) for c in b], fill=255)
    if not m.getbbox():
        return img
    return Image.composite(img.filter(ImageFilter.GaussianBlur(max(img.size) / 60)), img, m)


def perceptron_pass(paths, views, label_maps, id2label, ade_to_category, templates, log,
                    key_env="PERCEPTRON_API_KEY", reasoning="none", blur=True, depth_gate=True, use_model_map=True,
                    device="cpu", find_people=None, cache_dir=None, workers=5, vanilla_cat=None):
    """Adds open-vocabulary detections to the label maps. Returns (label maps, id2label, ade_to_category,
    report); the inputs are not modified. Detector names become classes "det:<name>" that map to a template
    id, to "generic:<name>" for things without a template, or to "other" when ignored."""
    import os
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"--perceptron needs an API key in ${key_env}")
    if len(paths) != len(views):
        log(f"[det] {len(paths)} photos but {len(views)} views; sending the view pixels")
    person = [k for k, v in id2label.items() if v == "person"]
    find = (find_people or (lambda img: person_boxes(img, device))) if blur else None
    # Uploads are prepared one by one (the person detector shares the GPU), then sent concurrently: the
    # API takes 20-120 s per photo, so the slowest photo sets the total instead of the sum of all of them.
    uploads = []
    for i, (v, lm) in enumerate(zip(views, label_maps)):
        path = paths[i] if len(paths) == len(views) else None
        pm = np.isin(lm, person) if blur else None
        if path is None:
            from PIL import Image
            img = Image.fromarray(v["rgb"]).resize((v["rgb"].shape[1] * 2, v["rgb"].shape[0] * 2), Image.LANCZOS)
            img, k = blur_people(img, pm, find(img) if find else ()), 0.5
        else:
            img, k = upload_image(path, v, pm, log=log, find_people=find)
        uploads.append((path.name if path else f"view {i}", jpeg_bytes(img), k))

    def run(u):
        name, data, k = u
        try:
            dets, finish = detect_objects(data, key, reasoning, cache_dir)
        except RuntimeError as ex:  # one photo failing must not cost the whole analysis
            log(f"[det] {name}: detection failed ({ex}); SegFormer labels only")
            return []
        log(f"[det] {name}: {len(dets)} object(s)" + ("" if finish == "complete" else f" (finish_reason={finish})"))
        return [(n, tuple(c * k for c in b), m) for n, b, m in dets]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, workers)) as ex:
        per_view = list(ex.map(run, uploads))

    names = sorted({d[0] for dets in per_view for d in dets})
    mapping = map_names(names, templates, ade_to_category, key, log, use_model_map, vanilla_cat)
    id2label, ade_to_category = dict(id2label), dict(ade_to_category)
    cls = {}
    for n in names:
        cls[n] = max(id2label) + 1
        id2label[cls[n]] = DET_PREFIX + n
        t = mapping[n][0]
        ade_to_category[DET_PREFIX + n] = ("other" if t == "ignore" else f"generic:{n}" if t == "generic"
                                           else f"{t}:{n}" if t.startswith("vanilla:") else t)

    out_maps, report = [], {"photos": [], "names": {}}
    observations = []
    seg_cat = category_lut(id2label, ade_to_category)
    for i, (v, lm, dets) in enumerate(zip(views, label_maps, per_view)):
        boxes = [(cls[n], b, MAY_TAKE.get(mapping[n][0], set())) for n, b, _ in dets if mapping[n][0] != "ignore"]
        depth = camera_depth(v) if depth_gate else None
        out_maps.append(paint_detections(lm, boxes, id2label, depth))
        rows = []
        for n, (x0, y0, x1, y1), mention in dets:
            t = mapping[n][0]
            cat = ade_to_category[DET_PREFIX + n]
            if cat not in ("other", "window", "door"):
                ys = slice(max(0, int(y0)), min(lm.shape[0], int(np.ceil(y1))))
                xs = slice(max(0, int(x0)), min(lm.shape[1], int(np.ceil(x1))))
                valid = ((out_maps[-1][ys, xs] == cls[n]) & v["mask"][ys, xs]
                         & np.isfinite(v["points"][ys, xs]).all(-1))
                if valid.sum() >= 20:
                    pts, cols = v["points"][ys, xs][valid], v["rgb"][ys, xs][valid]
                    step = max(1, len(pts) // 1500)
                    observations.append({"category": cat, "view": i, "points": pts[::step], "colors": cols[::step]})
            sub = lm[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))]
            cats = seg_cat(sub) if sub.size else np.array([])
            # How much of the box SegFormer already gave the same category (0 = SegFormer missed it).
            seg_share = float((cats == t).mean()) if sub.size and t not in ("ignore", "generic") else None
            top = Counter(c for c in cats.ravel() if c not in KEEP_SEGFORMER | {"other"}).most_common(1) if sub.size else []
            rows.append({"name": n, "mention": mention, "template": t, "box": [round(c, 1) for c in (x0, y0, x1, y1)],
                         "segformer_share": None if seg_share is None else round(seg_share, 3),
                         "segformer_top": top[0][0] if top else None})
        report["photos"].append({"photo": paths[i].name if len(paths) == len(views) else f"view {i}", "detections": rows})
    for n in names:
        report["names"][n] = {"template": mapping[n][0], "source": mapping[n][1],
                              "count": sum(n == d[0] for dets in per_view for d in dets)}
    report["_instances"] = match_object_observations(observations)
    log(f"[det] {len(report['_instances'])} object(s) matched across views")
    return out_maps, id2label, ade_to_category, report

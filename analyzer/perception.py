"""Photos -> labelled, coloured 3D point cloud, using established open-source models.

* 3D reconstruction from any number of unposed photos:
    - MapAnything (Meta, https://github.com/facebookresearch/map-anything) - metric scale, preferred
    - VGGT (Meta, https://github.com/facebookresearch/vggt) - fallback; its up-to-scale output is made
      metric with Depth Anything V2 Metric-Indoor (via Hugging Face transformers)
* Semantic segmentation: SegFormer fine-tuned on ADE20K (150 indoor/outdoor classes incl. wall, floor,
  ceiling, bed, window, door, table, sofa, ...) via Hugging Face transformers.
"""
from __future__ import annotations

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

def reconstruct_mapanything(paths, device, model_name, log):
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    log(f"[recon] MapAnything ({model_name}) on {len(paths)} photo(s), device={device}")
    model = MapAnything.from_pretrained(model_name).to(device)
    views = load_images([str(p) for p in paths])
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=device == "cuda",
                            amp_dtype="bf16", apply_mask=True, mask_edges=True,
                            apply_confidence_mask=False, confidence_percentile=10)
    out = []
    for p in preds:
        img = _np(p["img_no_norm"][0])
        out.append({"points": _np(p["pts3d"][0]),
                    "K": _np(p["intrinsics"][0]).astype(np.float64),
                    "cam2world": _np(p["camera_poses"][0]).astype(np.float64),
                    "conf": _np(p["conf"][0]),
                    "mask": _np(p["mask"][0])[..., 0] > 0.5,
                    "rgb": (img * 255 if img.max() <= 1.5 else img).clip(0, 255).astype(np.uint8)})
    return out, True


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


def reconstruct(paths, backend, device, opts, log):
    errors = []
    order = [backend] if backend != "auto" else ["mapanything", "vggt"]
    for b in order:
        try:
            if b == "mapanything":
                return reconstruct_mapanything(paths, device, opts.mapanything_model, log)
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
    P, C, L = [], [], []
    for v, lm in zip(views, label_maps):
        valid = v["mask"] & np.isfinite(v["points"]).all(-1)
        if valid.any():
            valid &= v["conf"] >= np.percentile(v["conf"][valid], conf_drop)
        P.append(v["points"][valid]); C.append(v["rgb"][valid]); L.append(lut[np.clip(lm[valid], 0, len(lut) - 1)])
    P, C, L = np.concatenate(P).astype(np.float64), np.concatenate(C).astype(np.float64), np.concatenate(L)

    key = np.floor(P / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)
    pts = np.stack([np.bincount(inv, P[:, i], n) for i in range(3)], 1) / cnt[:, None]
    rgb = np.stack([np.bincount(inv, C[:, i], n) for i in range(3)], 1) / cnt[:, None]
    # majority label per voxel
    combo = inv * len(cat_names) + L
    uc, ucnt = np.unique(combo, return_counts=True)
    vox, lab = uc // len(cat_names), uc % len(cat_names)
    order = np.lexsort((-ucnt, vox))
    first = np.r_[True, vox[order][1:] != vox[order][:-1]]
    labels = np.empty(n, int)
    labels[vox[order][first]] = lab[order][first]
    return pts, rgb, np.array(cat_names, dtype=object)[labels].astype(str)

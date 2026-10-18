"""Things hanging on the walls (paintings, posters, clocks, light switches, sockets, thermostats, signs, whiteboards),
found in every photo by an open-vocabulary detector: OWLv2 (Google, Apache-2.0, via Hugging Face transformers),
prompted with the names below. walltex projects the boxes onto the walls, so the paint can be one flat colour
and only these things are copied from the photos.

Plain switch plates are almost the colour of the paint, so colour alone cannot find them; a detector trained on
everyday objects can.
"""
from __future__ import annotations

import numpy as np

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
# Prompt -> lowest score kept. Small fixtures score lower than big, clear things.
WALL_ITEMS = {
    "light switch": 0.15, "electrical outlet": 0.18, "thermostat": 0.18, "wall clock": 0.15, "fire alarm": 0.2,
    "painting": 0.25, "framed picture": 0.25, "poster": 0.25, "paper notice": 0.2, "sign": 0.2,
    # Blank wall panels score up to ~0.4 as "whiteboard", window blinds as "projector screen"; real ones 0.6-0.9.
    "whiteboard": 0.5, "bulletin board": 0.4, "television": 0.4, "projector screen": 0.5,
}
ROUND = ("wall clock",)        # drawn as the ellipse inside their box (no paint corners from the photo)
BIG = ("painting", "framed picture", "poster", "whiteboard", "bulletin board", "television", "projector screen")
MAX_SMALL = 0.04   # a small fixture's box covers at most this share of the photo (else it is a false hit)
MAX_BIG = 0.6      # ... and a big thing's


def detect_wall_items(views, device, log=print, model_name=OWL_MODEL):
    """Per view (same order, same size as v["rgb"]): a bool mask of the pixels inside a detected wall item.
    Returns None when torch / transformers / the model are unavailable."""
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        try:   # the cached copy first: no network needed after the first run
            proc = Owlv2Processor.from_pretrained(model_name, local_files_only=True)
            model = Owlv2ForObjectDetection.from_pretrained(model_name, local_files_only=True)
        except OSError:
            log(f"[items] downloading {model_name} (about 600 MB, once)")
            proc = Owlv2Processor.from_pretrained(model_name)
            model = Owlv2ForObjectDetection.from_pretrained(model_name)
        model = model.to(device).eval()
    except Exception as ex:  # noqa: BLE001 - optional: the walls are then plain paint
        log(f"[items] detector unavailable ({ex}); walls get no items")
        return None
    prompts = list(WALL_ITEMS)
    lo = min(WALL_ITEMS.values())
    masks, counts = [], {}
    log(f"[items] {model_name} on {len(views)} photo(s), device={device}")
    for v in views:
        rgb = v["rgb"]
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), bool)
        with torch.no_grad():
            inp = proc(text=[prompts], images=Image.fromarray(rgb), return_tensors="pt").to(device)
            out = model(**inp)
        s = max(h, w)   # OWLv2 pads the photo to a square: boxes are in that square's pixels
        r = proc.post_process_grounded_object_detection(out, threshold=lo, target_sizes=[(s, s)],
                                                        text_labels=[prompts])[0]
        yy, xx = np.mgrid[0:h, 0:w]
        for box, score, name in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["text_labels"]):
            x0, y0, x1, y1 = np.clip(box, 0, [w, h, w, h])
            area = (x1 - x0) * (y1 - y0) / (h * w)
            if score < WALL_ITEMS[name] or area <= 0 or area > (MAX_BIG if name in BIG else MAX_SMALL):
                continue
            if name in ROUND:
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 1), max((y1 - y0) / 2, 1)
                mask |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
            else:
                mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
            counts[name] = counts.get(name, 0) + 1
        masks.append(mask)
    log("[items] detections: " + (", ".join(f"{n} {c}" for n, c in sorted(counts.items(), key=lambda t: -t[1]))
                                  or "none"))
    return masks


def wall_item_masks(opts, views, device, log=print):
    """Detector masks for "uniform" walls; None for other wall styles or without torch."""
    if opts.wall_detail != "uniform":
        return None
    try:
        from perception import pick_device
        device = device or pick_device(getattr(opts, "device", None))
    except Exception as ex:  # noqa: BLE001 - no torch
        log(f"[items] detector unavailable ({ex}); walls get no items")
        return None
    return detect_wall_items(views, device, log)

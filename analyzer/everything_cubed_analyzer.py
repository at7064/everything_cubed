"""everything_cubed analyzer: photos of a room -> plan.json for the Everything Cubed Fabric mod.

Usage:
  python everything_cubed_analyzer.py --images <folder> [--out <folder>/plan.json] [--mc-jar client.jar]
                                [--scale BLOCKS_PER_METRE] [--backend auto|arkit|mapanything|vggt]

A folder streamed from the iOS scanner app has a capture.json with LiDAR depth and camera poses;
`auto` then uses it directly (backend `arkit`) instead of reconstructing the room from the photos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from palette import surface_palette  # noqa: E402
from roomplan import ADE_TO_CATEGORY, build_plan  # noqa: E402


def log(msg):
    print(msg, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder with photos of one room")
    ap.add_argument("--out", type=Path, help="output plan (default: <images>/plan.json)")
    ap.add_argument("--mc-jar", help="Minecraft client jar, used to read real block texture colours")
    ap.add_argument("--scale", type=float, help="blocks per metre (default: automatic)")
    ap.add_argument("--backend", default="auto", choices=["auto", "arkit", "mapanything", "vggt"])
    ap.add_argument("--device", help="torch device (default: cuda > mps > cpu)")
    ap.add_argument("--mapanything-model", default="facebook/map-anything-apache")
    ap.add_argument("--vggt-model", default="facebook/VGGT-1B-Commercial")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
    ap.add_argument("--seg-model", default="nvidia/segformer-b5-finetuned-ade-640-640")
    ap.add_argument("--wall-detail", default="items", choices=["items", "photo", "none"],
                    help="items: plain walls + things hanging on them (default); photo: full photos; none: blocks only")
    ap.add_argument("--texture-px", type=int, help="texture pixels per block (default 64 for items, 128 for photo)")
    ap.add_argument("--save-cloud", action="store_true", help="also write the labelled point cloud (.npz)")
    opts = ap.parse_args(argv)

    from perception import category_lut, fuse, list_images, pick_device, reconstruct, segment

    t0 = time.time()
    paths = list_images(opts.images)
    device = pick_device(opts.device)
    log(f"[analyzer] {len(paths)} photo(s) from {opts.images}")
    views, metric = reconstruct(paths, opts.backend, device, opts, log)
    label_maps, id2label = segment(views, device, opts.seg_model, log)
    pts, rgb, cats = fuse(views, label_maps, id2label, ADE_TO_CATEGORY)
    log(f"[analyzer] fused cloud: {len(pts)} points")
    if opts.save_cloud:
        import numpy as np
        np.savez_compressed(opts.images / "cloud.npz", points=pts, colors=rgb, cats=cats)

    cache = Path(__file__).resolve().parent
    ctx = {}
    plan = build_plan(pts, rgb, cats, surface_palette(opts.mc_jar, cache), scale=opts.scale, metric=metric, log=log, ctx=ctx)
    plan["source"] = {"photos": [p.name for p in paths], "metric": metric}
    out = opts.out or opts.images / "plan.json"
    if opts.wall_detail != "none":
        try:
            from walltex import render_walls
            px = opts.texture_px or (64 if opts.wall_detail == "items" else 128)
            tex = render_walls(views, label_maps, category_lut(id2label, ADE_TO_CATEGORY), ctx,
                               out.parent, px=px, detail=opts.wall_detail, log=log)
            # Wall blocks follow the paint, so bare blocks and framed ones are the same colour.
            plan["walls"] = tex.pop("wall_blocks")
            plan["wall_main"] = tex.pop("wall_main")
            plan["textures"] = tex
        except Exception as ex:  # never lose the whole analysis over the wall pictures
            import traceback
            traceback.print_exc()
            log(f"[tex] wall pictures failed ({ex}); building with plain wall blocks")
    out.write_text(json.dumps(plan, indent=1))
    log(f"[analyzer] wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

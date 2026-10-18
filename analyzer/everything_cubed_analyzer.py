"""everything_cubed analyzer: photos of a room -> plan.json for the Everything Cubed Fabric mod.

Usage:
  python everything_cubed_analyzer.py --images <folder> [--out <folder>/plan.json] [--mc-jar client.jar]
                                [--scale BLOCKS_PER_METRE] [--backend auto|arkit|mapanything|vggt]

A folder streamed from the iOS scanner app has a capture.json with LiDAR depth and camera poses;
`auto` (or `arkit`) then hands it to lidar_analyzer.py, which needs no reconstruction or segmentation model.
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
from roomplan import ADE_TO_CATEGORY, TEMPLATE_INFO, apply_wall_features, build_plan  # noqa: E402


def log(msg):
    print(msg, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder with photos of one room")
    ap.add_argument("--out", type=Path, help="output plan (default: <images>/plan.json)")
    ap.add_argument("--mc-jar", help="Minecraft client jar, used to read real block texture colours")
    ap.add_argument("--scale", type=float, help="blocks per metre (default: automatic)")
    ap.add_argument("--enlarge", type=float, default=1.25,
                    help="build photo rooms this much larger than measured, e.g. 1.25 = 25%% (default); "
                         "LiDAR scans and --scale are always built at their measured size")
    ap.add_argument("--known-height", type=float,
                    help="the room's real floor-to-ceiling height in metres; corrects the reconstruction's scale")
    ap.add_argument("--backend", default="auto", choices=["auto", "arkit", "mapanything", "vggt"])
    ap.add_argument("--device", help="torch device (default: cuda > mps > cpu)")
    ap.add_argument("--batch-views", type=int, default=6,
                    help="MapAnything views per overlapping batch (default 6; lower uses less GPU memory)")
    ap.add_argument("--model-resolution", type=int, default=518,
                    help="MapAnything longest image side in pixels (default 518; 728 needs more GPU memory)")
    # Apache variant: on a taped 9.45 x 4.57 m room it measured 0.91x / 1.10x, the research variant
    # (facebook/map-anything, CC-BY-NC 4.0) 1.22x / 1.25x; the research variant was closer only on a 20 m high hall.
    ap.add_argument("--mapanything-model", default="facebook/map-anything-apache")
    ap.add_argument("--tracked-batch", type=int, default=0,
                    help="phone photo scans: MapAnything photos per batch, each batch placed with ARKit's poses "
                         "(0 = all photos in one run, the default; with batches --max-views can be raised)")
    ap.add_argument("--max-views", type=int, default=24,
                    help="photos of an iOS photo scan given to MapAnything at once (default 24; more needs more memory)")
    ap.add_argument("--no-exif-camera", action="store_true",
                    help="do not give MapAnything the camera geometry from the photos' lens data")
    ap.add_argument("--vggt-model", default="facebook/VGGT-1B-Commercial")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
    ap.add_argument("--seg-model", default="nvidia/segformer-b5-finetuned-ade-640-640")
    ap.add_argument("--wall-detail", default="uniform", choices=["uniform", "style", "pattern", "items", "photo", "none"],
                    help="uniform (default): every wall block is a map in the wall's measured paint colour, with the "
                         "things on the wall (posters, clocks, switches, boards) copied from the photos on top; "
                         "style: one wall style for the whole room on every wall; "
                         "pattern: plain walls are blocks, patterned walls get their own pattern on every block; "
                         "items: plain walls + things hanging on them; photo: full photos; none: blocks only")
    ap.add_argument("--texture-px", type=int, help="texture pixels per block (default 128 for photo, else 64)")
    ap.add_argument("--save-cloud", action="store_true", help="also write the labelled point cloud (.npz)")
    ap.add_argument("--mesh-out", type=Path, help="write reconstructed coloured surface mesh as PLY")
    ap.add_argument("--mesh-max-side", type=int, default=0,
                    help="downsample each mesh view to this side length; 0 keeps every reconstructed pixel")
    ap.add_argument("--upright-video", action="store_true",
                    help="keep video reconstruction batches upright and at metric scale")
    ap.add_argument("--low-gpu-memory", action="store_true",
                    help="offload MapAnything sharing layers to CPU to reduce GPU memory use")
    ap.add_argument("--perceptron", action="store_true",
                    help="also detect every object with Perceptron Mk1.5 (sends the photos to api.perceptron.inc; people "
                         "found locally (SegFormer + a COCO person detector) are blurred first, best effort; key in "
                         "$PERCEPTRON_API_KEY). Off by default: photos stay local")
    ap.add_argument("--perceptron-reasoning", default="none", choices=["none", "low", "medium", "high"],
                    help="none is about 3x faster than high and still finds the furniture (default none)")
    ap.add_argument("--perceptron-keep-people", action="store_true", help="do not blur people before upload")
    ap.add_argument("--mirror-walls", action="store_true",
                    help="a wall the photos hardly saw copies the opposite wall (halls, libraries, churches)")
    ap.add_argument("--no-depth-gate", action="store_true",
                    help="paint whole detection boxes, not only the pixels near the object's depth")
    opts = ap.parse_args(argv)
    tracked = False
    if opts.backend in ("auto", "arkit") and (opts.images / "capture.json").is_file():
        from perception import capture_has_depth
        if capture_has_depth(opts.images):   # a LiDAR scan
            import lidar_analyzer
            return lidar_analyzer.main(sys.argv[1:] if argv is None else argv)
        tracked = True                       # photos from the iOS app with ARKit camera positions, no depth

    from perception import category_lut, fuse, list_images, pick_device, reconstruct, segment

    t0 = time.time()
    paths = list_images(opts.images)
    device = pick_device(opts.device)
    log(f"[analyzer] {len(paths)} photo(s) from {opts.images}")
    planes = None
    if tracked:
        from perception import load_planes
        planes = load_planes(opts.images) or None   # walls / floor / ceiling ARKit measured, when the app sent them
    # ARKit measured where each photo was taken (camera + motion sensors): built at measured size like LiDAR.
    enlarge = 1.0 if tracked else opts.enlarge
    if tracked:
        import atexit
        import shutil
        import tempfile
        from perception import reconstruct_tracked
        # The upright copies are only needed while analysing: never left in the scan folder.
        work = tempfile.mkdtemp(prefix="everything_cubed_upright_")
        atexit.register(shutil.rmtree, work, ignore_errors=True)
        views, metric, paths = reconstruct_tracked(opts.images, device, opts.mapanything_model, log, opts.max_views,
                                                   work_dir=work, batch_views=opts.tracked_batch)
    else:
        views, metric = reconstruct(paths, opts.backend, device, opts, log)
    label_maps, id2label = segment(views, device, opts.seg_model, log, paths=paths)
    if opts.upright_video:
        from perception import stabilize_upright_video
        stabilize_upright_video(views, label_maps, id2label, log)
    if opts.mesh_out:
        from mesh_export import write_ply
        vertices, faces = write_ply(views, opts.mesh_out, max_side=opts.mesh_max_side)
        log(f"[mesh] wrote {opts.mesh_out}: {vertices} vertices, {faces} triangles")
    ade_to_category, detections = ADE_TO_CATEGORY, None
    if opts.perceptron:
        from perception import perceptron_pass
        from vanilla import catalog
        label_maps, id2label, ade_to_category, detections = perceptron_pass(
            paths, views, label_maps, id2label, ADE_TO_CATEGORY, TEMPLATE_INFO, log, reasoning=opts.perceptron_reasoning,
            blur=not opts.perceptron_keep_people, depth_gate=metric and not opts.no_depth_gate,  # gate is in metres
            device=device, cache_dir=opts.images / ".perceptron_cache",  # answers per exact upload
            vanilla_cat=catalog(opts.mc_jar, Path(__file__).resolve().parent, log))
    pts, rgb, cats = fuse(views, label_maps, id2label, ade_to_category)
    log(f"[analyzer] fused cloud: {len(pts)} points")
    if opts.save_cloud:
        import numpy as np
        np.savez_compressed(opts.images / "cloud.npz", points=pts, colors=rgb, cats=cats)

    cache = Path(__file__).resolve().parent
    ctx = {}
    object_instances = detections.pop("_instances", []) if detections else []
    plan = build_plan(pts, rgb, cats, surface_palette(opts.mc_jar, cache), scale=opts.scale, metric=metric, log=log, ctx=ctx,
                      known_height=opts.known_height, object_instances=object_instances, enlarge=enlarge,
                      planes=planes)
    plan["source"] = {"photos": [p.name for p in paths], "metric": metric, "tracked": tracked,
                      "enlarge": None if opts.scale else enlarge}
    out = opts.out or opts.images / "plan.json"
    if detections:
        # Every detected name and what it became, so missing templates are easy to spot.
        plan["detections"] = detections["names"]
        (out.parent / "detections.json").write_text(json.dumps(detections, indent=1))
    if opts.wall_detail != "none":
        try:
            from walltex import render_walls
            from wallitems import wall_item_masks
            px = opts.texture_px or (128 if opts.wall_detail == "photo" else 64)
            from perception import detection_masks
            # Detected windows and doors are never wall paint or pattern, even where SegFormer says wall
            # (big stained-glass windows); their boxes do not change the room's geometry, only the textures.
            tex = render_walls(views, label_maps, category_lut(id2label, ade_to_category), ctx,
                               out.parent, px=px, detail=opts.wall_detail, log=log,
                               exclude=detection_masks(detections, views),
                               item_masks=wall_item_masks(opts, views, device, log))
            # Wall blocks follow the paint, so bare blocks and framed ones are the same colour.
            plan["walls"] = tex.pop("wall_blocks")
            plan["wall_main"] = tex.pop("wall_main")
            plan["textures"] = tex
            apply_wall_features(plan, tex.pop("features"), log, mirror=opts.mirror_walls, tex_dir=out.parent)
        except Exception as ex:  # never lose the whole analysis over the wall pictures
            import traceback
            traceback.print_exc()
            log(f"[tex] wall pictures failed ({ex}); building with plain wall blocks")
    out.write_text(json.dumps(plan, indent=1))
    from plan_viewer import write_viewer
    write_viewer(plan, out.with_name("view.html"), opts.mesh_out)
    log(f"[analyzer] wrote {out} in {time.time() - t0:.0f}s")
    log(f"[analyzer] 3D viewer: {out.with_name('view.html')}")


if __name__ == "__main__":
    main()

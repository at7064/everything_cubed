"""Video or LiDAR capture -> Everything Cubed plan, standalone HTML viewer and OBJ mesh.

Examples:
  python analyzer/video_to_3d.py room.mp4 --out room_model
  python analyzer/video_to_3d.py scan_folder --out lidar_model

The scan folder must contain capture.json and its images/depth files. Plain video is sampled into
individual photos and reconstructed with the existing MapAnything analyzer.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from plan_viewer import scene_from_plan, write_obj, write_viewer


VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def extract_frames(video: Path, folder: Path, count: int, max_width: int, ffmpeg="ffmpeg"):
    if not shutil.which(ffmpeg):
        raise RuntimeError("ffmpeg is required for video input")
    info = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info.stderr)
    if not match:
        raise ValueError(f"could not read video duration: {video}")
    hours, minutes, seconds_part = match.groups()
    seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_part)
    if seconds <= 0:
        raise ValueError("video duration must be positive")
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("frame_*.jpg"):
        old.unlink()
    # Sample evenly across the entire clip; bounded frame count keeps reconstruction tractable.
    fps = count / seconds
    filt = f"fps={fps:.9f},scale='min(iw,{max_width})':-2"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", filt, "-frames:v", str(count), "-q:v", "3", str(folder / "frame_%04d.jpg")], check=True)
    images = sorted(folder.glob("frame_*.jpg"))
    if len(images) < 2:
        raise RuntimeError("fewer than two video frames were extracted")
    return images


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="video file, photos folder, or LiDAR capture folder")
    ap.add_argument("--out", type=Path, help="output directory (default: <input stem>_3d)")
    ap.add_argument("--frames", type=int, default=16, help="frames sampled across a video (default 16)")
    ap.add_argument("--max-width", type=int, default=1600, help="maximum extracted frame width")
    ap.add_argument("--batch-views", type=int, default=6, help="MapAnything views per GPU batch (default 6)")
    ap.add_argument("--model-resolution", type=int, default=518, help="reconstruction image size (default 518; try 728 for detail)")
    ap.add_argument("--mesh-max-side", type=int, default=0, help="mesh downsampling limit; 0 retains all pixels")
    ap.add_argument("--splat-radius", type=float, default=0.12,
                    help="radius in metres used to join nearby reconstructed surfaces (default 0.12)")
    ap.add_argument("--photo-faces", type=int, default=6000,
                    help="textured triangles per frame in photo_view.html (default 6000)")
    ap.add_argument("--backend", choices=("auto", "arkit", "mapanything", "vggt"), default="auto")
    ap.add_argument("--device", help="torch device, e.g. cuda")
    ap.add_argument("--mc-jar", type=Path, help="Minecraft jar for block colour matching")
    ap.add_argument("--scale", type=float, help="blocks per metre")
    ap.add_argument("--perceptron", action="store_true", help="also detect small objects via the Perceptron API")
    ap.add_argument("--wall-detail", default="uniform", choices=("uniform", "style", "pattern", "items", "photo", "none"))
    opts = ap.parse_args(argv)
    source = opts.input.expanduser().resolve()
    if not source.exists():
        ap.error(f"input does not exist: {source}")
    if opts.frames < 2 or opts.frames > 64:
        ap.error("--frames must be between 2 and 64")
    if opts.max_width < 320:
        ap.error("--max-width must be at least 320")
    if opts.batch_views < 3:
        ap.error("--batch-views must be at least 3")
    if opts.model_resolution < 224 or opts.model_resolution > 1024:
        ap.error("--model-resolution must be between 224 and 1024")
    if not 0.02 <= opts.splat_radius <= 0.5:
        ap.error("--splat-radius must be between 0.02 and 0.5 metres")
    if not 100 <= opts.photo_faces <= 50000:
        ap.error("--photo-faces must be between 100 and 50000")
    out = (opts.out or source.with_name(source.stem + "_3d")).expanduser().resolve()
    if source.is_file() and source.suffix.lower() not in VIDEO_EXT:
        ap.error("input file must be a video; use a folder for photos or LiDAR captures")
    if source.is_dir() and out == source:
        ap.error("output directory must differ from the input folder")
    out.mkdir(parents=True, exist_ok=True)
    images = out / "frames" if source.is_file() else source
    if source.is_file():
        extracted = extract_frames(source, images, opts.frames, opts.max_width)
        print(f"[video] extracted {len(extracted)} frames from {source}", flush=True)

    from everything_cubed_analyzer import main as analyze
    plan_path = out / "plan.json"
    args = ["--images", str(images), "--out", str(plan_path), "--mesh-out", str(out / "reconstruction.ply"),
            "--batch-views", str(opts.batch_views), "--model-resolution", str(opts.model_resolution),
            "--mesh-max-side", str(opts.mesh_max_side), "--backend", opts.backend,
            "--wall-detail", opts.wall_detail]
    if opts.device: args += ["--device", opts.device]
    if opts.mc_jar: args += ["--mc-jar", str(opts.mc_jar)]
    if opts.scale is not None: args += ["--scale", str(opts.scale)]
    if opts.perceptron: args += ["--perceptron"]
    analyze(args)
    if not plan_path.is_file():
        raise RuntimeError("analyzer did not write a plan")
    plan = json.loads(plan_path.read_text())
    if source.is_file():
        plan.setdefault("source", {})["video"] = str(source)
        plan_path.write_text(json.dumps(plan, indent=1))
    from solid_export import write_solid
    nverts, nfaces = write_solid(out / "reconstruction.ply", out / "solid.ply", splat_radius=opts.splat_radius)
    print(f"[solid] wrote {out / 'solid.ply'}: {nverts} vertices, {nfaces} triangles", flush=True)
    photo_view = False
    if not (images / "capture.json").exists():
        from perception import list_images
        from photo_texture import write_photo_viewer
        write_photo_viewer(out / "reconstruction.ply", list_images(images), out / "photo_view.html",
                           max_faces_per_frame=opts.photo_faces)
        photo_view = True
    scene = write_viewer(plan, out / "view.html", out / "solid.ply", photo_view=photo_view)
    write_obj(scene, out / "model.obj")
    print(f"[3d] {out / 'reconstruction.ply'}\n[3d] {out / 'solid.ply'}\n[3d] {out / 'view.html'}\n"
          + (f"[3d] {out / 'photo_view.html'}\n" if photo_view else "") +
          f"[3d] {out / 'model.obj'}\n[3d] {plan_path}", flush=True)


if __name__ == "__main__":
    main()

"""Add the untouched reconstructed mesh beneath an existing photo-textured view.

Run ``python original_mesh_overlay.py scan_output`` to create
``scan_output/photo_on_original.html`` from its existing PLY and photo viewer.
The video modeller calls this automatically after writing ``photo_view.html``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from photo_texture import PHOTO_HTML
from plan_viewer import mesh_preview_from_ply


def generate(folder: Path, max_mesh_faces: int = 150_000) -> Path:
    folder = Path(folder)
    ply = folder / "reconstruction.ply"
    photo_html = folder / "photo_view.html"
    output = folder / "photo_on_original.html"
    if not ply.is_file() or not photo_html.is_file():
        raise FileNotFoundError(f"{folder} needs reconstruction.ply and photo_view.html")

    source = photo_html.read_text(encoding="utf-8")
    marker = "const scene="
    if marker not in source:
        raise ValueError(f"{photo_html} is not a photo-textured viewer")
    scene, _ = json.JSONDecoder().raw_decode(source.split(marker, 1)[1])
    if not scene.get("batches"):
        raise ValueError(f"{photo_html} has no photo frames")
    scene["base_mesh"] = mesh_preview_from_ply(ply, max_faces=max_mesh_faces)
    encoded = json.dumps(scene, separators=(",", ":")).replace("<", "\\u003c")
    output.write_text(PHOTO_HTML.replace("__SCENE__", encoded), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="video-to-3D output directory")
    parser.add_argument("--mesh-faces", type=int, default=150_000,
                        help="maximum original-mesh faces embedded in the viewer")
    args = parser.parse_args()
    if args.mesh_faces < 1:
        parser.error("--mesh-faces must be positive")
    if not args.folder.is_dir():
        parser.error(f"output directory does not exist: {args.folder}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(args.folder / "original_mesh_overlay.log"),
                  logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Wrote %s", generate(args.folder, max_mesh_faces=args.mesh_faces))
    for handler in logging.getLogger().handlers:
        handler.flush()


if __name__ == "__main__":
    main()

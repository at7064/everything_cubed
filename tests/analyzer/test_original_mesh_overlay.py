"""The automatic overlay reuses photo patches and leaves the source mesh intact."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analyzer"))

from mesh_export import write_ply
from original_mesh_overlay import generate
from photo_texture import write_photo_viewer


def read_scene(path):
    return json.JSONDecoder().raw_decode(path.read_text().split("const scene=", 1)[1])[0]


def test_overlay_reuses_existing_photo_view(tmp_path):
    x, y = np.meshgrid(np.arange(5) * 0.1, np.arange(4) * 0.1)
    rgb = np.full((4, 5, 3), [120, 170, 210], dtype=np.uint8)
    view = {"points": np.dstack([x, y, np.ones_like(x)]).astype(np.float32),
            "rgb": rgb, "mask": np.ones((4, 5), bool)}
    ply = tmp_path / "reconstruction.ply"
    write_ply([view], ply, max_side=20)
    photo = tmp_path / "frame_0001.jpg"
    Image.fromarray(rgb).save(photo)
    source = tmp_path / "photo_view.html"
    write_photo_viewer(ply, [photo], source, max_faces_per_frame=100)
    original_mesh_bytes = ply.read_bytes()

    output = generate(tmp_path, max_mesh_faces=100)
    source_scene, overlay_scene = read_scene(source), read_scene(output)
    assert output.name == "photo_on_original.html"
    assert overlay_scene["batches"] == source_scene["batches"]
    assert overlay_scene["flip_xy"] == source_scene["flip_xy"]
    assert overlay_scene["base_mesh"]["preview_faces"] > 0
    assert ply.read_bytes() == original_mesh_bytes

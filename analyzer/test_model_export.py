"""Checks that reconstructed surfaces and room plans produce usable 3D artifacts."""
import json
import tempfile
from pathlib import Path

import numpy as np

from mesh_export import mesh_arrays, write_ply
from plan_viewer import write_obj, write_viewer


def test_export():
    x, y = np.meshgrid(np.arange(5) * 0.1, np.arange(4) * 0.1)
    points = np.dstack([x, y, np.ones_like(x)]).astype(np.float32)
    view = {"points": points, "rgb": np.full((4, 5, 3), 180, np.uint8),
            "mask": np.ones((4, 5), bool)}
    vertices, faces = mesh_arrays([view], max_side=20)
    assert len(vertices) == 20 and len(faces) == 24
    view["mask"][1, 1] = False
    vertices, faces = mesh_arrays([view], max_side=20)
    assert len(vertices) < 20 and len(faces) < 24
    plan = {"size": {"x": 3, "y": 3, "z": 2}, "floor": {"grid": [[0]*3 for _ in range(2)],
            "palette": ["minecraft:oak_planks"]}, "walls": {}, "wall_main": "minecraft:white_concrete",
            "openings": [{"type": "door", "side": "north", "u": 1, "w": 1, "y": 0, "h": 2},
                         {"type": "fire_alarm", "side": "west", "u": 1, "w": 1, "y": 1, "h": 1}],
            "objects": [{"type": "desk", "x": 1, "z": 1, "w": 1, "d": 1, "h": 1}]}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ply, html, obj = root / "reconstruction.ply", root / "view.html", root / "model.obj"
        write_ply([view], ply, max_side=20)
        scene = write_viewer(plan, html, ply)
        write_obj(scene, obj)
        assert len(scene["mesh"]["triangles"]) == len(faces)
        assert any(b[8] == "Fire Alarm" for b in scene["boxes"])
        assert all(p.is_file() for p in (ply, html, obj, obj.with_suffix(".mtl")))
        assert "Reconstructed surfaces" in html.read_text()
        assert obj.read_text().startswith("# Everything Cubed room")
        json.dumps(scene)


if __name__ == "__main__":
    test_export()
    print("model export OK")

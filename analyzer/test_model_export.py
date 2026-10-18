"""Checks that reconstructed surfaces and room plans produce usable 3D artifacts."""
import json
import tempfile
from pathlib import Path

import numpy as np

from mesh_export import mesh_arrays, write_ply
from photo_texture import align_wall_patches, photo_scene, write_photo_viewer
from plan_viewer import scene_from_plan, write_obj, write_viewer
from solid_export import write_solid


def test_export():
    x, y = np.meshgrid(np.arange(5) * 0.1, np.arange(4) * 0.1)
    points = np.dstack([x, y, np.ones_like(x)]).astype(np.float32)
    view = {"points": points, "rgb": np.full((4, 5, 3), 180, np.uint8),
            "mask": np.ones((4, 5), bool)}
    vertices, faces = mesh_arrays([view], max_side=20)
    assert len(vertices) == 20 and len(faces) == 24
    view["mask"][1, 1] = False
    view["rgb"][0, 0] = [255, 0, 0]
    vertices, faces = mesh_arrays([view], max_side=20)
    assert len(vertices) < 20 and len(faces) < 24
    view["mask"][1, 1] = True
    vertices, faces = mesh_arrays([view], max_side=20)
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
        assert scene["mesh"]["preview_faces"] == len(faces)
        import base64
        packed = np.frombuffer(base64.b64decode(scene["mesh"]["vertices_b64"]),
                               dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                               ("r", "u1"), ("g", "u1"), ("b", "u1"), ("pad", "u1")]))
        assert len(packed) == len(faces) * 3
        assert np.any((packed["r"] == 255) & (packed["g"] == 0))
        from PIL import Image
        photo = root / "frame_0001.jpg"
        Image.fromarray(view["rgb"]).save(photo)
        textured = photo_scene(ply, [photo], max_faces_per_frame=100)
        assert textured["batches"][0]["grid"] == [5, 4]
        assert 0 < textured["total_faces"] <= len(faces)
        assert base64.b64decode(textured["batches"][0]["coarse_b64"])
        photo_vertices = np.frombuffer(base64.b64decode(textured["batches"][0]["vertices_b64"]),
                                       dtype="<f4").reshape(-1, 5)
        assert np.all((photo_vertices[:, 3:] > 0) & (photo_vertices[:, 3:] < 1))
        write_photo_viewer(ply, [photo], root / "photo_view.html", max_faces_per_frame=100)
        assert "Original video pixels" in (root / "photo_view.html").read_text()
        shell = {"origin": [0, 0, 0], "rotation": np.eye(3).tolist(),
                 "y_shift": 0, "scale": 1,
                 "bounds": {"xmin": -1, "xmax": 100, "zmin": -1, "zmax": 100,
                            "ymin": -1, "ymax": 100}}
        clipped = photo_scene(ply, [photo], max_faces_per_frame=100, shell=shell)
        assert clipped["total_faces"] == textured["total_faces"]
        shell["bounds"]["xmax"] = -0.5
        outside = photo_scene(ply, [photo], max_faces_per_frame=100, shell=shell)
        assert outside["total_faces"] == textured["total_faces"]
        shell["bounds"]["xmax"] = 100
        shell["bounds"]["xmin"] = 0
        patch = np.array([[0.07, 1, 2, 0.5, 0.5]], dtype="<f4")
        aligned = align_wall_patches(patch, shell)
        assert aligned[0, 0] == 0 and np.array_equal(aligned[0, 3:], patch[0, 3:])
        assert any(b[8] == "Fire Alarm" for b in scene["boxes"])
        assert all(p.is_file() for p in (ply, html, obj, obj.with_suffix(".mtl")))
        assert "Reconstructed surfaces" in html.read_text()
        assert obj.read_text().startswith("# Everything Cubed room")
        json.dumps(scene)
        solid = root / "solid.ply"
        count, triangles = write_solid(ply, solid, voxel=0.03, splat_radius=0.09)
        assert count > 0 and triangles > 0 and solid.is_file()
        plan["footprint"] = [[1, 1, 1], [1, 0, 0]]
        shaped = scene_from_plan(plan)
        assert sum(box[7] == "floor" for box in shaped["boxes"]) == 4
        assert any(box[7] == "wall" and box[0] == 1 for box in shaped["boxes"])


if __name__ == "__main__":
    test_export()
    print("model export OK")

"""Bridge small gaps in reconstructed surfaces with overlapping 3D splats."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from mesh_export import VERTEX, FACE


def write_solid(source, target, voxel=0.03, splat_radius=0.12, max_cells=400):
    """Make a soft, colored surface from real vertices without imposing a box room."""
    from skimage.measure import marching_cubes

    source, target = Path(source), Path(target)
    with source.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 4096:
                raise ValueError("invalid PLY header")
            header += line
    text = header.decode("ascii")
    count = int(next(s.split()[2] for s in text.splitlines() if s.startswith("element vertex ")))
    vertices = np.memmap(source, dtype=VERTEX, mode="r", offset=len(header), shape=(count,))
    # The grid is bounded even for noisy depth estimates or large halls.
    stride = max(1, count // 1_500_000)
    sample = vertices[::stride]
    points = np.stack([sample[k] for k in ("x", "y", "z")], axis=1)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = np.stack([sample[k] for k in ("red", "green", "blue")], axis=1)[finite]
    lo, hi = np.percentile(points, [0.2, 99.8], axis=0)
    voxel = max(float(voxel), float(np.max((hi - lo + 4 * splat_radius) / max_cells)))
    origin = lo - 2 * splat_radius
    dims = np.ceil((hi - origin + 2 * splat_radius) / voxel).astype(int) + 1
    field = np.zeros(tuple(dims), dtype=bool)
    cells = np.floor((points - origin) / voxel).astype(int)
    good = np.all((cells >= 0) & (cells < dims), axis=1)
    cells = cells[good]
    field[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    radius = max(1, round(splat_radius / voxel))
    axis = np.arange(-radius, radius + 1)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    ball = xx * xx + yy * yy + zz * zz <= radius * radius
    field = ndimage.binary_dilation(field, structure=ball)
    smooth = ndimage.gaussian_filter(field.astype(np.float32), sigma=1.0)
    xyz, faces, _, _ = marching_cubes(smooth, level=0.35, spacing=(voxel,) * 3)
    xyz += origin
    tree = cKDTree(points[good], leafsize=32)
    distance, nearest = tree.query(xyz, k=4, workers=-1)
    weights = 1.0 / np.maximum(distance, voxel * 0.25)
    weights /= weights.sum(axis=1, keepdims=True)
    rgb = np.rint((colors[good][nearest] * weights[..., None]).sum(axis=1)).astype(np.uint8)
    out_vertices = np.empty(len(xyz), VERTEX)
    for i, key in enumerate(("x", "y", "z")):
        out_vertices[key] = xyz[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        out_vertices[key] = rgb[:, i]
    out_faces = np.empty(len(faces), FACE)
    out_faces["count"] = 3
    for i, key in enumerate(("a", "b", "c")):
        out_faces[key] = faces[:, i]
    header_out = ("ply\nformat binary_little_endian 1.0\ncomment Everything Cubed gap-filled surface\n"
                  f"element vertex {len(out_vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(out_faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        stream.write(header_out.encode("ascii"))
        out_vertices.tofile(stream)
        out_faces.tofile(stream)
    return len(out_vertices), len(out_faces)

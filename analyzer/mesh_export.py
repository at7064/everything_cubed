"""Turn organised reconstructed depth views into a coloured triangle mesh (binary PLY)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
FACE = np.dtype([("count", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])


def mesh_arrays(views, max_side=180):
    """Triangulate neighbouring depth pixels; reject triangles crossing depth discontinuities."""
    vertex_parts, face_parts = [], []
    offset = 0
    for view in views:
        points = np.asarray(view["points"])
        rgb = np.asarray(view["rgb"])
        if points.ndim != 3 or points.shape[-1] != 3 or rgb.shape != points.shape:
            raise ValueError("each view needs aligned H x W x 3 points and RGB")
        h0, w0 = points.shape[:2]
        stride = max(1, int(np.ceil(max(h0, w0) / max_side)))
        p = np.ascontiguousarray(points[::stride, ::stride], dtype=np.float32)
        c = np.ascontiguousarray(rgb[::stride, ::stride], dtype=np.uint8)
        h, w = p.shape[:2]
        valid = np.isfinite(p).all(-1) & (np.linalg.norm(p, axis=-1) > 1e-6)
        if "mask" in view:
            valid &= np.asarray(view["mask"])[::stride, ::stride].astype(bool)
        if "conf" in view:
            conf = np.asarray(view["conf"])[::stride, ::stride]
            valid &= np.isfinite(conf) & (conf > 0)
        vertices = np.zeros(h * w, VERTEX)
        vertices["x"], vertices["y"], vertices["z"] = p[..., 0].ravel(), p[..., 1].ravel(), p[..., 2].ravel()
        vertices["red"], vertices["green"], vertices["blue"] = c[..., 0].ravel(), c[..., 1].ravel(), c[..., 2].ravel()
        vertex_parts.append(vertices)
        if h < 2 or w < 2:
            offset += h * w
            continue
        # A local threshold follows the view's sampling density, with a conservative absolute cap.
        edges = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)
        good_edges = edges[valid[:, 1:] & valid[:, :-1] & np.isfinite(edges)]
        typical = float(np.median(good_edges)) if len(good_edges) else 0.02
        limit = min(0.5, max(0.05, 4 * typical + 0.02))
        ids = np.arange(h * w, dtype=np.int32).reshape(h, w) + offset
        triangles = []
        for indices in ((ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1]),
                        (ids[1:, 1:], ids[1:, :-1], ids[:-1, 1:])):
            a, b, d = indices
            aa, bb, dd = a - offset, b - offset, d - offset
            av, bv, dv = valid.ravel()[aa], valid.ravel()[bb], valid.ravel()[dd]
            pa, pb, pd = p.reshape(-1, 3)[aa], p.reshape(-1, 3)[bb], p.reshape(-1, 3)[dd]
            keep = av & bv & dv & (np.linalg.norm(pa-pb, axis=-1) < limit) & (
                np.linalg.norm(pa-pd, axis=-1) < limit) & (np.linalg.norm(pb-pd, axis=-1) < limit)
            triangles.append(np.stack((a[keep], b[keep], d[keep]), axis=1))
        tri = np.concatenate(triangles) if triangles else np.empty((0, 3), np.int32)
        faces = np.zeros(len(tri), FACE)
        faces["count"] = 3
        if len(tri):
            faces["a"], faces["b"], faces["c"] = tri.T
        face_parts.append(faces)
        offset += h * w
    if not vertex_parts:
        raise ValueError("no reconstructed views")
    vertices = np.concatenate(vertex_parts)
    faces = np.concatenate(face_parts) if face_parts else np.empty(0, FACE)
    if not len(faces):
        raise ValueError("no continuous surfaces found in reconstructed views")
    used = np.unique(np.concatenate((faces["a"], faces["b"], faces["c"])))
    remap = np.full(len(vertices), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    for key in ("a", "b", "c"):
        faces[key] = remap[faces[key]]
    return vertices[used], faces


def write_ply(views, path: Path, max_side=180):
    vertices, faces = mesh_arrays(views, max_side=max_side)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat binary_little_endian 1.0\ncomment Everything Cubed reconstructed surfaces\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
        faces.tofile(stream)
    return len(vertices), len(faces)

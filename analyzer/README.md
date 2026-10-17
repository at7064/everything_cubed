# Everything Cubed analyzer

Turns a handful of photos of one room into `plan.json`, which the Everything Cubed mod builds.
The mod runs this automatically; you only need to install the Python environment once.

## Pipeline (all established open-source projects)

| Step | Project |
|---|---|
| 3D reconstruction from unposed photos, in metres | [MapAnything](https://github.com/facebookresearch/map-anything) (`facebook/map-anything-apache`) |
| Fallback reconstruction | [VGGT](https://github.com/facebookresearch/vggt) + [Depth Anything V2 Metric Indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf) for scale |
| What each pixel is (wall, floor, bed, window, door, sofa, ...) | [SegFormer-B5 fine-tuned on ADE20K](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) via Hugging Face `transformers` |
| Plane fitting, clustering, colour quantisation | scikit-learn (RANSAC, DBSCAN, KMeans), SciPy |

Then `roomplan.py`:
1. Fits the floor plane (gravity) and the dominant wall direction, so the room is axis-aligned.
2. Measures each wall. Walls that never appear in a photo are inferred from the floor and furniture extents,
   and get the room's dominant wall material.
3. Picks a scale (1.25 blocks per metre, enlarged so the short side is at least 6.25 blocks, max 2.5 blocks/m)
   unless you pass one.
4. Colours each wall/floor cell from the photos. Up to 3 distinct materials are kept per surface, so you don't
   get a blotchy result. They are matched in CIELAB to the real average texture colours read from your
   Minecraft jar (concrete, terracotta, wool, planks, stone, ...).
5. Clusters each object class in 3D and converts it to real, usable blocks at the right size and position:
   beds (colour-matched, side by side for wide beds), doors, glass-pane windows, stair chairs and sofas, slab tables,
   barrels and cabinets, chiseled bookshelves, shelves, furnace/smoker, cauldron sinks, carpets for rugs,
   lanterns, potted plants and paintings.
6. Walls (`walltex.py`): every wall is resampled straight from the photos with the reconstructed cameras.
   Each pixel comes from the most head-on photo that shows the wall surface itself, not furniture in front
   or things beyond it. The paint is found jointly over all walls. A second paint is only used for a big,
   clearly different area, like an accent wall. Each paint snaps to one of 33 wall blocks (the 17
   terracottas and 16 concretes), and the walls are built from those real blocks, so they are continuous.
   Unseen walls get the same block. Things on the wall (posters, boards, clocks, ...) are drawn on
   locked maps in invisible item frames: snapped to whole blocks, in a few flat colours, filling their frames
   completely. Frames are lit per entity and can never blend with smoothly lit blocks, so they are not used
   for plain paint. Players can punch the frames off and break the blocks. `--wall-detail photo` puts the
   raw photo on every block, and `--wall-detail none` uses blocks only. Extra flags go in `extraArgs` in
   `config/everything_cubed.json`.
7. Guarantees a front door on a free stretch of wall, and keeps the floor in front of every door clear.

## Install (once)

Run these from the repository root, one at a time. A venv at `<game dir>/everything_cubed/venv` is picked up automatically.

```bash
python3 -m venv ~/.minecraft/everything_cubed/venv
```

```bash
~/.minecraft/everything_cubed/venv/bin/pip install -r analyzer/requirements.txt
```

Python 3.12 is the most tested with PyTorch and MapAnything. If a package fails to install on a newer Python,
recreate the venv with `python3.12`.

If you put Python somewhere else, set `"python"` in `config/everything_cubed.json`. Extra CLI flags (for example
`["--backend", "vggt"]` or `["--device", "cpu"]`) go in `"extraArgs"`.
The first run downloads the models from Hugging Face (several GB). A CUDA GPU is strongly recommended.
CPU and Apple MPS work, but they are slow.

## Run by hand

```bash
python everything_cubed_analyzer.py --images ~/rooms/bedroom --mc-jar path/to/minecraft-client.jar [--scale 1.5]
```

`python test_synthetic.py` and `python test_textures.py` check the geometry and wall-texture stages on
synthetic rooms without any ML models.

## Photo tips

Take 4–12 overlapping photos from the doorway and the corners, and include the floor and the ceiling.
Wider shots work better than close-ups.

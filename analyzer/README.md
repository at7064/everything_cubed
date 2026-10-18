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
   lanterns, chandeliers hung from the ceiling, potted plants and paintings. Things with no template yet (from
   `--perceptron`) become a plain box named after what was detected.
6. Walls (`walltex.py`): every wall is resampled straight from the photos with the reconstructed cameras.
   Each pixel comes from the most head-on photo that shows the wall surface itself, not furniture in front
   or things beyond it. The paint is found jointly over all walls. A second paint is only used for a big,
   clearly different area, like an accent wall. Each paint snaps to one of 33 wall blocks (the 17
   terracottas and 16 concretes), and the walls are built from those real blocks, so they are continuous.
   Unseen walls get the same block. Things on the wall (posters, boards, clocks, ...) are drawn on
   locked maps in invisible item frames: snapped to whole blocks, in a few flat colours, filling their frames
   completely. Frames are lit per entity and can never blend with smoothly lit blocks, so they are not used
   for plain paint. Players can punch the frames off and break the blocks.

   **Room style** (`--wall-detail style`, the default): one wall style for the whole room, so every wall
   matches. It is the look shared by the most walls (an accent wall, such as dark acoustic panels, is left
   out rather than averaged in): its colour at each height is the median over all clean wall pixels of those
   walls, posterised into a few flat bands (a white wall, a wood trim line, a wainscot) by an optimal
   piecewise-constant fit, so noise between heights disappears. A block-sized patch of the surface's own fine
   grain (no outlets or posters, lighting removed) is made seamless and tiled over it. Every wall gets this same
   picture on maps; a room with neither bands nor grain is built as plain blocks.

   With `--wall-detail pattern`, each wall is instead judged **plain** or **patterned** on its own. A plain wall
   is built as above: real blocks plus framed wall items. A patterned wall (stone, brick, panelling,
   wallpaper) gets a map on every block: the photo where a camera saw it well, and everywhere else patches of
   that same surface grown outward, brightness-matched and blended at the seams (patch-based filling in the
   spirit of image quilting, Efros & Freeman, SIGGRAPH 2001). Only pixels seen within about 60 degrees of
   straight on are kept or copied, since steep views are smeared. The photos' own lighting (sun bands, cast
   shadows, glare) is evened out first: brightness changes slower than about 2 blocks are removed, colour and
   finer detail stay, and blown-out glare or near-black shadow is treated as unseen and filled with pattern. A wall the photos hardly saw borrows
   patches from the room's other patterned walls. With `--perceptron`, detected windows and doors are kept
   out of the wall pattern.

   Other modes: `--wall-detail items` (plain blocks and wall items only), `photo` (the raw photo on every block,
   flat paint where unseen) and `none` (blocks only). Extra flags go in `extraArgs` in
   `config/everything_cubed.json`.
7. Windows and built-in shelves come from what the photos show on each wall (`walltex.project_features`): glass
   only where the most head-on photo shows glass, so the stone between windows stays, and shelving seen on a
   wall becomes vanilla bookshelf furniture standing on the floor in front of it (furniture is never part of a
   wall; a bookcase already placed at the same spot is dropped as a duplicate). Windows guessed from 3D clusters alone, on walls the photos did
   not see, never take more than 60% of a wall. `--mirror-walls` (opt-in) lets a wall the photos hardly saw
   copy the opposite wall when that one was seen well: right for halls, libraries and churches, wrong for a
   bedroom with a window on one side only.
8. Guarantees a front door on a free stretch of wall, and keeps the floor in front of every door clear.
   Lights are only the ones found in the photos (lamps, chandeliers, ceiling lights): a dark room stays dark.

## Optional: open-vocabulary detection (`--perceptron`)

Off by default, because it sends the photos to [Perceptron](https://docs.perceptron.inc) (`api.perceptron.inc`).
With `--perceptron` (key in `$PERCEPTRON_API_KEY`), every photo also goes through Perceptron Mk1.5 `/v1/detect`
with no category list, so it finds things SegFormer mislabels (Suzzallo's reading tables, which SegFormer calls
benches, or its chandeliers).

- Before upload, each photo is cropped to the part the reconstruction kept, and people are blurred: SegFormer's
  person pixels plus the boxes of a local COCO person detector (torchvision Faster R-CNN v2, weights downloaded
  once). This is best effort: very small, distant people can be missed.
- **Vanilla first**: a detected name that is a vanilla block's name, or very nearly (plurals, "flowerpot"), is
  built as that block. The block list (1,265 blocks in 26.3) is read from the Minecraft client jar
  (`vanilla.py`). Full cubes such as bookshelves and barrels fill the object's footprint; anything else
  (lantern, bell, candle, lectern) is one block at its real place, hanging if it hangs. Building materials
  ("glass", "stone") never match, since in a photo those are almost always something else.
- For the remaining names, a text-only Mk1.5 call maps each one to a template (descriptions in `roomplan.TEMPLATE_INFO`), to
  `unknown` (built as a plain `generic` box, with the detected name kept in the plan) or to `ignore` (people,
  walls, ...). Every name and what it became is in `plan.json` under `detections`, and per photo in
  `detections.json`.
- Boxes are painted over SegFormer's labels, except wall, floor and ceiling (paintings, clocks, mirrors, sconces and
  curtains may take wall pixels, rugs floor pixels). Each box only relabels pixels near the object's own depth.
- Detections from all views meet in the reconstructed 3D space. Each view contributes one label vote per 3 cm
  voxel before furniture is clustered, so a close photo with more pixels does not outweigh several other views.
- When the same object is detected in at least two views, its matching 3D observations give a more stable centre
  and footprint for placement. Single-view objects keep the normal cluster path. Matched objects record their
  source view indices and estimated centre in `plan.json`.
- Photos are sent concurrently with `reasoning_effort` `none` (about 3x faster than `high`;
  `--perceptron-reasoning` changes it). Answers are cached per exact upload in `<photos>/.perceptron_cache/`, so
  re-runs send nothing. Delete that folder to detect again.

`--known-height METRES` corrects the reconstruction's scale from the room's real ceiling height. MapAnything's metric
scale is learned from ordinary rooms; it put Suzzallo's 19.8 m ceiling at 7.85 m.

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

**For good walls**, also take one photo of each wall **straight on**: stand back near the opposite wall, face
the wall squarely and get as much of it in the frame as you can. Wall textures and patterns only come from
what a camera saw well, and a wall seen only at a steep angle from far away comes out streaked. In big rooms
take several straight-on photos along each long wall. Avoid shooting into bright windows; their glare is
not wall.

**Big or unusual rooms** (halls, atriums, very high ceilings): the reconstruction's sense of scale is learned
from ordinary rooms and can be far off. If you know the floor-to-ceiling height, pass it with
`--known-height METRES` (for example Suzzallo's reading room is 65 ft, `--known-height 19.8`).

**Turning options on in the mod**: put flags in `extraArgs` in `config/everything_cubed.json`, for example

```json
{
  "python": "/path/to/venv/bin/python",
  "extraArgs": ["--known-height", "19.8", "--perceptron"]
}
```

`--perceptron` needs `PERCEPTRON_API_KEY` set in the environment Minecraft is started from, and sends the photos
(people blurred) to Perceptron; see above.

# Everything Cubed

Everything Cubed turns a real room into a Minecraft build. Give it photos or a scan from the iOS app; the Python analyzer writes a `plan.json`, and the Fabric mod builds the room in front of you. Beds, doors, storage, lights, and seats use working Minecraft blocks with custom furniture models around them.

The same Python code can export a video or scan as a 3D mesh with browser viewers. Those exports are useful for inspection, but image-only 3D reconstruction is still imperfect. Check the geometry before treating it as a measurement.

## What you need

- **Minecraft 26.3** with Fabric. The mod builds with **JDK 25** (`build.gradle` sets Java release 25).
- **Python 3.12** is recommended for PyTorch and MapAnything. The analyzer dependencies are in [`analyzer/requirements.txt`](analyzer/requirements.txt).
- A **CUDA GPU** is recommended for photo and video reconstruction. LiDAR scans use measured depth and do not need MapAnything.
- **FFmpeg** only if you give the standalone 3D modeller a video file; it extracts the sampled frames.

The first photo analysis downloads model weights from Hugging Face. Leave several gigabytes free for the models and outputs.

## Build a room in Minecraft

Create the Python environment the mod searches for, from the repository root:

```bash
python3.12 -m venv ~/.minecraft/everything_cubed/venv
~/.minecraft/everything_cubed/venv/bin/pip install -r analyzer/requirements.txt
./gradlew build
```

If your game directory is elsewhere, put the venv in `<game dir>/everything_cubed/venv`, or set its Python executable in `config/everything_cubed.json`. The mod jar is written under `build/libs/`.

Put 4–12 overlapping photos of one room in `<game dir>/everything_cubed/photos/bedroom`. Show the floor, ceiling, corners, and each wall. Wider, straight-on wall photos help the wall art; avoid blur and strong window glare. Then run in Minecraft with commands enabled:

```text
/everything_cubed build bedroom
```

The build appears a few blocks in front of you. The argument can be a folder name under `everything_cubed/photos`, an absolute folder path, or a `plan.json` path. These are the other main commands:

| Command | Action |
| --- | --- |
| `/everything_cubed buildscaled 1.5 bedroom` | Force 1.5 blocks per metre. |
| `/everything_cubed undo` | Revert the last build or spectate changes. |
| `/everything_cubed phone` | Pair the iOS scanner with the game. |
| `/everything_cubed spectate` | Ask the connected phone to spectate the last built LiDAR scan. |
| `/everything_cubed spectate stop` | End spectating and return to your previous game mode. |

The mod reanalyzes a photo folder each time by default. Passing a `plan.json` path uses that saved plan directly.

## Scan with the iOS app

Run `/everything_cubed phone`, open the [scanner app](ios/README.md) on the same network, and enter the pairing code from Minecraft chat. Finish the scan in the app. The mod saves it under `everything_cubed/photos/`, analyzes it, and builds it.

On a LiDAR device, **RoomPlan** sends measured depth, walls, openings, furniture boxes, and camera poses. The analyzer uses those measurements without running an image-only depth model. On an iPhone without LiDAR, an **ARKit photo scan** sends tracked camera poses; MapAnything estimates depth from the photos.

After a built RoomPlan scan, the app can **Spectate this room**. Both player and phone approve the session. RoomPlan furniture moves in Minecraft as whole pieces once its new position is stable; people seen by the phone appear as villagers, including sleeping in a bed when lying there. Either side can stop the session. The [iOS guide](ios/README.md) covers pairing, scan types, and app setup.

## Export a video or scan to 3D

The standalone modeller accepts a video, a plain photo folder, or an iOS scan folder with `capture.json`. Run these from the repository root:

```bash
uv run --no-project --with-requirements analyzer/requirements.txt python analyzer/video_to_3d.py room.mp4 --out room_3d --device cuda
uv run --no-project --with-requirements analyzer/requirements.txt python analyzer/video_to_3d.py scan_folder --out scan_3d
```

For video, it samples **16 frames** across the clip by default. `--frames` accepts **2–512**. More frames cost more time and memory, and can make a long scan worse if its batches fail to align. For a first pass, use a short, steady clip with the room seen from several positions.

| Output | What it contains |
| --- | --- |
| `plan.json` | The Minecraft room plan. |
| `reconstruction.ply` | Colored triangles at reconstructed image pixels. |
| `solid.ply` | A smoothed surface joining nearby reconstructed points. |
| `model.obj` and `model.mtl` | A simplified block model for a 3D editor. |
| `view.html` | A browser view of the room plan and reconstructed surfaces. |
| `photo_view.html` | Cutouts from the source frames at their reconstructed 3D positions. |
| `photo_on_original.html` | The same cutouts over a preview of the unchanged reconstruction mesh. |

The two photo HTML files are generated automatically for **video and plain photo folders**. Scan folders with `capture.json` currently get the plan, mesh, OBJ, and `view.html`; they do not get the photo HTML files. Open an HTML file directly in a browser. In the photo viewer, drag to look, use **WASD** to move, **Q/E** to change height, and **Enter room** to start inside. The frame slider and **Combine all frames** control let you inspect source views separately or together.

If an older output folder already has `reconstruction.ply` and `photo_view.html`, make the original-mesh overlay without rerunning the models:

```bash
uv run --no-project --with-requirements analyzer/requirements.txt python analyzer/original_mesh_overlay.py room_3d
```

The main tuning options are:

| Option | Effect |
| --- | --- |
| `--frames 64` | Sample more of a video; default is 16. |
| `--model-resolution 728` | Give MapAnything more image detail; default is 518. |
| `--batch-views 3 --low-gpu-memory` | Use smaller batches and CPU offload when GPU memory is tight. |
| `--max-width 2160` | Keep larger extracted frames for photo textures; mesh inference still uses `--model-resolution`. |
| `--photo-faces 12000` | Put more photo triangles from each frame into the HTML viewer; default is 6000. |
| `--mesh-max-side 0` | Keep every reconstructed pixel in the PLY; 0 is the default. |
| `--splat-radius 0.12` | Set the smoothing reach in metres for `solid.ply`. |
| `--wall-detail photo` | Make the Minecraft wall plan use photo detail. |

For a better source video, move slowly, keep focus and zoom steady, and show the floor, ceiling, walls, and furniture from several angles. A single long sweep with repeated or blurred views can leave gaps, split furniture, or collapse the room. A textured viewer shows what the reconstruction placed; it cannot fix incorrect depth by itself.

## Run the analyzer directly

For photos when you only need the Minecraft plan and its `view.html`:

```bash
uv run --no-project --with-requirements analyzer/requirements.txt python analyzer/everything_cubed_analyzer.py --images ~/rooms/bedroom --out ~/rooms/bedroom/plan.json
```

MapAnything is the default model for photo reconstruction. SegFormer labels walls, floor, ceiling, and objects on the **original images**; `roomplan.py` uses their reconstructed 3D positions to make the block plan. The default MapAnything weights are `facebook/map-anything-apache`. `--backend vggt` selects the VGGT fallback. LiDAR input goes through `lidar_analyzer.py` and RoomPlan geometry instead.

Walls, floor, openings, and furniture are inferred from what the cameras saw. Unseen walls are estimated. The default `--wall-detail uniform` draws measured wall paint and detected wall items; the other modes are `style`, `pattern`, `items`, `photo`, and `none`. Use `--known-height METRES` if you know the true ceiling height and need to correct a photo reconstruction's scale. `--scale BLOCKS_PER_METRE` forces the Minecraft scale.

The build templates cover household furniture and classroom objects such as lecterns, boards, clocks, curtains, monitors, and projection screens. Small detected items can sit on desks and tables; switches, outlets, alarms, dispensers, and detectors attach to walls or ceilings. Placement depends on the image labels and reconstructed positions, so a missing or badly positioned object usually starts with the scan rather than the block template.

`--perceptron` is optional. It sends photos to the Perceptron API to name additional objects and needs `PERCEPTRON_API_KEY`. The analyzer tries to blur people before upload, but that detection is best effort. Without this flag, the photos stay in the local analysis pipeline.

## Mod settings

The mod creates `config/everything_cubed.json`. For a custom Python path or analyzer options, edit it:

```json
{
  "python": "/path/to/venv/bin/python",
  "extraArgs": ["--known-height", "2.7"],
  "alwaysReanalyze": true,
  "keepScans": 5
}
```

`alwaysReanalyze` defaults to `true`. Set it to `false` to reuse a `plan.json` newer than the photos. `keepScans` keeps the newest five phone scans; `0` keeps all. Photo folders you created yourself are not pruned.

## Code map

| Part | Job |
| --- | --- |
| `analyzer/everything_cubed_analyzer.py` | Main photo and scan pipeline; writes `plan.json` and `view.html`. |
| `analyzer/perception.py` and `analyzer/lidar_analyzer.py` | Reconstruct and label photos, or read measured LiDAR geometry. |
| `analyzer/roomplan.py` and `analyzer/walltex.py` | Convert measured surfaces and objects into blocks and wall images. |
| `analyzer/video_to_3d.py` | Drive video extraction, analysis, and 3D export. |
| `analyzer/photo_texture.py` and `analyzer/original_mesh_overlay.py` | Build the textured browser views. |
| `src/main/java/com/dubhacks/everything_cubed/` | Run the analyzer, receive phone scans, build and undo rooms, and spectate. |
| `ios/` | Capture and stream ARKit or RoomPlan scans. |

## Build and test

```bash
./gradlew build
./gradlew test
uv run --no-project --with-requirements analyzer/requirements.txt --with pytest python -m pytest tests/analyzer
```

Java tests live in `tests/mod/`, Python tests in `tests/analyzer/`. The iOS app has its own [build instructions](ios/README.md).

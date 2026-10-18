# Analyzer setup

The repository's [main README](../README.md) covers Minecraft builds, scans, video exports, options, and tests. This short copy ships inside the mod jar so setup instructions are still available when the mod reports a Python error. If you are reading the unpacked copy in a game directory, open the [online main guide](https://github.com/at7064/everything_cubed/blob/main/README.md).

Use Python 3.12 if possible. From the repository root, install dependencies in the venv the mod searches for:

```bash
python3.12 -m venv ~/.minecraft/everything_cubed/venv
~/.minecraft/everything_cubed/venv/bin/pip install -r analyzer/requirements.txt
```

If you are reading this copy from `<game dir>/everything_cubed/analyzer/`, run the pip command there with `-r requirements.txt` instead. If Minecraft uses a different game directory, put the venv in `<game dir>/everything_cubed/venv`, or set `"python"` in `config/everything_cubed.json` to your executable. The first photo scan downloads model weights from Hugging Face. A CUDA GPU is recommended for photo and video scans; LiDAR scans use measured depth.

To run the analyzer by hand:

```bash
uv run --no-project --with-requirements analyzer/requirements.txt python analyzer/everything_cubed_analyzer.py --images /path/to/room --out /path/to/room/plan.json
```

`analyzer/video_to_3d.py` exports meshes and browser viewers. Its `--help` lists the current options; videos and plain photo folders automatically get `photo_on_original.html` as well.

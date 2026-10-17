"""Minecraft block colour palettes and nearest-colour matching (CIELAB).

Surface blocks (walls / floors / ceilings) are matched against the *actual* average texture
colours read from the Minecraft client jar when one is available, so the result tracks whatever
resource version the game is running. A built-in table of approximate colours is used otherwise.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

DYES = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
        "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]

# Approximate average texture colours (used when no client jar is available).
WOOL = dict(zip(DYES, [(233, 236, 236), (240, 118, 19), (189, 68, 179), (58, 175, 217), (248, 197, 39),
                       (112, 185, 25), (237, 141, 172), (62, 68, 71), (142, 142, 134), (21, 137, 145),
                       (121, 42, 172), (53, 57, 157), (114, 71, 40), (84, 109, 27), (160, 39, 34), (20, 21, 25)]))
CONCRETE = dict(zip(DYES, [(207, 213, 214), (224, 97, 0), (169, 48, 159), (35, 137, 198), (241, 175, 21),
                           (94, 168, 24), (213, 101, 142), (54, 57, 61), (125, 125, 115), (21, 119, 136),
                           (100, 31, 156), (44, 46, 143), (96, 59, 31), (73, 91, 36), (142, 32, 32), (8, 10, 15)]))
TERRACOTTA = dict(zip(DYES, [(209, 178, 161), (161, 83, 37), (149, 88, 108), (113, 108, 137), (186, 133, 35),
                             (103, 117, 52), (161, 78, 78), (57, 42, 35), (135, 106, 97), (86, 91, 91),
                             (118, 70, 86), (74, 59, 91), (77, 51, 35), (76, 83, 42), (143, 61, 46), (37, 22, 16)]))
WOODS = {"oak": (162, 130, 78), "spruce": (114, 84, 48), "birch": (192, 175, 121), "jungle": (160, 115, 80),
         "acacia": (168, 90, 50), "dark_oak": (66, 43, 20), "mangrove": (117, 54, 48), "cherry": (226, 178, 172),
         "bamboo": (193, 173, 80), "pale_oak": (227, 217, 215), "crimson": (101, 48, 70), "warped": (43, 104, 99)}
STONES = {"stone": (125, 125, 125), "smooth_stone": (158, 158, 158), "stone_bricks": (122, 121, 122),
          "bricks": (150, 97, 83), "quartz_block": (235, 229, 222), "polished_andesite": (132, 134, 133),
          "polished_diorite": (192, 193, 194), "polished_granite": (154, 106, 89), "deepslate_tiles": (54, 54, 55),
          "polished_deepslate": (72, 72, 73), "mud_bricks": (137, 103, 79), "calcite": (223, 224, 220),
          "smooth_sandstone": (223, 214, 170), "polished_blackstone": (53, 48, 56), "tuff_bricks": (98, 104, 95)}

# Texture names whose block id differs from the texture file name.
TEXTURE_ALIASES = {"quartz_block_side": "quartz_block", "sandstone_top": "smooth_sandstone"}
SURFACE_PATTERNS = [r"^[a-z_]+_planks$", r"^[a-z_]+_concrete$", r"^[a-z_]+_terracotta$", r"^terracotta$",
                    r"^[a-z_]+_wool$", r"^(stone|smooth_stone|stone_bricks|bricks|calcite|mud_bricks|tuff_bricks)$",
                    r"^polished_(andesite|diorite|granite|deepslate|blackstone|tuff)$", r"^deepslate_(tiles|bricks)$",
                    r"^quartz_bricks$", r"^(quartz_block_side|sandstone_top)$"]
SURFACE_EXCLUDE = re.compile(r"glazed")


def srgb_to_lab(rgb) -> np.ndarray:
    """sRGB (0..255, shape (...,3)) -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


class Palette:
    def __init__(self, entries: dict[str, tuple[int, int, int]]):
        self.ids = list(entries)
        self.rgb = np.array([entries[k] for k in self.ids], dtype=np.float64)
        self.lab = srgb_to_lab(self.rgb)

    def nearest(self, rgb, lightness_weight: float = 0.75) -> str:
        """Nearest block id. Lightness is down-weighted because photos are unevenly lit."""
        lab = srgb_to_lab(rgb)
        d = self.lab - lab
        d[:, 0] *= lightness_weight
        return self.ids[int(np.argmin((d ** 2).sum(1)))]


def _prefixed(d: dict, fmt: str) -> dict:
    return {fmt.format(k): v for k, v in d.items()}


def builtin_surface_entries() -> dict:
    e = {}
    e.update(_prefixed(CONCRETE, "minecraft:{}_concrete"))
    e.update(_prefixed(TERRACOTTA, "minecraft:{}_terracotta"))
    e.update(_prefixed(WOOL, "minecraft:{}_wool"))
    e.update(_prefixed(WOODS, "minecraft:{}_planks"))
    e.update(_prefixed(STONES, "minecraft:{}"))
    e["minecraft:terracotta"] = (152, 94, 67)
    return e


def texture_average(png_bytes: bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    a = np.asarray(img, dtype=np.float64)
    h = a.shape[1]
    a = a[:h]  # animated textures are vertical strips; keep the first frame
    if (a[..., 3] < 250).any():
        return None  # transparent / cut-out textures are not solid surfaces
    return tuple(int(round(v)) for v in a[..., :3].reshape(-1, 3).mean(0))


def jar_surface_entries(jar: Path, cache: Path | None = None) -> dict:
    """Average colours of solid surface block textures inside a Minecraft client jar."""
    if cache and cache.exists() and cache.stat().st_mtime >= jar.stat().st_mtime:
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    pats = [re.compile(p) for p in SURFACE_PATTERNS]
    out = {}
    with zipfile.ZipFile(jar) as z:
        for name in z.namelist():
            m = re.fullmatch(r"assets/minecraft/textures/block/([a-z0-9_]+)\.png", name)
            if not m:
                continue
            tex = m.group(1)
            if not any(p.match(tex) for p in pats) or SURFACE_EXCLUDE.search(tex):
                continue
            avg = texture_average(z.read(name))
            if avg is not None:
                out["minecraft:" + TEXTURE_ALIASES.get(tex, tex)] = avg
    if cache and out:
        cache.write_text(json.dumps(out))
    return out


def surface_palette(jar: str | None, cache_dir: Path | None = None) -> Palette:
    if jar and Path(jar).is_file():
        try:
            entries = jar_surface_entries(Path(jar), cache_dir / "surface_palette.json" if cache_dir else None)
            if len(entries) > 30:
                return Palette(entries)
        except Exception as ex:  # corrupt jar, missing Pillow, ...
            print(f"[palette] could not read textures from {jar}: {ex}; using built-in colours")
    return Palette(builtin_surface_entries())


# Walls only use these blocks, with these exact colours: the paint drawn on the wall frames is the very
# same colour as the block behind them, so frames and any bare wall block blend together.
_WALL_HEX = {
    "terracotta": "985E43", "white_terracotta": "D1B1A1", "orange_terracotta": "9F5224",
    "magenta_terracotta": "95576C", "light_blue_terracotta": "706C8A", "yellow_terracotta": "BA8524",
    "lime_terracotta": "677535", "pink_terracotta": "A04D4E", "gray_terracotta": "392923",
    "light_gray_terracotta": "876B62", "cyan_terracotta": "575C5C", "purple_terracotta": "7A4958",
    "blue_terracotta": "4C3E5C", "brown_terracotta": "4C3223", "green_terracotta": "4C522A",
    "red_terracotta": "8E3C2E", "black_terracotta": "251610",
    "white_concrete": "CFD5D6", "orange_concrete": "E06101", "magenta_concrete": "A9309F",
    "light_blue_concrete": "2389C7", "yellow_concrete": "F1AF15", "lime_concrete": "5EA918",
    "pink_concrete": "D5658E", "gray_concrete": "373A3E", "light_gray_concrete": "7D7D73",
    "cyan_concrete": "157788", "purple_concrete": "64209C", "blue_concrete": "2C2E8F",
    "brown_concrete": "603C1F", "green_concrete": "495B24", "red_concrete": "8E2121", "black_concrete": "080A0F",
}
WALL_COLOURS = {"minecraft:" + k: tuple(int(v[i:i + 2], 16) for i in (0, 2, 4)) for k, v in _WALL_HEX.items()}
WALL_PALETTE = Palette(WALL_COLOURS)

DYE_PALETTE = Palette(WOOL)          # beds, carpets
WOOD_PALETTE = Palette(WOODS)        # tables, chairs, doors, shelves

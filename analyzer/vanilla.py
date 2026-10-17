"""The vanilla Minecraft block catalogue, read from the client jar: id, English name and shape.

Detected things whose name is a vanilla block's name (or very nearly) are built as that block before any of our
own furniture templates are considered: a lantern is a lantern, a lectern a lectern. Only what vanilla does not
have falls through to the templates (sofa, desk, chandelier, ...) and then to plain "generic" boxes.

Shape matters for placement: a full cube (bookshelf, barrel) fills the object's footprint; anything else
(lantern, candle, bell) is a single block at the object's position, hanging or standing.
"""
from __future__ import annotations

import difflib
import json
import re
import zipfile
from pathlib import Path

# Technical or invisible blocks: never what a photo shows.
TECHNICAL = {"air", "cave_air", "void_air", "barrier", "light", "structure_void", "structure_block", "jigsaw",
             "moving_piston", "piston_head", "command_block", "chain_command_block", "repeating_command_block",
             "end_gateway", "end_portal", "nether_portal", "fire", "soul_fire", "water", "lava", "bubble_column",
             "test_block", "test_instance_block"}
# Full cubes that are objects, not building materials. Other cubes ("glass", "stone", "oak planks") are
# materials: a detected "glass" is a drinking glass far more often than a glass block.
OBJECT_CUBES = {"bookshelf", "chiseled_bookshelf", "barrel", "crafting_table", "jukebox", "note_block", "furnace",
                "smoker", "blast_furnace", "loom", "cartography_table", "fletching_table", "smithing_table", "beehive",
                "target", "tnt", "dispenser", "dropper", "observer", "hay_block", "pumpkin", "carved_pumpkin",
                "jack_o_lantern", "melon", "sponge", "crafter"}
# Used when no client jar is available (tests, a missing jar): common household objects in recent versions.
BUILTIN = {
    "lantern": ("Lantern", False), "soul_lantern": ("Soul Lantern", False), "bell": ("Bell", False),
    "lectern": ("Lectern", False), "chest": ("Chest", False), "barrel": ("Barrel", True),
    "anvil": ("Anvil", False), "cauldron": ("Cauldron", False), "flower_pot": ("Flower Pot", False),
    "candle": ("Candle", False), "bookshelf": ("Bookshelf", True), "chiseled_bookshelf": ("Chiseled Bookshelf", True),
    "crafting_table": ("Crafting Table", True), "jukebox": ("Jukebox", True), "decorated_pot": ("Decorated Pot", False),
    "ladder": ("Ladder", False), "scaffolding": ("Scaffolding", False), "brewing_stand": ("Brewing Stand", False),
    "composter": ("Composter", False), "campfire": ("Campfire", False), "grindstone": ("Grindstone", False),
    "loom": ("Loom", True), "iron_bars": ("Iron Bars", False), "end_rod": ("End Rod", False),
}
SIMILAR = 0.88  # difflib ratio for "very nearly the same name" (plurals, "flowerpot" for "flower pot")


def _model_chain_is_cube(z, model, names, depth=0):
    """True when a block model inherits from block/cube (full cube geometry)."""
    model = model.split(":")[-1]
    if model in ("block/cube", "block/cube_all", "block/cube_column"):
        return True
    path = f"assets/minecraft/models/{model}.json"
    if depth > 8 or path not in names:
        return False
    parent = json.loads(z.read(path)).get("parent")
    return bool(parent) and _model_chain_is_cube(z, parent, names, depth + 1)


def read_catalog(jar: Path, cache: Path | None = None) -> dict:
    """{block id (no namespace): (English name, is full cube)} for every vanilla block in the jar."""
    if cache and cache.exists() and cache.stat().st_mtime >= jar.stat().st_mtime:
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    out = {}
    with zipfile.ZipFile(jar) as z:
        names = set(z.namelist())
        lang = json.loads(z.read("assets/minecraft/lang/en_us.json"))
        for path in names:
            m = re.fullmatch(r"assets/minecraft/blockstates/([a-z0-9_]+)\.json", path)
            if not m or m.group(1) in TECHNICAL:
                continue
            bid, state = m.group(1), json.loads(z.read(path))
            cube = False
            if "variants" in state and state["variants"]:
                first = next(iter(state["variants"].values()))
                first = first[0] if isinstance(first, list) else first
                cube = _model_chain_is_cube(z, first.get("model", ""), names)
            out[bid] = (lang.get(f"block.minecraft.{bid}", bid.replace("_", " ").title()), cube)
    if cache and out:
        cache.write_text(json.dumps(out))
    return out


def catalog(jar: str | None, cache_dir: Path | None = None, log=print) -> dict:
    if jar and Path(jar).is_file():
        try:
            c = read_catalog(Path(jar), cache_dir / "vanilla_catalog.json" if cache_dir else None)
            log(f"[vanilla] {len(c)} blocks from {Path(jar).name}")
            return c
        except Exception as ex:  # corrupt jar, unexpected format
            log(f"[vanilla] could not read {jar}: {ex}; using the built-in list")
    return dict(BUILTIN)


def _norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("_", " "))
    return " ".join(s.split())


def match(name: str, cat: dict):
    """The vanilla block a detected name refers to, as (id, is cube), or None. Exact name (or singular) first,
    then a very close spelling. Only objects: non-cube blocks and OBJECT_CUBES, never building materials."""
    n = _norm(name)
    if not n:
        return None
    keys = {}
    for bid, (eng, cube) in cat.items():
        if cube and bid not in OBJECT_CUBES:
            continue
        for k in {_norm(eng), _norm(bid)}:
            keys.setdefault(k, (bid, cube))
    tries = [n] + ([n[:-1]] if n.endswith("s") and not n.endswith("ss") else []) + ([n[:-2]] if n.endswith("es") else [])
    for t in tries:
        if t in keys:
            return keys[t]
    best, score = None, 0.0
    for k, v in keys.items():
        r = difflib.SequenceMatcher(None, n.replace(" ", ""), k.replace(" ", "")).ratio()
        if r > score:
            best, score = v, r
    return best if score >= SIMILAR else None

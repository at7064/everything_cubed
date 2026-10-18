"""Checks roomplan.arrange_seating on the dorm scan that showed the problem: four sofas evenly around a coffee table
(RoomPlan put each 1.0-1.2 m from the table's centre, one per side), at 1.385 blocks/m. Placed one by one they came
out 1 block away on one side, 2 on another and one on a diagonal; as a group, each must sit on its own side, all at
the same gap. A group that doesn't fit stays where it was. And the scale is nudged so a table side and the seats
along it differ by an even number of blocks (a 1-block table can't have a 2-block sofa centred on it).
Run: python tests/analyzer/test_seating.py
"""
import sys
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2] / "analyzer"   # the modules under test
sys.path.insert(0, str(ANALYZER))

import numpy as np

from roomplan import Room, arrange_seating, seating_scale


def cl(c, size):
    c, half = np.array(c, float), np.array(size, float) / 2
    return {"c": c, "lo": c - half, "hi": c + half}


room = Room(0, 7.2, 0, 4.5, 3.7, 1.385)
occ = np.full((room.W, room.D), -1)
table = ({"type": "coffee_table", "x": 4, "z": 2, "w": 1, "d": 1}, cl([3.4, 0.4, 2.0], [0.89, 0.54, 0.82]))
# Where they ended up one by one (as in the dorm plan), and their real RoomPlan centres around the table.
sofas = [({"type": "sofa", "x": 3, "z": 3, "w": 2, "d": 1, "facing": "south"}, cl([3.5, 0.4, 3.1], [1.27, 0.86, 0.86])),
         ({"type": "sofa", "x": 3, "z": 0, "w": 2, "d": 1, "facing": "north"}, cl([3.3, 0.4, 0.9], [1.25, 0.85, 0.8])),
         ({"type": "sofa", "x": 5, "z": 1, "w": 1, "d": 1, "facing": "east"}, cl([4.5, 0.4, 2.2], [0.8, 0.81, 0.77])),
         ({"type": "sofa", "x": 2, "z": 2, "w": 1, "d": 1, "facing": "west"}, cl([2.4, 0.4, 1.8], [0.83, 0.86, 0.86]))]
placed = [table] + sofas
for o, _ in placed:
    occ[o["x"]:o["x"] + o["w"], o["z"]:o["z"] + o["d"]] = 1
arrange_seating(placed, room, occ, log=print)
t = table[0]
south, north, east, west = (o for o, _ in sofas)
assert (south["facing"], north["facing"], east["facing"], west["facing"]) == ("south", "north", "east", "west")
gaps = [south["z"] - (t["z"] + t["d"]), t["z"] - (north["z"] + north["d"]), east["x"] - (t["x"] + t["w"]),
        t["x"] - (west["x"] + west["w"])]
assert len(set(gaps)) == 1, f"uneven gaps {gaps}"
# Each one overlaps the table along its side (none on a diagonal).
assert east["z"] <= t["z"] < east["z"] + east["d"] and west["z"] <= t["z"] < west["z"] + west["d"]
assert south["x"] <= t["x"] < south["x"] + south["w"] and north["x"] <= t["x"] < north["x"] + north["w"]
assert (occ == 1).sum() == 1 + 2 + 2 + 1 + 1, "occupancy follows the new places"
print(f"four sofas, one per side, all {gaps[0]} block(s) from the table")

# A lamp right where the east sofa would go: the group stays exactly as it was.
occ2 = np.full((room.W, room.D), -1)
occ2[t["x"] + 1, t["z"]] = 1
for o, _ in placed:
    occ2[o["x"]:o["x"] + o["w"], o["z"]:o["z"] + o["d"]] = 1
sofas[2][0].update(x=6, z=1)   # the east sofa had to go elsewhere
occ2[6, 1] = 1
kept = [dict(o) for o, _ in sofas]
arrange_seating(placed, room, occ2, log=print)
assert [dict(o) for o, _ in sofas] == kept, "a group that doesn't fit is left alone"
print("OK")

# The scale: at 1.385 blocks/m the 0.89 m table is 1 block and the 1.27 m sofas 2, which can't be centred. A
# nearby scale makes the table 2 blocks along the sofas while it stays 1 along the 1-block end sofas.
clusters = [("coffee_table", table[1])] + [("sofa", c) for _, c in sofas]
s2 = seating_scale(1.385, clusters, log=print)
assert 1.385 * 0.9 <= s2 <= 1.385 * 1.35 and s2 != 1.385, s2
assert round(0.89 * s2) == 2 and round(1.27 * s2) == 2 and round(0.82 * s2) == 1 and round(0.83 * s2) == 1, s2
assert seating_scale(s2, clusters, log=print) == s2, "an even scale is kept"
print(f"scale 1.385 -> {s2:.2f}: OK")

# Beds keep their proportions: Minecraft beds are 2 long, so a 1.1 x 2.0 m single is one bed wide and a 1.6 m queen
# two, at this room's scale too (counting from the width alone made the single 3 beds wide at 2.5 blocks/m).
from roomplan import place_object
big = Room(0, 4.0, 0, 4.0, 2.6, 2.5)
for width, beds in ((1.12, 1), (1.6, 2), (1.95, 2)):
    bed_cl = {"lo": np.array([1.0, 0, 1.0]), "hi": np.array([1.0 + width, 0.6, 3.0]), "c": np.array([1.0 + width / 2, 0.3, 2.0]),
              "rgb": np.array([200.0, 200, 200]), "n": 1000}
    b = place_object("bed", bed_cl, big, np.full((big.W, big.D), -1), [], lambda x, z: ("west", 1.0))
    assert sorted((b["w"], b["d"])) == sorted((beds, 2)), (width, b)
print("beds: single 1 wide, queen and king 2 wide: OK")

# A lone chair in front of a table's long side (a LiDAR scan: 5 x 9 blocks at 2.5 blocks/m). RoomPlan's chair box is
# 0.6 m (1.5 blocks) square, centred at block (3.13, 2.36): just in front of the table (x 2-4, z 0-1). The footprint
# was centred for 2 x 2 before chairs became 1 x 1, which aimed at (2, 1), on the table, and the search then took
# the first free cell, north-west, beside the fridge at the table's end. It must stand at (3, 2).
dorm = Room(0, 2.0, 0, 3.6, 2.3, 2.5)
taken = np.full((dorm.W, dorm.D), -1)
taken[2:5, 0:2] = 1   # table
taken[1, 0] = 1       # fridge
chair_cl = {**cl([3.13 / 2.5, 0.4, 2.36 / 2.5], [0.6, 0.8, 0.6]), "rgb": np.array([110.0, 80, 50]), "n": 1000}
tables_cl = [cl([3.5 / 2.5, 0.4, 1.0 / 2.5], [1.2, 0.8, 0.8])]
ch = place_object("chair", chair_cl, dorm, taken, tables_cl, lambda x, z: ("north", z))
assert (ch["x"], ch["z"], ch["w"], ch["d"]) == (3, 2, 1, 1), ch
# Target taken: the free cell nearest the real centre wins, not the first one in the ring.
taken2 = np.full((dorm.W, dorm.D), -1)
taken2[2:5, 0:2] = 1; taken2[1, 0] = 1; taken2[3, 2] = 1   # something already stands in the chair's spot
ch2 = place_object("chair", chair_cl, dorm, taken2, tables_cl, lambda x, z: ("north", z))
assert (ch2["x"], ch2["z"]) in ((3, 3), (2, 2)), ch2
print(f"lone chair in front of the table: (3, 2); spot taken: ({ch2['x']}, {ch2['z']}): OK")

# A chair pulled up to a table that stands against the north wall (newest scan): the chair's centre is 0.58 m from
# the wall, the table's 0.51 m from it, so the chair faces the table (back to the south), not the wall.
by_wall = np.full((dorm.W, dorm.D), -1)
by_wall[2:5, 0:2] = 1
tucked = {**cl([4.59 / 2.5, 0.4, 1.41 / 2.5], [0.6, 0.8, 0.6]), "rgb": np.array([110.0, 80, 50]), "n": 1000}
near_table = [cl([4.48 / 2.5, 0.4, 0.9 / 2.5], [1.3, 0.8, 0.6])]
ch3 = place_object("chair", tucked, dorm, by_wall, near_table, lambda x, z: ("north", z))
assert ch3["facing"] == "south", ch3
# A sofa against the wall with its coffee table in front keeps its back to the wall.
sofa = place_object("sofa", {**cl([1.0, 0.4, 0.35], [1.6, 0.8, 0.7]), "rgb": np.array([90.0, 90, 90]), "n": 1000},
                    dorm, np.full((dorm.W, dorm.D), -1), [cl([1.0, 0.2, 1.3], [0.9, 0.4, 0.5])], lambda x, z: ("north", z))
assert sofa["facing"] == "north", sofa
print("seat by a table at the wall faces the table; sofa at the wall keeps its back to it: OK")

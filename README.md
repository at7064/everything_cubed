# Everything Cubed

A Fabric mod for Minecraft 26.3. Give it a few photos of a real room and it builds a matching house:
the room's real size is scaled to blocks, the walls are drawn from the photos (flat paint plus posters, boards and so on, as map art in
invisible frames on every wall block), the floor is colour-matched block by block,
detected furniture becomes real, usable blocks (beds you can sleep in, doors that open, barrels and
bookshelves that hold items, cauldron sinks, ...), and the build gets a gable roof and a front door.

## Use

1. Install the Python analyzer once. See [analyzer/README.md](analyzer/README.md).
2. Put 4–12 photos of one room in a folder, for example `<game dir>/everything_cubed/photos/bedroom`.
3. In game (operator / cheats on):
   - `/everything_cubed build bedroom`. This can be a folder name under `everything_cubed/photos`, an absolute path, or a `plan.json`.
   - `/everything_cubed buildscaled 1.5 bedroom` forces 1.5 blocks per metre.
   - `/everything_cubed undo` reverts the last build (up to 5).

### From an iPhone with LiDAR

Run `/everything_cubed phone` and scan the room with the iOS app in [ios/](ios/README.md). The scan streams straight
into the game and is built in front of you. LiDAR gives depth and camera positions directly, so the 3D
reconstruction step is skipped.

Furniture is drawn as detailed models (legs, drawers, cushions, burners, ...) around the real blocks that do the work:
open cabinets, wardrobes and fridges, sleep in beds, cook on stoves, fill bottles at sinks. Right-click a chair, sofa
or toilet to sit (sneak to stand up), and right-click a lamp to switch it on or off. Breaking the block inside a model,
or punching a chair or lamp, removes that piece of furniture.

The house is built a few blocks in front of you, with the floor at your feet level. Analysis runs in the
background, and progress shows in chat. If `plan.json` is newer than the photos, the analysis is skipped.

## How it works

`analyzer/` (Python, bundled into the mod jar and unpacked to `<game dir>/everything_cubed/analyzer`):
MapAnything / VGGT reconstruct the room in 3D, SegFormer (ADE20K) labels walls, floor, ceiling and furniture,
and `roomplan.py` turns that into a block-level `plan.json`.
`src/main/java/.../build/HouseBuilder.java` places that plan in the world.

## Build

Requires JDK 25: `./gradlew build`.

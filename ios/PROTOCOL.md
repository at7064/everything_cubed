# Phone stream protocol (version 1)

The iOS app connects to the mod over TCP: port 25599 by default (`"phonePort"` in `config/everything_cubed.json`),
advertised over Bonjour as `_everythingcubed._tcp`.

## Framing

Every message, in both directions, is:

```
u32 header length | header (UTF-8 JSON) | u32 payload length | payload
```

Lengths are big-endian. The header always has `"type"`. Binary attachments are listed in
`"blobs": [{"name", "format", "length"}]` and stored back to back in the payload, in that order.
Limits: 1 MB header, 64 MB payload.

Implementations: `ios/EverythingCubedScanner/Network/WireMessage.swift` and
`src/main/java/com/dubhacks/everything_cubed/stream/StreamMessage.java`.

## Phone to mod

| type | fields | blobs |
|---|---|---|
| `hello` | `protocol` (1), `code` (pairing code from `/everything_cubed phone`), `device`, `app` | |
| `begin` | `mode` (`capture` / `spectate`), `folder` (spectate: the scan to spectate, from `ready`), `name`, `source` (`roomplan`: RoomPlan's LiDAR scan; `tracked`: photos under ARKit world tracking, any iPhone), `app` | |
| `frame` | `seq`, `kind` (`keyframe` / `photo`), `time`, `tracking`, `imageSize` [w,h], `depthSize` [w,h], `K` (3×3), `cam2world` (4×4), `pointCount` | `image` (jpeg), `depth` (f16), `confidence` (u8), `points` (f32xyz: ARKit's tracked feature points, world metres), `person` (u8, spectate) |
| `room` | `tracking` (spectating only: ARKit's tracking state, `normal`, `limited: relocalizing`, ...) | `room` (RoomPlan `CapturedRoom` as JSON: the finished scan's, or the live one while spectating) |
| `planes` | | `planes` (JSON `{"planes": [...]}`: per ARKit plane anchor `transform` 4×4 row-major, `center`, `extent` [w,h], `alignment`, `classification` (wall/floor/ceiling/table/seat/window/door/none), `boundary`), sent at the end of a `tracked` scan |
| `end` | `build` (default true), `scale` (blocks per metre, optional) | |
| `ping` | | | (also every second while spectating: the phone is there)
| `spectate_decline` | `text`: why the phone said no to `spectate_invite` | |

`hello` must come first. Matrices are row-major, in ARKit conventions: world +Y is up, metres; the camera looks
down -Z with +Y up, and `K` belongs to the image in the sensor's landscape orientation (the analyzer turns
portrait shots upright). `depth` is little-endian float16 metres, `confidence` is ARKit's 0 (low) to 2 (high),
both `depthSize` pixels and aligned with the image. Frames from a phone without LiDAR (`tracked`) have no `depthSize`,
`depth` or `confidence`; their `K` and `cam2world` are still ARKit's, in metres. `person` (spectate only) is 1 where a
person is, also per depth pixel; it is left out when nobody is in view.

## Mod to phone

| type | fields |
|---|---|
| `welcome` | `protocol`, `modes` (supported `begin` modes) |
| `error` | `text`; the mod closes the connection after a failed `hello` |
| `ready` | `folder`, the scan's folder under `everything_cubed/photos` (capture); `mode` = `spectate` (spectate) |
| `built` | `folder`: a capture's house stands in game; the app can now offer to spectate it |
| `stopped` | `text`: the game ended spectating (`/everything_cubed spectate stop`, the player said no or left, ...) |
| `waiting` | `text`: the phone asked to spectate; the player has to allow it in game |
| `spectate_invite` | `folder`: the player ran `/everything_cubed spectate`; the phone asks its user, then sends `begin` or `spectate_decline` |
| `alive` | sent every second while spectating: the game is there |
| `status` | `level` (`info` / `progress` / `success` / `error`), `text`: progress and chat lines for the app's log |
| `pong` | |

## On disk (mode `capture`)

```
photos/<name>-<yyyyMMdd-HHmmss>/
  frame_0001.jpg ...           photos (also usable by the photo-only analyzer backends)
  depth/frame_0001.depth .conf LiDAR depth and confidence
  points/frame_0001.points     ARKit feature points (float32 x, y, z per point)
  planes.json                  ARKit planes, if sent
  room.json                    RoomPlan room model, if sent
  capture.json                 begin fields + per-frame fields, with file names instead of blobs
```

The analyzer reads `capture.json`: frames with depth go to `analyzer/lidar_analyzer.py`; frames without depth
(`tracked`) are reconstructed by MapAnything given each photo's ARKit camera pose and intrinsics
(`perception.reconstruct_tracked`), so the room's scale comes from ARKit instead of being guessed.

## Mode `spectate`

Starts from a RoomPlan scan the player has built (their last build, named by `folder`); the mod refuses it
otherwise. The phone continues that scan's RoomPlan session, so everything is in the scan's coordinates, and sends
`room` (RoomPlan's live room model) and `frame` (LiDAR frames with a `person` mask, for people) about every 0.2 s
each, until `end` (or until the mod sends `stopped`). Nothing is saved: they go to
`analyzer/spectate_analyzer.py`, which the mod keeps running with the scan's `plan.json` (its `arkit` frame maps
ARKit metres onto the plan's blocks) and `room.json`. It answers with whole pieces of furniture to move or add, never
single blocks. While it works on one message, only the newest of each kind that arrives is kept. The player is in
spectator mode until the session ends.

Both sides must allow it. When the phone asks first (`begin`), the mod answers `waiting` and asks the player in
chat ([Allow] / [Deny], `/everything_cubed spectate allow|deny`, 60 s); only then does it send `ready`, and the
phone streams nothing before that. When the player asks first (`/everything_cubed spectate`), the mod sends
`spectate_invite`; the phone asks its user and answers with `begin` (which starts at once) or `spectate_decline`.
Either side ends it (`end` / `stopped`). While spectating each side sends something every second (`ping` / `alive`)
and ends it after 5 s of silence, so a dropped connection stops spectating within seconds.

People are in the worker's answers as `{"id", "x", "y", "z"}` in plan blocks; someone lying on a bed also has
`"sleep": [x, z]`, the bed block they lie on, and the mod puts their villager to sleep in that bed.

## Extending

Add a mode by registering a `StreamSession.Factory` in `StreamModes` (mod) and sending that `mode` in `begin`
(app). Unknown message types are ignored by `capture`, so new optional messages don't break older mods.

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
| `begin` | `mode` (`capture`), `name`, `source` (`roomplan` / `depth`), `app` | |
| `frame` | `seq`, `kind` (`keyframe` / `photo`), `time`, `tracking`, `imageSize` [w,h], `depthSize` [w,h], `K` (3×3), `cam2world` (4×4) | `image` (jpeg), `depth` (f16), `confidence` (u8) |
| `room` | | `room` (RoomPlan `CapturedRoom` as JSON) |
| `end` | `build` (default true), `scale` (blocks per metre, optional) | |
| `ping` | | |

`hello` must come first. Matrices are row-major, in ARKit conventions: world +Y is up, metres; the camera looks
down -Z with +Y up, and `K` belongs to the image in the sensor's landscape orientation (the analyzer turns
portrait shots upright). `depth` is little-endian float16 metres, `confidence` is ARKit's 0 (low) to 2 (high),
both `depthSize` pixels and aligned with the image.

## Mod to phone

| type | fields |
|---|---|
| `welcome` | `protocol`, `modes` (supported `begin` modes) |
| `error` | `text`; the mod closes the connection after a failed `hello` |
| `ready` | `folder`, the scan's folder under `everything_cubed/photos` |
| `status` | `level` (`info` / `progress` / `success` / `error`), `text`: progress and chat lines for the app's log |
| `pong` | |

## On disk (mode `capture`)

```
photos/<name>-<yyyyMMdd-HHmmss>/
  frame_0001.jpg ...           photos (also usable by the photo-only analyzer backends)
  depth/frame_0001.depth .conf LiDAR depth and confidence
  room.json                    RoomPlan room model, if sent
  capture.json                 begin fields + per-frame fields, with file names instead of blobs
```

The analyzer's `arkit` backend (`analyzer/perception.py`) reads `capture.json`.

## Extending

Add a mode by registering a `StreamSession.Factory` in `StreamModes` (mod) and sending that `mode` in `begin`
(app). Unknown message types are ignored by `capture`, so new optional messages don't break older mods.

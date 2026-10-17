# Everything Cubed Scanner (iOS)

Scans a room with the LiDAR scanner of an iPhone / iPad Pro and streams it straight into the Everything Cubed
mod, which builds it in front of you. No photo copying, and no 3D reconstruction model on the PC:
ARKit already measures depth and camera positions, so the analyzer only has to label what it sees.

## Use

1. In Minecraft (operator / cheats on), run `/everything_cubed phone`. Chat shows a 6-digit pairing code and the
   computer's address.
2. Open the app on the same Wi-Fi. The game shows up under *Games on this network* (Bonjour). If it doesn't,
   type the address from chat. Enter the code and connect.
3. Name the room, pick a mode and start the scan:
   - **Room scan**: Apple RoomPlan's guided scan. When you finish, RoomPlan's room model (walls, doors, windows,
     furniture boxes) is sent too and saved as `room.json`.
   - **Depth only**: plain ARKit with LiDAR depth. Use this if the depth counter (the cube icon) stays at 0 in
     Room scan.
4. Walk slowly around the room. Photos are taken automatically when you move about 30 cm or turn 15°.
   Tap the shutter for extra close-ups (posters, boards, the floor). Then tap **Finish**.
5. The mod saves the scan under `everything_cubed/photos/<name>-<date>`, analyses it, and builds the house in
   front of you. Progress shows both in chat and in the app. `/everything_cubed build <folder>` rebuilds a saved scan.

## Build

Open `EverythingCubedScanner.xcodeproj` in Xcode 16 or newer, choose your team under *Signing & Capabilities*,
and run on a LiDAR device (iOS 17+). The simulator has no LiDAR or RoomPlan.

## Code

| Folder | What |
|---|---|
| `Network/` | `WireMessage` (the framing in [PROTOCOL.md](PROTOCOL.md)), `StreamClient` (TCP), `ServerBrowser` (Bonjour) |
| `Capture/` | `CaptureSource` with `RoomPlanSource` and `DepthSource`; `FramePump` samples the AR session, `KeyframeSelector` decides what to keep, `FrameEncoder` turns a frame into a `frame` message |
| `AppModel.swift` | connection state, and the `FrameSink` that streams frames to the mod |
| `Views/` | SwiftUI screens |

To add live capture (for example continuous video for real-time world updates): add a `CaptureSource` (or reuse
`DepthSource`), give `FramePump` a time-based selector, send `begin` with `"mode": "live"`, and register a
`live` session in the mod's `StreamModes`. The connection, framing and pairing stay as they are.

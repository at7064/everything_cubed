# Everything Cubed Scanner (iOS)

Scans a room with the LiDAR scanner of an iPhone / iPad Pro and streams it straight into the Everything Cubed
mod, which builds it in front of you. No photo copying, and no 3D reconstruction model on the PC:
ARKit already measures depth and camera positions, so the analyzer only has to label what it sees.

## Use

1. In Minecraft (operator / cheats on), run `/everything_cubed phone`. Chat shows a 6-digit pairing code and the
   computer's address.
2. Open the app on the same Wi-Fi. The game shows up under *Games on this network* (Bonjour). If it doesn't,
   type the address from chat. Enter the code and connect.
3. Name the room and start the scan. It is Apple RoomPlan's guided scan: when you finish, RoomPlan's room model
   (walls, doors, windows, furniture boxes) is sent too and saved as `room.json`.
4. Walk slowly around the room. Photos are taken automatically when you move about 30 cm or turn 15°.
   Tap the shutter for extra close-ups (posters, boards, the floor). Then tap **Finish**.
5. The mod saves the scan under `everything_cubed/photos/<name>-<date>`, analyses it, and builds the house in
   front of you. Progress shows both in chat and in the app. `/everything_cubed build <folder>` rebuilds a saved scan.

### Spectate (live)

After pairing, tap **Spectate** instead of starting a scan. The camera streams to the game about 5 times a second,
and the world around you follows the real room. Hold still for the first 3 seconds while the room is scanned. After
that, a block only changes once the LiDAR has clearly seen the change for about a second. People in view show up as
villagers. You are in spectator mode in game until you tap **Stop** or run `/everything_cubed spectate stop`. You
then get your game mode and position back, and `/everything_cubed undo` reverts the mirrored blocks.

## Build

Open `EverythingCubedScanner.xcodeproj` in Xcode 16 or newer, choose your team under *Signing & Capabilities*,
and run on a LiDAR device (iOS 17+). The simulator has no LiDAR or RoomPlan.

## Code

| Folder | What |
|---|---|
| `Network/` | `WireMessage` (the framing in [PROTOCOL.md](PROTOCOL.md)), `StreamClient` (TCP), `ServerBrowser` (Bonjour) |
| `Capture/` | `CaptureSource`, implemented by `RoomPlanSource` (scan) and `LiveSource` (spectate); `FramePump` samples the AR session, a `FrameSelector` decides what to keep (`KeyframeSelector` for scans, `IntervalSelector` every 0.2 s for spectate), `FrameEncoder` turns a frame into a `frame` message (plus a person mask for spectate) |
| `AppModel.swift` | connection state, and the `FrameSink` that streams frames to the mod |
| `Views/` | SwiftUI screens |

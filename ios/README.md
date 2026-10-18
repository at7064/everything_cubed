# Everything Cubed Scanner (iOS)

Scans a room and streams it straight into the Everything Cubed mod, which builds it in front of you. No photo
copying. Two scan types:

- **LiDAR room scan** (iPhone / iPad Pro): Apple RoomPlan's guided scan. ARKit measures depth and camera
  positions, so no 3D reconstruction model is needed on the PC.
- **Photo scan with ARKit tracking** (any iPhone with ARKit, LiDAR or not): ARKit tracks the phone with the camera
  and motion sensors, so every photo carries where it was taken, in metres. The analyzer gives those positions to
  MapAnything, which then only has to work out depth, not the room's scale.

## Use

1. In Minecraft (operator / cheats on), run `/everything_cubed phone`. Chat shows a 6-digit pairing code and the
   computer's address.
2. Open the app on the same Wi-Fi. The game shows up under *Games on this network* (Bonjour). If it doesn't,
   type the address from chat. Enter the code and connect.
3. Name the room and start the scan. On a LiDAR device it is Apple RoomPlan's guided scan by default: when you
   finish, RoomPlan's room model (walls, doors, windows, furniture boxes) is sent too and saved as `room.json`.
   Without LiDAR (or with the LiDAR switch off) it is a photo scan with ARKit tracking: walk along the walls rather
   than turning on the spot.
4. Walk slowly around the room. Photos are taken automatically when you move about 30 cm or turn 15°.
   Tap the shutter for extra close-ups (posters, boards, the floor). Then tap **Finish**.
5. The mod saves the scan under `everything_cubed/photos/<name>-<date>`, analyses it, and builds the house in
   front of you. Progress shows both in chat and in the app. `/everything_cubed build <folder>` rebuilds a saved scan.

### Spectate (live)

Needs LiDAR (Pro models). After pairing, tap **Spectate** instead of starting a scan. The camera streams to the game about 5 times a second,
and the world around you follows the real room. Hold still for the first 3 seconds while the room is scanned. After
that, a block only changes once the LiDAR has clearly seen the change for about a second. People in view show up as
villagers. You are in spectator mode in game until you tap **Stop** or run `/everything_cubed spectate stop`. You
then get your game mode and position back, and `/everything_cubed undo` reverts the mirrored blocks.

## Build

Open `EverythingCubedScanner.xcodeproj` in Xcode 16 or newer, choose your team under *Signing & Capabilities*,
and run on a device (iOS 17+): any iPhone with ARKit for photo scans, a LiDAR device for RoomPlan scans. The
simulator has no camera tracking, LiDAR or RoomPlan.

## Code

| Folder | What |
|---|---|
| `Network/` | `WireMessage` (the framing in [PROTOCOL.md](PROTOCOL.md)), `StreamClient` (TCP), `ServerBrowser` (Bonjour) |
| `Capture/` | `CaptureSource`, implemented by `RoomPlanSource` (LiDAR), `TrackedPhotoSource` (ARKit world tracking, any iPhone; in `CaptureSource.swift`) and `LiveSource` (spectate, LiDAR); `FramePump` samples the AR session, a `FrameSelector` decides what to keep (`KeyframeSelector` for scans, `IntervalSelector` every 0.2 s for spectate), `FrameEncoder` turns a frame into a `frame` message (depth only when the device has it, plus a person mask for spectate) |
| `AppModel.swift` | connection state, and the `FrameSink` that streams frames to the mod |
| `Views/` | SwiftUI screens |

import AVFoundation
import QuartzCore
import RoomPlan
import Foundation
import Network
import UIKit

/// Connection to the game and the current scan. Also the scan's `FrameSink`: frames go straight to the mod.
@MainActor
final class AppModel: ObservableObject, FrameSink {
    enum Phase: Equatable {
        case disconnected
        case connecting
        case paired
        case scanning
        case finishing   // waiting for RoomPlan's room model and the upload
        case done        // the mod is analysing and building; its messages show in the log
        case spectating  // streaming live; the game world follows the room
    }

    struct LogLine: Identifiable {
        let id = UUID()
        let level: String
        let text: String
    }

    @Published private(set) var phase: Phase = .disconnected
    @Published private(set) var error: String?
    /// Why a scan could not start (shown on the setup screen).
    @Published private(set) var scanError: String?
    @Published private(set) var log: [LogLine] = []
    @Published private(set) var framesSent = 0
    @Published private(set) var framesWithDepth = 0
    @Published private(set) var hint: String?
    @Published private(set) var source: CaptureSource?
    /// The last scan is built in game and can be spectated: it had LiDAR depth and ARKit's map of it was kept.
    @Published private(set) var canSpectate = false
    /// The last scan was a LiDAR room scan (RoomPlan) on a device with LiDAR: the only kind spectate can follow.
    /// The Spectate button is hidden otherwise (photo scans, phones without LiDAR).
    @Published private(set) var scanIsLidar = false

    /// The last scan's folder in the mod, and its RoomPlan source, kept running: spectate continues that RoomPlan
    /// session, so its live objects are in the same coordinates as the scan (and the built house).
    private var scanFolder: String?
    private var scanSource: RoomPlanSource?
    private var scanHadDepth = false
    /// RoomPlan's live room model is sent at most this often (seconds).
    private let roomInterval: TimeInterval = 0.2
    private var lastRoomSent: TimeInterval = 0

    private let client = StreamClient()
    private var pump: FramePump?
    /// Keyframes are skipped while more than this is waiting to go out.
    private let maxQueuedBytes = 12 << 20
    /// Spectate frames are skipped sooner: an old frame is worth less than staying live.
    private let maxQueuedLiveBytes = 1 << 20

    init() {
        client.onState = { [weak self] in self?.connectionChanged($0) }
        client.onMessage = { [weak self] in self?.received($0) }
    }

    // MARK: connection

    func connect(to endpoint: NWEndpoint, code: String) {
        error = nil
        phase = .connecting
        pairingCode = code
        client.connect(to: endpoint)
    }

    func disconnect() {
        stopCapture()
        client.close()
        phase = .disconnected
    }

    private var pairingCode = ""

    private func connectionChanged(_ state: StreamClient.State) {
        switch state {
        case .ready:
            client.send(WireMessage("hello", ["protocol": WireMessage.protocolVersion, "code": pairingCode,
                                              "device": UIDevice.current.name, "app": Self.appVersion]))
        case .failed(let message):
            fail(message)
        case .closed:
            if phase != .disconnected { fail(error ?? "The game closed the connection.") }
        case .connecting:
            break
        }
    }

    private func received(_ msg: WireMessage) {
        switch msg.type {
        case "welcome":
            phase = .paired
        case "error":
            fail(msg.string("text") ?? "The game refused the connection.")
        case "status":
            append(msg.string("level") ?? "info", msg.string("text") ?? "")
        case "ready":
            if msg.string("mode") == "spectate" {
                append("info", "Spectating. Point the phone at the scanned room; changes you clearly see update live.")
            } else {
                scanFolder = msg.string("folder")
                append("info", "Saving as \(msg.string("folder") ?? "a new scan")")
            }
        case "built":
            // The mod built the scan: it can be spectated if it is this scan, with depth and a map to line up to.
            if msg.string("folder") == scanFolder, scanSource != nil, scanHadDepth {
                canSpectate = true
                append("success", "You can spectate this room now.")
            }
        case "stopped":
            // The game ended spectating (/everything_cubed spectate stop, the player left, ...).
            append("info", msg.string("text") ?? "Spectating ended.")
            if phase == .spectating {
                pump?.stop()
                scanSource?.finishLive()
                stopCapture()
                phase = .done
            }
        default:
            break
        }
    }

    private func fail(_ message: String) {
        print("[app] failed: \(message)")
        stopCapture()
        client.close()
        error = message
        phase = .disconnected
    }

    // MARK: scan

    /// `lidar`: Apple RoomPlan's guided LiDAR scan; otherwise photos with ARKit tracking (any iPhone).
    func startScan(name: String, lidar: Bool) {
        print("[app] start scan '\(name)'")
        withCamera { self.beginCapture(name: name, lidar: lidar) }
    }

    /// Streams the camera live over the room just scanned and built; the house in game follows it. Needs LiDAR.
    func startSpectate() {
        guard canSpectate else { return }
        print("[app] start spectate")
        withCamera { self.beginSpectate() }
    }

    private func withCamera(_ then: @escaping @MainActor () -> Void) {
        scanError = nil
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        print("[app] camera permission \(status.rawValue)")
        switch status {
        case .authorized:
            then()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    if granted { then() } else { self.cameraDenied() }
                }
            }
        default:
            cameraDenied()
        }
    }

    private func cameraDenied() {
        print("[app] camera permission denied")
        scanError = "Camera access is off. Turn it on in Settings → Privacy & Security → Camera → Cubed Scanner."
    }

    private func beginCapture(name: String, lidar: Bool) {
        canSpectate = false
        scanIsLidar = false
        scanFolder = nil
        scanSource?.close()
        scanSource = nil
        let src: CaptureSource = lidar && RoomPlanSource.isSupported ? RoomPlanSource() : TrackedPhotoSource()
        source = src
        framesSent = 0
        framesWithDepth = 0
        log.removeAll()
        client.send(WireMessage("begin", ["mode": "capture", "name": name, "source": src.name, "app": Self.appVersion]))
        let p = FramePump(session: src.arSession, sink: self)
        p.onHint = { [weak self] text in if self?.hint != text { self?.hint = text } }
        pump = p
        captureRunning = false
        phase = .scanning
    }

    private func beginSpectate() {
        guard let src = scanSource, let folder = scanFolder else { return }
        source = src
        framesSent = 0
        framesWithDepth = 0
        log.removeAll()
        client.send(WireMessage("begin", ["mode": "spectate", "folder": folder, "source": src.name, "app": Self.appVersion]))
        let p = FramePump(session: src.arSession, sink: self, selector: IntervalSelector(), options: .live)
        p.onHint = { [weak self] text in if self?.hint != text { self?.hint = text } }
        pump = p
        captureRunning = false
        phase = .spectating
    }

    /// Ends spectating; the game gives the player their game mode back and keeps the mirrored blocks.
    func stopSpectate() {
        guard phase == .spectating else { return }
        print("[app] spectate stopped")
        pump?.stop()
        scanSource?.finishLive()
        client.send(WireMessage("end", ["build": false]))
        stopCapture()
        phase = .done   // back to the finished scan: spectate again or scan another room
    }

    private var captureRunning = false

    /// Called once the source's camera view is on screen: RoomPlan only starts reliably inside a window.
    func captureViewAppeared() {
        guard phase == .scanning || phase == .spectating, !captureRunning, let src = source else { return }
        captureRunning = true
        if phase == .spectating, let live = scanSource {
            // RoomPlan may report off the main thread.
            live.startLive { [weak self] room in Task { @MainActor in self?.sendLiveRoom(room) } }
        } else {
            src.start()
        }
        pump?.start()
        print("[app] capture started (\(src.name))")
    }

    /// Abandons the scan without building; the mod keeps whatever photos already arrived.
    func cancelScan() {
        guard phase == .scanning || phase == .finishing else { return }
        print("[app] scan cancelled")
        pump?.stop()
        source?.finish { _ in }
        client.send(WireMessage("end", ["build": false]))
        stopCapture()
        phase = .paired
    }

    func takePhoto() {
        pump?.takePhoto()
    }

    func finishScan() {
        guard phase == .scanning, let src = source else { return }
        pump?.stop()
        phase = .finishing
        append("info", "Finishing scan…")
        scanHadDepth = framesWithDepth > 0
        // A RoomPlan scan stays running (paused RoomPlan, live AR session) so spectate can continue it.
        scanSource = src as? RoomPlanSource
        scanIsLidar = scanSource != nil && RoomPlanSource.isSupported
        sendScan()
    }

    /// Spectating: RoomPlan's live room model to the mod, at most every `roomInterval` and not while Wi-Fi is behind.
    private func sendLiveRoom(_ room: CapturedRoom) {
        let now = CACurrentMediaTime()
        guard phase == .spectating, now - lastRoomSent >= roomInterval, canAcceptFrame,
              let json = try? JSONEncoder().encode(room) else { return }
        lastRoomSent = now
        client.send(WireMessage("room", blobs: [.init(name: "room", format: "roomplan.json", data: json)]))
    }

    private func sendScan() {
        guard let src = source else { return }
        src.finish { [weak self] extra in
            DispatchQueue.main.async {
                guard let self else { return }
                if let extra { self.client.send(extra) }
                self.client.send(WireMessage("end", ["build": true])) { [weak self] in
                    self?.phase = .done
                }
                self.pump = nil
            }
        }
    }

    /// Back to the start screen for another room, staying connected.
    func scanAgain() {
        stopCapture()
        phase = .paired
    }

    private func stopCapture() {
        pump?.stop()
        pump = nil
        source = nil
        hint = nil
    }

    private func append(_ level: String, _ text: String) {
        log.append(LogLine(level: level, text: text))
        if log.count > 200 { log.removeFirst(log.count - 200) }
    }

    // MARK: FrameSink

    var canAcceptFrame: Bool {
        client.queuedBytes < (phase == .spectating ? maxQueuedLiveBytes : maxQueuedBytes)
    }

    func send(frame: WireMessage, hasDepth: Bool) {
        client.send(frame)
        framesSent += 1
        if hasDepth { framesWithDepth += 1 }
    }

    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
    }
}

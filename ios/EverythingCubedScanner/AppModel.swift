import AVFoundation
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

    private let client = StreamClient()
    private var pump: FramePump?
    /// Keyframes are skipped while more than this is waiting to go out.
    private let maxQueuedBytes = 12 << 20

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
            append("info", "Saving as \(msg.string("folder") ?? "a new scan")")
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
        scanError = nil
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        print("[app] start scan '\(name)', camera permission \(status.rawValue)")
        switch status {
        case .authorized:
            beginCapture(name: name, lidar: lidar)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    if granted { self.beginCapture(name: name, lidar: lidar) } else { self.cameraDenied() }
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

    private var captureRunning = false

    /// Called once the source's camera view is on screen: RoomPlan only starts reliably inside a window.
    func captureViewAppeared() {
        guard phase == .scanning, !captureRunning, let src = source else { return }
        captureRunning = true
        src.start()
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
        client.queuedBytes < maxQueuedBytes
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

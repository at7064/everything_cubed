import ARKit
import QuartzCore

/// Where encoded frames go. The scan uplink sends them to the mod; a live mode can use its own sink.
@MainActor
protocol FrameSink: AnyObject {
    /// False while the network is behind; the pump then skips keyframes (it retries a moment later).
    var canAcceptFrame: Bool { get }
    func send(frame: WireMessage, hasDepth: Bool)
}

/// Samples the AR session about 15 times a second, keeps the frames its selector wants (keyframes for a scan,
/// a steady 5 a second for spectate) and shutter photos, encodes them off the main thread and hands them to the
/// sink in order.
@MainActor
final class FramePump {
    private let session: ARSession
    private weak var sink: FrameSink?
    private var selector: any FrameSelector
    private let encoder: FrameEncoder
    private let queue = DispatchQueue(label: "everything_cubed.encode", qos: .userInitiated)
    private var link: CADisplayLink?
    private var inFlight = 0
    private var seq = 0
    private var photoRequested = false

    /// Set when the user should know why nothing is being kept (tracking lost, network behind, ...).
    var onHint: ((String?) -> Void)?

    init(session: ARSession, sink: FrameSink, selector: any FrameSelector = KeyframeSelector(),
         options: FrameEncoder.Options = .scan) {
        self.session = session
        self.sink = sink
        self.selector = selector
        self.encoder = FrameEncoder(options: options)
    }

    func start() {
        guard link == nil else { return }
        let l = CADisplayLink(target: self, selector: #selector(tick))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 10, maximum: 15, preferred: 15)
        l.add(to: .main, forMode: .common)
        link = l
    }

    func stop() {
        link?.invalidate()
        link = nil
    }

    func takePhoto() {
        photoRequested = true
    }

    /// Counts for the console, printed every 2 s, so a silent capture problem shows up in the device log.
    private var stats = (ticks: 0, noFrame: 0, limited: 0, busy: 0, kept: 0, encodeFailed: 0)
    private var lastReport = CACurrentMediaTime()

    private func report() {
        let now = CACurrentMediaTime()
        guard now - lastReport >= 2 else { return }
        lastReport = now
        print("[pump] ticks=\(stats.ticks) noFrame=\(stats.noFrame) limited=\(stats.limited) busy=\(stats.busy) kept=\(stats.kept) encodeFailed=\(stats.encodeFailed) tracking=\(session.currentFrame.map { "\($0.camera.trackingState)" } ?? "none") depth=\(session.currentFrame.map { ($0.sceneDepth ?? $0.smoothedSceneDepth) != nil } ?? false)")
    }

    @objc private func tick() {
        stats.ticks += 1
        report()
        guard let frame = session.currentFrame, let sink else { stats.noFrame += 1; return }
        guard case .normal = frame.camera.trackingState else {
            stats.limited += 1
            if case .limited(.relocalizing) = frame.camera.trackingState {
                return hint("Point at the scanned room to line up with it")
            }
            return hint("Move slowly, tracking is limited")
        }
        let kind: FrameKind
        if photoRequested {
            kind = .photo
        } else if selector.wants(frame) {
            kind = .keyframe
        } else {
            return hint(nil)
        }
        guard inFlight < 2, sink.canAcceptFrame else {
            stats.busy += 1
            return hint("Sending… hold on")
        }
        stats.kept += 1
        photoRequested = false
        selector.kept(frame)
        seq += 1
        let snap = FrameSnapshot(frame, seq: seq, kind: kind)
        inFlight += 1
        let encoder = encoder
        queue.async { [weak self] in
            let msg = encoder.encode(snap)
            Task { @MainActor in
                guard let self else { return }
                self.inFlight -= 1
                if msg == nil { self.stats.encodeFailed += 1 }
                if let msg { self.sink?.send(frame: msg, hasDepth: snap.depth != nil) }
            }
        }
        hint(nil)
    }

    private func hint(_ text: String?) {
        onHint?(text)
    }
}

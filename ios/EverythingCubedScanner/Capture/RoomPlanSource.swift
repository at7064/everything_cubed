import ARKit
import RoomPlan
import UIKit

/// Apple RoomPlan's guided scan. RoomPlan creates and runs its own AR session; the pump samples LiDAR
/// frames from it while it scans, and when finished RoomPlan also gives the parametric room
/// (walls, doors, furniture).
///
/// Spectate continues the same scan: finishing keeps the AR session running, so `startLive` runs RoomPlan again
/// in the scan's world coordinates and reports its live room model (the furniture as it is now) as it updates.
final class RoomPlanSource: NSObject, CaptureSource, RoomCaptureViewDelegate, RoomCaptureSessionDelegate {
    static var isSupported: Bool { RoomCaptureSession.isSupported }

    let name = "roomplan"
    // A view-owned session: handing RoomPlan our own ARSession left the camera stopped right after it started.
    private let captureView = RoomCaptureView(frame: .zero)
    private var onRoom: ((WireMessage?) -> Void)?
    /// Spectating: RoomPlan's live room model goes here, and no finished room is processed.
    private var onLiveRoom: ((CapturedRoom) -> Void)?

    var arSession: ARSession { captureView.captureSession.arSession }
    var view: UIView { captureView }

    override init() {
        super.init()
        captureView.delegate = self
    }

    // RoomCaptureViewDelegate inherits NSCoding; this object is never archived.
    required init?(coder: NSCoder) { nil }
    func encode(with coder: NSCoder) {}

    func start() {
        captureView.captureSession.run(configuration: RoomCaptureSession.Configuration())
        print("[roomplan] run, supported=\(Self.isSupported), in window=\(captureView.window != nil)")
    }

    func finish(extra: @escaping (WireMessage?) -> Void) {
        onRoom = extra
        // The AR session keeps running: spectate continues it, in the same world coordinates as this scan.
        captureView.captureSession.stop(pauseARSession: false)
    }

    /// Spectate: runs RoomPlan again on the scan's AR session; `onUpdate` gets its live room model as it changes.
    func startLive(onUpdate: @escaping (CapturedRoom) -> Void) {
        onLiveRoom = onUpdate
        captureView.captureSession.delegate = self
        captureView.captureSession.run(configuration: RoomCaptureSession.Configuration())
        print("[roomplan] live run for spectate")
    }

    func finishLive() {
        onLiveRoom = nil
        captureView.captureSession.stop(pauseARSession: false)
    }

    /// Done with this scan (another scan starts): the AR session stops.
    func close() {
        onLiveRoom = nil
        captureView.captureSession.arSession.pause()
    }

    func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
        onLiveRoom?(room)
    }

    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
        if onLiveRoom != nil { return false }   // spectating: nothing to process
        if let error { print("[roomplan] capture ended with error: \(error)"); deliver(nil) }
        return error == nil
    }

    func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
        if let error { print("[roomplan] processing failed: \(error)") }
        guard error == nil, let json = try? JSONEncoder().encode(processedResult) else { return deliver(nil) }
        deliver(WireMessage("room", blobs: [.init(name: "room", format: "roomplan.json", data: json)]))
    }

    private func deliver(_ msg: WireMessage?) {
        let f = onRoom
        onRoom = nil
        f?(msg)
    }
}

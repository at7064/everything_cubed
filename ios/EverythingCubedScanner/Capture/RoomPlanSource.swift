import ARKit
import RoomPlan
import UIKit

/// Apple RoomPlan's guided scan. RoomPlan creates and runs its own AR session; the pump samples LiDAR
/// frames from it while it scans, and when finished RoomPlan also gives the parametric room
/// (walls, doors, furniture).
final class RoomPlanSource: NSObject, CaptureSource, RoomCaptureViewDelegate {
    static var isSupported: Bool { RoomCaptureSession.isSupported }

    let name = "roomplan"
    // A view-owned session: handing RoomPlan our own ARSession left the camera stopped right after it started.
    private let captureView = RoomCaptureView(frame: .zero)
    private var onRoom: ((WireMessage?) -> Void)?

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
        captureView.captureSession.stop()
    }

    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
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

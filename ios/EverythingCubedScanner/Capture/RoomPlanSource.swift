import ARKit
import RoomPlan
import UIKit

/// Apple RoomPlan's guided scan. RoomPlan runs on an AR session we own, so the same LiDAR frames
/// can be streamed while it scans; when finished it also gives the parametric room (walls, doors, furniture).
final class RoomPlanSource: NSObject, CaptureSource, RoomCaptureViewDelegate {
    static var isSupported: Bool { RoomCaptureSession.isSupported }

    let name = "roomplan"
    let arSession = ARSession()
    private lazy var captureView = RoomCaptureView(frame: .zero, arSession: arSession)
    private var onRoom: ((WireMessage?) -> Void)?

    var view: UIView { captureView }

    override init() {
        super.init()
    }

    // RoomCaptureViewDelegate inherits NSCoding; this object is never archived.
    required init?(coder: NSCoder) { nil }
    func encode(with coder: NSCoder) {}

    func start() {
        captureView.delegate = self
        captureView.captureSession.run(configuration: RoomCaptureSession.Configuration())
    }

    func finish(extra: @escaping (WireMessage?) -> Void) {
        onRoom = extra
        captureView.captureSession.stop()
    }

    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
        if error != nil { deliver(nil) }
        return error == nil
    }

    func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
        guard error == nil, let json = try? JSONEncoder().encode(processedResult) else { return deliver(nil) }
        deliver(WireMessage("room", blobs: [.init(name: "room", format: "roomplan.json", data: json)]))
    }

    private func deliver(_ msg: WireMessage?) {
        let f = onRoom
        onRoom = nil
        f?(msg)
    }
}

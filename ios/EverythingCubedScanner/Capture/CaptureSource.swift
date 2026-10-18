import ARKit
import SceneKit
import UIKit

/// Something that runs an AR session and shows its own camera UI. The `FramePump` samples frames from
/// `arSession` no matter which source it is, so a new source (for example continuous video for live
/// world updates) only has to provide a session and a view.
protocol CaptureSource: AnyObject {
    /// Shown on the `begin` message so the mod knows how the scan was made.
    var name: String { get }
    var arSession: ARSession { get }
    var view: UIView { get }
    func start()
    /// Stops capturing. `extra` receives any whole-scan result to send before `end` (RoomPlan's room model).
    func finish(extra: @escaping (WireMessage?) -> Void)
}

/// Photos under plain ARKit world tracking, without RoomPlan: works on any iPhone with ARKit, LiDAR or not.
/// ARKit tracks the phone with the camera and the motion sensors, so every kept frame carries where it was
/// taken, in metres; the analyzer gives those camera positions to MapAnything instead of letting it guess the
/// room's scale. On a LiDAR device the frames also carry depth (and the scan is then analysed as a LiDAR scan).
final class TrackedPhotoSource: NSObject, CaptureSource {
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }

    let name = "tracked"
    private let sceneView = ARSCNView(frame: .zero)

    var arSession: ARSession { sceneView.session }
    var view: UIView { sceneView }

    override init() {
        super.init()
        sceneView.automaticallyUpdatesLighting = false
        sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints]  // shows where tracking has a grip
    }

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        print("[tracked] run, depth=\(config.frameSemantics.contains(.sceneDepth))")
    }

    func finish(extra: @escaping (WireMessage?) -> Void) {
        let planes = Self.planesJSON(sceneView.session.currentFrame?.anchors ?? [])
        sceneView.session.pause()
        extra(planes.map { WireMessage("planes", blobs: [.init(name: "planes", format: "json", data: $0)]) })
    }

    /// The flat surfaces ARKit found, measured in metres, labelled where the device can (A12 and later):
    /// wall, floor, ceiling, table, seat, window, door. The analyzer takes the room's walls, floor and ceiling
    /// from them instead of from depth MapAnything had to guess.
    private static func planesJSON(_ anchors: [ARAnchor]) -> Data? {
        let planes: [[String: Any]] = anchors.compactMap { $0 as? ARPlaneAnchor }.map { p in
            let t = p.transform
            return [
                "transform": (0 ..< 4).flatMap { r in (0 ..< 4).map { c in t[c][r] } },  // row-major, like cam2world
                "center": [p.center.x, p.center.y, p.center.z],
                "extent": [p.planeExtent.width, p.planeExtent.height],
                "alignment": p.alignment == .vertical ? "vertical" : "horizontal",
                "classification": label(p.classification),
                "boundary": p.geometry.boundaryVertices.map { [$0.x, $0.y, $0.z] },  // in the plane's own frame
            ]
        }
        guard !planes.isEmpty else { return nil }
        return try? JSONSerialization.data(withJSONObject: ["planes": planes])
    }

    private static func label(_ c: ARPlaneAnchor.Classification) -> String {
        switch c {
        case .wall: return "wall"
        case .floor: return "floor"
        case .ceiling: return "ceiling"
        case .table: return "table"
        case .seat: return "seat"
        case .window: return "window"
        case .door: return "door"
        default: return "none"
        }
    }
}

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
        sceneView.session.pause()
        extra(nil)  // no whole-scan result: the frames and their camera positions are the scan
    }
}

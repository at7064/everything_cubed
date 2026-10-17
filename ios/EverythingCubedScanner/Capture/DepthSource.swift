import ARKit
import SceneKit
import UIKit

/// Plain ARKit world tracking with LiDAR depth, no RoomPlan. Use it if RoomPlan does not hand out depth
/// on a device, and as the starting point for continuous (live) capture.
final class DepthSource: CaptureSource {
    static var isSupported: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

    let name = "depth"
    private let sceneView = ARSCNView(frame: .zero)

    var arSession: ARSession { sceneView.session }
    var view: UIView { sceneView }

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        sceneView.debugOptions = [.showFeaturePoints]
        sceneView.session.run(config)
    }

    func finish(extra: @escaping (WireMessage?) -> Void) {
        sceneView.session.pause()
        extra(nil)
    }
}

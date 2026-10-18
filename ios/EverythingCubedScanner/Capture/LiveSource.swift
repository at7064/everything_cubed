import ARKit
import UIKit

/// Plain ARKit world tracking with LiDAR depth and no RoomPlan UI, for spectate: the camera runs continuously and
/// the pump streams about 5 frames a second. It starts from the scan's world map, so once ARKit has relocalised
/// (recognised the room) its frames are in the scan's coordinates; until then tracking is limited and no frame is
/// sent. Where the device can, ARKit also segments people
/// (`ARFrame.segmentationBuffer`); otherwise the encoder finds them with Vision. Either way they are sent as a mask
/// so the mod shows them as villagers instead of building them.
final class LiveSource: NSObject, CaptureSource {
    static var isSupported: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }
    /// ARKit's own person segmentation can run together with scene depth on this device.
    static var segmentsPeople: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth, .personSegmentationWithDepth])
    }

    let name = "arkit-live"
    let arSession = ARSession()
    private let arView = ARSCNView(frame: .zero)

    var view: UIView { arView }

    private let worldMap: ARWorldMap?

    init(worldMap: ARWorldMap?) {
        self.worldMap = worldMap
        super.init()
        arView.session = arSession
        arView.automaticallyUpdatesLighting = false
    }

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = Self.segmentsPeople ? [.sceneDepth, .personSegmentationWithDepth] : [.sceneDepth]
        config.worldAlignment = .gravity
        config.initialWorldMap = worldMap
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        print("[live] run, people by \(Self.segmentsPeople ? "ARKit" : "Vision"), world map \(worldMap != nil)")
    }

    func finish(extra: @escaping (WireMessage?) -> Void) {
        arSession.pause()
        extra(nil)
    }
}

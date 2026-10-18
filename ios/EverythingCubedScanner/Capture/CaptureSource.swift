import ARKit
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

import ARKit
import simd

/// Decides which AR frames the `FramePump` keeps.
protocol FrameSelector {
    func wants(_ frame: ARFrame) -> Bool
    mutating func kept(_ frame: ARFrame)
}

/// Keeps a frame when the phone has moved or turned enough since the last kept one, so the scan covers
/// the room from many viewpoints without sending near-duplicate photos.
struct KeyframeSelector: FrameSelector {
    var minDistance: Float = 0.3            // metres
    var minAngle: Float = 15 * .pi / 180    // radians
    var minInterval: TimeInterval = 0.25

    private var last: (transform: simd_float4x4, time: TimeInterval)?

    func wants(_ frame: ARFrame) -> Bool {
        guard let last else { return true }
        if frame.timestamp - last.time < minInterval { return false }
        let a = last.transform, b = frame.camera.transform
        let moved = simd_distance(simd_make_float3(a.columns.3), simd_make_float3(b.columns.3))
        let fa = -simd_make_float3(a.columns.2), fb = -simd_make_float3(b.columns.2)  // viewing directions
        let turned = acos(min(1, max(-1, simd_dot(simd_normalize(fa), simd_normalize(fb)))))
        return moved >= minDistance || turned >= minAngle
    }

    mutating func kept(_ frame: ARFrame) {
        last = (frame.camera.transform, frame.timestamp)
    }
}

/// Keeps a frame every `interval` seconds whatever the phone does: a steady stream for spectate.
struct IntervalSelector: FrameSelector {
    var interval: TimeInterval = 0.2    // 5 frames a second

    private var last: TimeInterval?

    func wants(_ frame: ARFrame) -> Bool {
        guard let last else { return true }
        // A little early is fine: display-link ticks don't land exactly on the interval.
        return frame.timestamp - last >= interval * 0.9
    }

    mutating func kept(_ frame: ARFrame) {
        last = frame.timestamp
    }
}

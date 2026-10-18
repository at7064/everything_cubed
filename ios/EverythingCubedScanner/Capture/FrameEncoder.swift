import ARKit
import CoreImage
import simd

/// Why a frame was sent. `photo` = the user pressed the shutter (kept even when the scan is thinned).
enum FrameKind: String {
    case keyframe
    case photo
}

/// The parts of an ARFrame we need, taken on the main thread so the ARFrame itself is released quickly.
/// Immutable; the pixel buffers are only read after this is made.
struct FrameSnapshot: @unchecked Sendable {
    let seq: Int
    let kind: FrameKind
    let time: TimeInterval
    let image: CVPixelBuffer
    let depth: CVPixelBuffer?
    let confidence: CVPixelBuffer?
    let intrinsics: simd_float3x3
    let imageResolution: CGSize
    let transform: simd_float4x4
    let tracking: String
    /// ARKit's tracked 3D feature points (world, metres) seen in this frame: sparse, but measured, also without LiDAR.
    let points: [simd_float3]

    init(_ frame: ARFrame, seq: Int, kind: FrameKind) {
        self.seq = seq
        self.kind = kind
        time = frame.timestamp
        image = frame.capturedImage
        let d = frame.sceneDepth ?? frame.smoothedSceneDepth
        depth = d?.depthMap
        confidence = d?.confidenceMap
        intrinsics = frame.camera.intrinsics
        imageResolution = frame.camera.imageResolution
        transform = frame.camera.transform
        let raw = frame.rawFeaturePoints?.points ?? []
        let step = max(1, raw.count / 3000)  // at most ~3000 per frame
        points = stride(from: 0, to: raw.count, by: step).map { raw[$0] }
        switch frame.camera.trackingState {
        case .normal: tracking = "normal"
        case .limited: tracking = "limited"
        case .notAvailable: tracking = "unavailable"
        }
    }
}

/// Turns a snapshot into a `frame` message: JPEG photo, LiDAR depth as little-endian float16 metres,
/// ARKit confidence (0-2) as bytes, plus intrinsics and the camera-to-world pose (both row-major,
/// ARKit conventions: world +Y up, camera looks down -Z, matrices for the image in sensor orientation).
/// Only used from the pump's serial encode queue.
final class FrameEncoder: @unchecked Sendable {
    private let context = CIContext()
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    func encode(_ s: FrameSnapshot) -> WireMessage? {
        let w = CVPixelBufferGetWidth(s.image), h = CVPixelBufferGetHeight(s.image)
        let quality: CGFloat = s.kind == .photo ? 0.92 : 0.8
        guard let jpeg = context.jpegRepresentation(
            of: CIImage(cvPixelBuffer: s.image), colorSpace: colorSpace,
            options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality])
        else { return nil }

        // Intrinsics are for camera.imageResolution; rescale in case the buffer differs.
        let sx = Float(w) / Float(s.imageResolution.width), sy = Float(h) / Float(s.imageResolution.height)
        let k = s.intrinsics
        let K: [Float] = [k[0][0] * sx, k[1][0] * sx, k[2][0] * sx,
                          k[0][1] * sy, k[1][1] * sy, k[2][1] * sy,
                          0, 0, 1]
        let t = s.transform
        let pose: [Float] = (0 ..< 4).flatMap { r in (0 ..< 4).map { c in t[c][r] } }

        var fields: [String: Any] = [
            "seq": s.seq, "kind": s.kind.rawValue, "time": s.time, "tracking": s.tracking,
            "imageSize": [w, h], "K": K, "cam2world": pose,
        ]
        var blobs = [WireMessage.Blob(name: "image", format: "jpeg", data: jpeg)]
        if !s.points.isEmpty {
            // x, y, z little-endian float32 per point, ARKit world coordinates (metres, +Y up).
            let xyz = s.points.flatMap { [$0.x, $0.y, $0.z] }
            fields["pointCount"] = s.points.count
            blobs.append(.init(name: "points", format: "f32xyz", data: xyz.withUnsafeBytes { Data($0) }))
        }
        if let depth = s.depth, let d = Self.float16(depth) {
            fields["depthSize"] = [CVPixelBufferGetWidth(depth), CVPixelBufferGetHeight(depth)]
            blobs.append(.init(name: "depth", format: "f16", data: d))
            if let conf = s.confidence, let c = Self.bytes(conf) {
                blobs.append(.init(name: "confidence", format: "u8", data: c))
            }
        }
        return WireMessage("frame", fields, blobs: blobs)
    }

    private static func float16(_ buf: CVPixelBuffer) -> Data? {
        guard CVPixelBufferGetPixelFormatType(buf) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf), stride = CVPixelBufferGetBytesPerRow(buf)
        var out = [Float16](repeating: 0, count: w * h)
        for y in 0 ..< h {
            let row = (base + y * stride).assumingMemoryBound(to: Float32.self)
            for x in 0 ..< w { out[y * w + x] = Float16(row[x]) }
        }
        return out.withUnsafeBytes { Data($0) }  // arm64 is little-endian
    }

    private static func bytes(_ buf: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf), stride = CVPixelBufferGetBytesPerRow(buf)
        var out = Data(capacity: w * h)
        for y in 0 ..< h { out.append((base + y * stride).assumingMemoryBound(to: UInt8.self), count: w) }
        return out
    }
}

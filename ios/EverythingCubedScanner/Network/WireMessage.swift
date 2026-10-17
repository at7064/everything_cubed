import Foundation

/// One message of the phone stream protocol (see `ios/PROTOCOL.md`, mirrored by the mod's `StreamMessage`).
/// Wire format: u32 header length, UTF-8 JSON header, u32 payload length, payload (big-endian lengths).
/// Binary attachments are listed in the header's `blobs` and stored back to back in the payload.
struct WireMessage {
    static let protocolVersion = 1

    struct Blob {
        let name: String
        let format: String
        let data: Data
    }

    var header: [String: Any]
    var blobs: [Blob] = []

    init(_ type: String, _ fields: [String: Any] = [:], blobs: [Blob] = []) {
        header = fields
        header["type"] = type
        self.blobs = blobs
    }

    var type: String { header["type"] as? String ?? "" }

    func string(_ key: String) -> String? { header[key] as? String }

    var byteCount: Int { blobs.reduce(0) { $0 + $1.data.count } }

    func encoded() throws -> Data {
        var h = header
        h["blobs"] = blobs.map { ["name": $0.name, "format": $0.format, "length": $0.data.count] as [String: Any] }
        let json = try JSONSerialization.data(withJSONObject: h)
        var out = Data(capacity: 8 + json.count + byteCount)
        out.appendBigEndian(UInt32(json.count))
        out.append(json)
        out.appendBigEndian(UInt32(byteCount))
        for b in blobs { out.append(b.data) }
        return out
    }

    /// Takes one complete message off the front of `buffer`, or returns nil if more bytes are needed.
    static func take(from buffer: inout Data) throws -> WireMessage? {
        guard buffer.count >= 4 else { return nil }
        let hl = Int(buffer.bigEndianUInt32(at: 0))
        guard hl > 0, hl < 1 << 20 else { throw WireError.malformed }
        guard buffer.count >= 8 + hl else { return nil }
        let pl = Int(buffer.bigEndianUInt32(at: 4 + hl))
        guard pl < 64 << 20 else { throw WireError.malformed }
        let total = 8 + hl + pl
        guard buffer.count >= total else { return nil }
        let start = buffer.startIndex
        let headerData = buffer.subdata(in: start + 4 ..< start + 4 + hl)
        let payload = buffer.subdata(in: start + 8 + hl ..< start + total)
        buffer = buffer.subdata(in: start + total ..< buffer.endIndex)
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else { throw WireError.malformed }
        var msg = WireMessage(header["type"] as? String ?? "", header)
        if let list = header["blobs"] as? [[String: Any]] {
            var offset = 0
            for b in list {
                let len = b["length"] as? Int ?? 0
                guard offset + len <= payload.count else { throw WireError.malformed }
                msg.blobs.append(Blob(name: b["name"] as? String ?? "", format: b["format"] as? String ?? "",
                                      data: payload.subdata(in: offset ..< offset + len)))
                offset += len
            }
        }
        return msg
    }
}

enum WireError: Error {
    case malformed
}

extension Data {
    mutating func appendBigEndian(_ v: UInt32) {
        var be = v.bigEndian
        Swift.withUnsafeBytes(of: &be) { append(contentsOf: $0) }
    }

    func bigEndianUInt32(at offset: Int) -> UInt32 {
        let i = startIndex + offset
        return self[i ..< i + 4].reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
    }
}

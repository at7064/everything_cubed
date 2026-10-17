import Foundation
import Network

/// TCP connection to the mod. Messages are queued in order; `queuedBytes` lets producers back off
/// (drop video frames, not data) when Wi-Fi can't keep up. Callbacks arrive on the main queue.
final class StreamClient {
    enum State: Equatable {
        case connecting
        case ready
        case failed(String)
        case closed
    }

    var onState: ((State) -> Void)?
    var onMessage: ((WireMessage) -> Void)?

    private let queue = DispatchQueue(label: "everything_cubed.stream")
    private var connection: NWConnection?
    private var buffer = Data()
    private let lock = NSLock()
    private var _queued = 0

    /// Bytes handed to `send` that the network has not taken yet.
    var queuedBytes: Int {
        lock.lock(); defer { lock.unlock() }
        return _queued
    }

    func connect(to endpoint: NWEndpoint) {
        close()
        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 15
        let c = NWConnection(to: endpoint, using: NWParameters(tls: nil, tcp: tcp))
        connection = c
        print("[stream] connecting to \(endpoint)")
        c.stateUpdateHandler = { [weak self] state in
            print("[stream] state \(state)")
            switch state {
            case .ready: self?.emit(.ready); self?.receive(on: c)
            case .failed(let e): self?.emit(.failed(e.localizedDescription))
            case .waiting(let e): self?.emit(.failed(e.localizedDescription))
            case .cancelled: self?.emit(.closed)
            default: break
            }
        }
        emit(.connecting)
        c.start(queue: queue)
    }

    func send(_ message: WireMessage, completion: (() -> Void)? = nil) {
        guard let c = connection, let data = try? message.encoded() else {
            print("[stream] not sending '\(message.type)': no connection")
            return
        }
        if message.type != "frame" { print("[stream] send \(message.type) (\(data.count) bytes)") }
        lock.lock(); _queued += data.count; lock.unlock()
        c.send(content: data, completion: .contentProcessed { [weak self] error in
            if let error { print("[stream] send failed: \(error)") }
            guard let self else { return }
            self.lock.lock(); self._queued -= data.count; self.lock.unlock()
            if let completion { DispatchQueue.main.async(execute: completion) }
        })
    }

    func close() {
        // Detach first: a cancelled connection's late state updates must not reach the next one's listener.
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        buffer.removeAll()
        lock.lock(); _queued = 0; lock.unlock()
    }

    private func receive(on c: NWConnection) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, done, error in
            guard let self, c === self.connection else { return }
            if let data { self.buffer.append(data) }
            do {
                while let msg = try WireMessage.take(from: &self.buffer) {
                    print("[stream] received \(msg.type) \(msg.string("text") ?? "")")
                    DispatchQueue.main.async { self.onMessage?(msg) }
                }
            } catch {
                self.emit(.failed("The game sent something unreadable."))
                c.cancel()
                return
            }
            if done || error != nil {
                print("[stream] receive ended: done=\(done) error=\(String(describing: error))")
                self.emit(.closed)
                c.cancel()
            } else {
                self.receive(on: c)
            }
        }
    }

    private func emit(_ s: State) {
        DispatchQueue.main.async { self.onState?(s) }
    }
}

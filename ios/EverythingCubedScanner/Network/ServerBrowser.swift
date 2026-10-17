import Foundation
import Network

/// Finds games that ran `/everything_cubed phone` on the local network (Bonjour `_everythingcubed._tcp`).
final class ServerBrowser: ObservableObject {
    static let serviceType = "_everythingcubed._tcp"

    struct Server: Identifiable, Hashable {
        let name: String
        let endpoint: NWEndpoint
        var id: String { name }
    }

    @Published private(set) var servers: [Server] = []
    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let b = NWBrowser(for: .bonjour(type: Self.serviceType, domain: nil), using: .tcp)
        b.browseResultsChangedHandler = { [weak self] results, _ in
            let list = results.compactMap { r -> Server? in
                if case let .service(name, _, _, _) = r.endpoint { return Server(name: name, endpoint: r.endpoint) }
                return nil
            }.sorted { $0.name < $1.name }
            DispatchQueue.main.async { self?.servers = list }
        }
        b.start(queue: .main)
        browser = b
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }
}

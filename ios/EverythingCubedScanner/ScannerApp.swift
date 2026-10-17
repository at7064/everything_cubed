import SwiftUI

@main
struct ScannerApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        switch model.phase {
        case .disconnected, .connecting:
            ConnectView()
        case .paired:
            ScanSetupView()
        case .scanning, .finishing, .done:
            ScanView()
        }
    }
}

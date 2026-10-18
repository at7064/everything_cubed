import SwiftUI

/// The camera view of the running capture source, with progress and the shutter / finish buttons.
/// After finishing, the mod's messages (analysis progress, "Built a ... house") appear in the log.
struct ScanView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ZStack {
            if let source = model.source {
                SourceView(source: source)
                    .ignoresSafeArea()
                    .onAppear { model.captureViewAppeared() }
            }
            VStack(spacing: 12) {
                stats
                Spacer()
                if model.phase != .scanning { logPanel }
                controls
            }
            .padding()
        }
        .background(Color.black)
    }

    private var stats: some View {
        VStack(spacing: 4) {
            HStack(spacing: 16) {
                if model.phase == .scanning || model.phase == .finishing {
                    Button("Cancel", role: .cancel) { model.cancelScan() }
                }
                Label("\(model.framesSent)", systemImage: "photo.stack")
                Label("\(model.framesWithDepth)", systemImage: "cube.transparent")
                    .foregroundStyle(model.framesSent > 3 && model.framesWithDepth == 0 ? .red : .primary)
            }
            .font(.headline.monospacedDigit())
            if let hint = model.hint {
                Text(hint).font(.caption)
            } else if model.framesSent > 3 && model.framesWithDepth == 0 {
                Text("No LiDAR depth yet. Cancel and scan again.").font(.caption)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var logPanel: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(model.log) { line in
                        Text(line.text)
                            .font(.caption.monospaced())
                            .foregroundStyle(color(line.level))
                            .id(line.id)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)
            .onChange(of: model.log.count) {
                if let last = model.log.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder private var controls: some View {
        switch model.phase {
        case .scanning:
            HStack(spacing: 40) {
                Button { model.takePhoto() } label: {
                    Image(systemName: "camera.circle.fill").font(.system(size: 64))
                }
                Button { model.finishScan() } label: {
                    Label("Finish", systemImage: "checkmark.circle.fill").font(.title2.bold())
                }
                .buttonStyle(.borderedProminent)
            }
        case .finishing:
            ProgressView("Sending the scan…").padding().background(.ultraThinMaterial, in: Capsule())
        default:
            Button("Scan another room") { model.scanAgain() }.buttonStyle(.borderedProminent)
        }
    }

    private func color(_ level: String) -> Color {
        switch level {
        case "error": return .red
        case "success": return .green
        case "progress": return .secondary
        default: return .primary
        }
    }
}

/// Hosts a capture source's own UIKit view (RoomCaptureView, ARSCNView, ...).
struct SourceView: UIViewRepresentable {
    let source: CaptureSource

    func makeUIView(context: Context) -> UIView { source.view }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

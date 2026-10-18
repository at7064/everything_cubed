import SwiftUI

/// The live camera while spectating, with the frame count, the mod's messages and Stop.
struct SpectateView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ZStack {
            if let source = model.source {
                SourceView(source: source)
                    .ignoresSafeArea()
                    .onAppear { model.captureViewAppeared() }
            }
            VStack(spacing: 12) {
                VStack(spacing: 4) {
                    Label("Live · \(model.framesSent) frames", systemImage: "dot.radiowaves.left.and.right")
                        .font(.headline.monospacedDigit())
                    if !model.spectateAllowed {
                        Text("Waiting for the game to allow spectating…").font(.caption)
                    } else if let hint = model.hint {
                        Text(hint).font(.caption)
                    } else if model.framesSent > 5 && model.framesWithDepth == 0 {
                        Text("No LiDAR depth: this needs a Pro iPhone or iPad.").font(.caption).foregroundStyle(.red)
                    }
                }
                .padding(10)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                Spacer()
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(model.log.suffix(4)) { line in
                        Text(line.text)
                            .font(.caption.monospaced())
                            .foregroundStyle(line.level == "error" ? .red : line.level == "success" ? .green : .primary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                Button { model.stopSpectate() } label: {
                    Label("Stop", systemImage: "stop.circle.fill").font(.title2.bold())
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            }
            .padding()
        }
        .background(Color.black)
    }
}

import SwiftUI

struct ScanSetupView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("roomName") private var name = "room"
    @State private var kind: CaptureKind = CaptureKind.roomPlan.isSupported ? .roomPlan : .depth

    var body: some View {
        NavigationStack {
            Form {
                Section("Room name") {
                    TextField("bedroom", text: $name)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section {
                    Picker("Scan with", selection: $kind) {
                        ForEach(CaptureKind.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                } footer: {
                    Text(kind == .roomPlan
                         ? "Guided LiDAR room scan. Walk slowly around the room and point at every wall, the floor and the furniture."
                         : "LiDAR depth only, without RoomPlan's guidance or room model.")
                }
                if let scanError = model.scanError {
                    Section { Text(scanError).foregroundStyle(.red) }
                }
                if !kind.isSupported {
                    Section {
                        Text("This needs an iPhone or iPad with LiDAR (Pro models).").foregroundStyle(.red)
                    }
                }
                Section {
                    Text("Photos are taken automatically as you move. Tap the shutter for extra close-ups of walls and posters. When you finish, the house is built in front of you in game.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Section {
                    Button("Start scan") { model.startScan(name: name, kind: kind) }
                        .disabled(!kind.isSupported || name.trimmingCharacters(in: .whitespaces).isEmpty)
                    Button("Disconnect", role: .destructive) { model.disconnect() }
                }
            }
            .navigationTitle("New scan")
        }
    }
}

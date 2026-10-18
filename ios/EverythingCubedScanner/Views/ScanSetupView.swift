import SwiftUI

struct ScanSetupView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("roomName") private var name = "room"
    private let supported = RoomPlanSource.isSupported

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("bedroom", text: $name)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Room name")
                } footer: {
                    Text("Guided LiDAR room scan. Walk slowly around the room and point at every wall, the floor and the furniture.")
                }
                if let scanError = model.scanError {
                    Section { Text(scanError).foregroundStyle(.red) }
                }
                if !supported {
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
                    Button("Start scan") { model.startScan(name: name) }
                        .disabled(!supported || name.trimmingCharacters(in: .whitespaces).isEmpty)
                    Button("Disconnect", role: .destructive) { model.disconnect() }
                }
            }
            .navigationTitle("New scan")
        }
    }
}

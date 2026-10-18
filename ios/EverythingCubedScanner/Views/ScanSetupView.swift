import SwiftUI

struct ScanSetupView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("roomName") private var name = "room"
    @AppStorage("useLidar") private var useLidar = true
    private let hasLidar = RoomPlanSource.isSupported
    private let supported = TrackedPhotoSource.isSupported

    private var lidar: Bool { hasLidar && useLidar }

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
                    Text(lidar
                         ? "Guided LiDAR room scan. Walk slowly around the room and point at every wall, the floor and the furniture."
                         : "Photo scan with ARKit tracking. Walk slowly along the walls (don't just turn on the spot) and point at every wall, the floor and the furniture.")
                }
                if hasLidar {
                    Section {
                        Toggle("LiDAR room scan (RoomPlan)", isOn: $useLidar)
                    } footer: {
                        Text("Off: photo scan with ARKit tracking, as on iPhones without LiDAR.")
                    }
                }
                if let scanError = model.scanError {
                    Section { Text(scanError).foregroundStyle(.red) }
                }
                if !supported {
                    Section {
                        Text("This device does not support ARKit world tracking.").foregroundStyle(.red)
                    }
                }
                Section {
                    Text("Photos are taken automatically as you move, each with where the phone was. Tap the shutter for extra close-ups of walls and posters. When you finish, the house is built in front of you in game.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Section {
                    Button("Start scan") { model.startScan(name: name, lidar: lidar) }
                        .disabled(!supported || name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                Section {
                    Button("Spectate") { model.startSpectate() }
                        .disabled(!LiveSource.isSupported)
                } footer: {
                    Text(LiveSource.isSupported
                         ? "Streams the camera live: the world around you in game follows the room about 5 times a second, and people show up as villagers. You are put in spectator mode until you stop. Hold still for the first few seconds while the room is scanned."
                         : "Spectate needs LiDAR (Pro models): live updates are measured with LiDAR depth. Photo scans above work on this iPhone.")
                }
                Section {
                    Button("Disconnect", role: .destructive) { model.disconnect() }
                }
            }
            .navigationTitle("New scan")
        }
    }
}

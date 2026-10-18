import Network
import SwiftUI

/// Pick the game (found over Bonjour, or typed in) and enter the code from `/everything_cubed phone`.
struct ConnectView: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var browser = ServerBrowser()
    @AppStorage("pairingCode") private var code = ""
    @AppStorage("manualHost") private var host = ""
    @AppStorage("manualPort") private var port = "25599"
    @State private var selected: ServerBrowser.Server?
    /// The address fields are only for when Bonjour can't find the game; they stay hidden until asked for.
    @State private var showManual = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("In Minecraft, run /everything_cubed phone. Your iPhone must be on the same Wi-Fi as the game.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Section {
                    if browser.servers.isEmpty {
                        Label("Looking…", systemImage: "antenna.radiowaves.left.and.right")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(browser.servers) { server in
                        Button {
                            selected = server
                        } label: {
                            HStack {
                                Label(server.name, systemImage: "cube")
                                Spacer()
                                if selected == server { Image(systemName: "checkmark") }
                            }
                        }
                    }
                    if showManual {
                        TextField("Address shown in chat, e.g. 192.168.1.20", text: $host)
                            .keyboardType(.numbersAndPunctuation)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .onChange(of: host) { if !host.isEmpty { selected = nil } }
                        TextField("Port", text: $port)
                            .keyboardType(.numberPad)
                    }
                } header: {
                    Text("Games on this network")
                } footer: {
                    Button(showManual ? "Hide address" : "Game not listed? Enter the address shown in chat") {
                        withAnimation { showManual.toggle() }
                    }
                    .font(.footnote)
                }
                Section("Pairing code") {
                    TextField("6 digits", text: $code)
                        .keyboardType(.numberPad)
                        .font(.title2.monospacedDigit())
                }
                if let error = model.error {
                    Section { Text(error).foregroundStyle(.red) }
                }
                Section {
                    Button {
                        if let endpoint { model.connect(to: endpoint, code: code) }
                    } label: {
                        HStack {
                            Text("Connect")
                            if model.phase == .connecting { Spacer(); ProgressView() }
                        }
                    }
                    // Stays enabled while connecting: a new attempt replaces a stuck one.
                    .disabled(endpoint == nil || code.count != 6)
                }
            }
            .navigationTitle("Everything Cubed")
        }
        .onAppear { browser.start() }
        .onDisappear { browser.stop() }
    }

    private var endpoint: NWEndpoint? {
        if let selected { return selected.endpoint }
        guard showManual else { return nil }   // a hidden address is never used
        let h = host.trimmingCharacters(in: .whitespaces)
        guard !h.isEmpty, let p = NWEndpoint.Port(port) else { return nil }
        return .hostPort(host: NWEndpoint.Host(h), port: p)
    }
}

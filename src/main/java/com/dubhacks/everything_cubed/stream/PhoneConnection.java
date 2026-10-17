package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.google.gson.JsonArray;
import net.minecraft.commands.CommandSource;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Consumer;

/**
 * One connected phone. The first message must be {@code hello} with the pairing code shown by
 * {@code /everything_cubed phone}; the phone then acts for the player who ran that command.
 */
public final class PhoneConnection implements Runnable {
	private final Socket socket;
	private final MinecraftServer server;
	private final DataOutputStream out;
	private UUID player;
	private String device = "phone";

	PhoneConnection(Socket socket, MinecraftServer server) throws IOException {
		this.socket = socket;
		this.server = server;
		this.out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
	}

	public MinecraftServer server() {
		return server;
	}

	public String device() {
		return device;
	}

	/** Runs {@code action} on the server thread with the paired player, if they are online. */
	public void onServer(Consumer<Optional<ServerPlayer>> action) {
		server.execute(() -> action.accept(Optional.ofNullable(server.getPlayerList().getPlayer(player))));
	}

	public synchronized void send(StreamMessage msg) {
		if (socket.isClosed()) return;
		try {
			msg.write(out);
		} catch (IOException e) {
			EverythingCubed.LOGGER.debug("phone send failed", e);
		}
	}

	/** A line for the phone's log. {@code level} is info, progress, success or error. */
	public void status(String level, String text) {
		send(StreamMessage.of("status").with("level", level).with("text", text));
	}

	/** A chat target that shows messages to the player and mirrors them to the phone. */
	public CommandSource mirror(ServerPlayer target) {
		return new CommandSource() {
			@Override
			public void sendSystemMessage(Component message) {
				target.sendSystemMessage(message);
				status("info", message.getString());
			}

			@Override
			public boolean acceptsSuccess() {
				return true;
			}

			@Override
			public boolean acceptsFailure() {
				return true;
			}

			@Override
			public boolean shouldInformAdmins() {
				return false;
			}
		};
	}

	@Override
	public void run() {
		StreamSession session = null;
		try (socket) {
			DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream(), 1 << 16));
			if (!handshake(StreamMessage.read(in))) return;
			while (true) {
				StreamMessage msg = StreamMessage.read(in);
				switch (msg.type()) {
					case "ping" -> send(StreamMessage.of("pong"));
					case "begin" -> {
						if (session != null) session.onDisconnect();
						session = begin(msg);
					}
					case "end" -> {
						if (session != null) session.onEnd(msg);
						session = null;
					}
					default -> {
						if (session != null) session.onMessage(msg);
						else status("error", "'" + msg.type() + "' before 'begin'");
					}
				}
			}
		} catch (IOException e) {
			EverythingCubed.LOGGER.info("Phone {} disconnected: {}", device, e.getMessage());
		} catch (Exception e) {
			EverythingCubed.LOGGER.error("Phone stream failed", e);
			status("error", "Server error: " + e.getMessage());
		} finally {
			if (session != null) session.onDisconnect();
		}
	}

	private boolean handshake(StreamMessage hello) {
		device = hello.string("device", device);
		if (!hello.type().equals("hello") || !String.valueOf(StreamMessage.PROTOCOL).equals(hello.string("protocol", ""))) {
			send(StreamMessage.of("error").with("text", "Unsupported app version, update the mod or the app."));
			return false;
		}
		Optional<UUID> owner = PhoneServer.claim(hello.string("code", ""));
		if (owner.isEmpty()) {
			send(StreamMessage.of("error").with("text", "Wrong pairing code. Run /everything_cubed phone in game."));
			return false;
		}
		player = owner.get();
		JsonArray modes = new JsonArray();
		StreamModes.names().forEach(modes::add);
		send(StreamMessage.of("welcome").with("protocol", StreamMessage.PROTOCOL).with("modes", modes));
		onServer(p -> p.ifPresent(sp -> sp.sendSystemMessage(Component.literal("Phone connected: " + device))));
		EverythingCubed.LOGGER.info("Phone {} connected from {}", device, socket.getRemoteSocketAddress());
		return true;
	}

	private StreamSession begin(StreamMessage msg) throws IOException {
		String mode = msg.string("mode", "capture");
		Optional<StreamSession.Factory> f = StreamModes.get(mode);
		if (f.isEmpty()) {
			status("error", "This mod version does not support mode '" + mode + "'.");
			return null;
		}
		return f.get().open(this, msg);
	}
}

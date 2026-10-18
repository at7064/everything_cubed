package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.command.EverythingCubedCommand;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.minecraft.ChatFormatting;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerPlayer;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Mode {@code spectate}: starts from a room the player has scanned with LiDAR and built ({@code folder} in
 * {@code begin}, their last build). The phone keeps the scan's RoomPlan session running and streams RoomPlan's live
 * room model ({@code room}) and camera frames (about 5 a second each). They go to {@code spectate_analyzer.py}, a
 * worker process that moves built furniture (or adds new pieces) only when RoomPlan clearly shows it, and finds
 * people in the frames. Only the newest message of each kind waits while the worker is busy, so the game never
 * falls behind the phone. Ends with {@code end} from the phone, a dropped connection or
 * {@code /everything_cubed spectate stop}.
 */
public final class SpectateSession implements StreamSession {
	private static final Map<UUID, SpectateSession> ACTIVE = new ConcurrentHashMap<>();

	private final PhoneConnection phone;
	/** Started on the server thread once the build is found; frames before that are dropped. */
	private volatile Process worker;
	private volatile DataOutputStream toWorker;
	private final Object lock = new Object();
	/** The newest RoomPlan update and camera frame not yet sent, while the worker works on the previous message. */
	private StreamMessage waitingRoom, waitingFrame;
	private boolean workerBusy;
	private volatile boolean closed;
	/** Set on the server thread once the player is in spectator mode. */
	private volatile SpectateWorld world;
	private int updates;

	/** Called once from the mod initializer. */
	public static void register() {
		ServerTickEvents.END_SERVER_TICK.register(server -> ACTIVE.values().forEach(s -> {
			SpectateWorld w = s.world;
			if (w != null) w.tick();
		}));
		// Before the player is saved, so they never stay in spectator mode.
		ServerPlayConnectionEvents.DISCONNECT.register((handler, server) -> stop(handler.getPlayer().getUUID(), "you left the game"));
	}

	SpectateSession(PhoneConnection phone, StreamMessage begin) {
		this.phone = phone;
		String folder = begin.string("folder", "");
		phone.onServer(p -> {
			if (p.isEmpty()) {
				phone.status("error", "You are not in the game; join it, then start spectating again.");
				close("the player is not in the game", true);
				return;
			}
			ServerPlayer player = p.get();
			Optional<EverythingCubedCommand.Built> built = EverythingCubedCommand.lastBuild(player.getUUID())
					.filter(b -> folder.isEmpty() || b.plan().getParent().getFileName().toString().equals(folder));
			if (built.isEmpty()) {
				phone.status("error", "Spectate starts from a room you have scanned and built: finish a scan and wait for "
						+ "\"Built a ... house\" first.");
				close("there is no built scan to spectate", true);
				return;
			}
			SpectateSession old = ACTIVE.put(player.getUUID(), this);
			if (old != null) old.close("a new spectate session started", true);
			if (closed) return;
			SpectateWorld w;
			try {
				w = new SpectateWorld(player, built.get());
			} catch (IOException e) {
				phone.status("error", "Could not read the scan's plan: " + e.getMessage());
				close("the scan's plan could not be read", true);
				return;
			}
			try {
				startWorker(built.get().plan());
			} catch (IOException e) {
				EverythingCubed.LOGGER.warn("Could not start spectating", e);
				phone.status("error", "Could not start the spectate analyzer: " + e.getMessage());
				world = w;   // so close() gives the player their game mode back
				close("the analyzer could not start", true);
				return;
			}
			world = w;
			player.sendSystemMessage(Component.literal("Spectating " + built.get().plan().getParent().getFileName()
					+ ". Furniture you move in the room moves here; /everything_cubed spectate stop to exit.").withStyle(ChatFormatting.AQUA));
			phone.send(StreamMessage.of("ready").with("mode", "spectate"));
		});
		EverythingCubed.LOGGER.info("Spectate requested by {} for '{}'", phone.device(), folder);
	}

	private void startWorker(Path plan) throws IOException {
		Process p = AnalyzerRunner.startSpectate(plan);
		toWorker = new DataOutputStream(new BufferedOutputStream(p.getOutputStream(), 1 << 16));
		worker = p;
		thread("everything_cubed-spectate-out", this::readUpdates);
		thread("everything_cubed-spectate-log", this::readLog);
	}

	/** Ends the player's spectate session, if there is one. Server thread: the player is restored right away. */
	public static boolean stop(UUID player, String why) {
		SpectateSession s = ACTIVE.get(player);
		if (s == null) return false;
		s.close(why, true);
		return true;
	}

	@Override
	public void onMessage(StreamMessage msg) throws IOException {
		boolean room = msg.type().equals("room");
		if (closed || worker == null || !(room || msg.type().equals("frame"))) return;
		synchronized (lock) {
			if (workerBusy) {
				// An older waiting one of the same kind is dropped: only the newest matters.
				if (room) waitingRoom = msg;
				else waitingFrame = msg;
				return;
			}
			workerBusy = true;
		}
		send(msg);
	}

	private void send(StreamMessage msg) {
		try {
			msg.write(toWorker);
		} catch (IOException e) {
			if (!closed) EverythingCubed.LOGGER.warn("Spectate worker input closed", e);
			close("the analyzer stopped", false);
		}
	}

	private void readUpdates() {
		try (BufferedReader r = new BufferedReader(new InputStreamReader(worker.getInputStream(), StandardCharsets.UTF_8))) {
			for (String line; (line = r.readLine()) != null; ) {
				if (!line.startsWith("{")) continue;
				JsonObject msg = JsonParser.parseString(line).getAsJsonObject();
				String type = msg.has("type") ? msg.get("type").getAsString() : "";
				if (type.equals("status")) {
					String level = msg.has("level") ? msg.get("level").getAsString() : "info";
					String text = msg.get("text").getAsString();
					phone.status(level, text);
					chat(text, level.equals("success") ? ChatFormatting.GREEN : ChatFormatting.GRAY);
				} else if (type.equals("update")) {
					next();
					if (msg.has("error")) EverythingCubed.LOGGER.warn("Spectate frame failed: {}", msg.get("error").getAsString());
					SpectateWorld w = world;
					if (w != null) phone.onServer(p -> { if (world == w) w.update(msg); });
					if (++updates % 25 == 0 && msg.has("ms"))
						EverythingCubed.LOGGER.debug("Spectate: {} updates, last took {} ms", updates, msg.get("ms").getAsInt());
				}
			}
		} catch (IOException | RuntimeException e) {
			if (!closed) EverythingCubed.LOGGER.warn("Spectate worker output failed", e);
		}
		if (!closed) {
			phone.status("error", "The spectate analyzer stopped; see the game log.");
			close("the analyzer stopped (see the game log)", false);
		}
	}

	/** The worker answered: hand it what waited, if anything (RoomPlan's furniture first, then people). */
	private void next() {
		StreamMessage msg;
		synchronized (lock) {
			msg = waitingRoom != null ? waitingRoom : waitingFrame;
			if (msg == waitingRoom) waitingRoom = null;
			else waitingFrame = null;
			workerBusy = msg != null;
		}
		if (msg != null) send(msg);
	}

	private void readLog() {
		try (BufferedReader r = new BufferedReader(new InputStreamReader(worker.getErrorStream(), StandardCharsets.UTF_8))) {
			for (String line; (line = r.readLine()) != null; ) EverythingCubed.LOGGER.info("[spectate] {}", line);
		} catch (IOException ignored) {
		}
	}

	private void chat(String text, ChatFormatting style) {
		phone.onServer(p -> p.ifPresent(sp -> sp.sendSystemMessage(Component.literal(text).withStyle(style))));
	}

	@Override
	public void onEnd(StreamMessage msg) {
		close("the phone stopped spectating", false);
	}

	@Override
	public void onDisconnect() {
		close("the phone disconnected", false);
	}

	/**
	 * Stops the worker and gives the player back their game mode: right here when {@code onServerThread} (the
	 * command, the player leaving, the server stopping), else on the server's next turn. Idempotent.
	 */
	private void close(String why, boolean onServerThread) {
		synchronized (lock) {
			if (closed) return;
			closed = true;
			waitingRoom = waitingFrame = null;
		}
		try {
			if (toWorker != null) toWorker.close();
		} catch (IOException ignored) {
		}
		if (worker != null) worker.destroy();
		phone.send(StreamMessage.of("stopped").with("text", "Spectating ended: " + why + "."));
		if (onServerThread) finishWorld(why);
		else phone.onServer(p -> finishWorld(why));
		EverythingCubed.LOGGER.info("Spectate ended: {}", why);
	}

	private void finishWorld(String why) {
		ACTIVE.values().remove(this);
		SpectateWorld w = world;
		world = null;
		if (w == null) return;
		w.finish();
		if (w.undo().size() > 0) EverythingCubedCommand.remember(w.undo());
		w.player().sendSystemMessage(Component.literal("Spectating ended (" + why + "); " + w.placed()
				+ " piece(s) of furniture moved or added. /everything_cubed undo reverts them.").withStyle(ChatFormatting.AQUA));
	}

	private static void thread(String name, Runnable r) {
		Thread t = new Thread(r, name);
		t.setDaemon(true);
		t.start();
	}

	/** Ends every session (the server is stopping). */
	public static void stopAll() {
		ACTIVE.values().forEach(s -> s.close("the game is closing", true));
	}
}

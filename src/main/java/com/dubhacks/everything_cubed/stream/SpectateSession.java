package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.command.EverythingCubedCommand;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.minecraft.ChatFormatting;
import net.minecraft.network.chat.ClickEvent;
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
	/** Scan folders the player asked their phone to spectate (/everything_cubed spectate): the player's yes. */
	private static final Map<UUID, Invite> INVITES = new ConcurrentHashMap<>();
	/** How long a request waits for the other side's yes. */
	private static final long ANSWER_MS = 60_000;
	/** While spectating, both sides send something every second; this long without anything ends it. */
	private static final int LOST_MS = 5_000;

	private record Invite(String folder, long at) {
	}

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
	/** Waiting for the player's yes in game (the phone asked first), since this time; 0 once allowed. */
	private volatile long askedAt;
	private EverythingCubedCommand.Built built;
	private ServerPlayer player;
	private int updates;

	/** Called once from the mod initializer. */
	public static void register() {
		ServerTickEvents.END_SERVER_TICK.register(server -> {
			boolean second = server.getTickCount() % 20 == 0;
			long now = System.currentTimeMillis();
			for (SpectateSession s : ACTIVE.values()) {
				SpectateWorld w = s.world;
				if (w != null) {
					w.tick();
					if (second) s.phone.send(StreamMessage.of("alive"));   // the phone's sign that the game is there
				} else if (s.askedAt > 0 && now - s.askedAt > ANSWER_MS) {
					s.close("it was not allowed in game in time", true);
				}
			}
			if (second) INVITES.values().removeIf(i -> now - i.at() > ANSWER_MS);
		});
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
			player = p.get();
			Optional<EverythingCubedCommand.Built> b = EverythingCubedCommand.lastBuild(player.getUUID())
					.filter(x -> folder.isEmpty() || folderOf(x).equals(folder));
			if (b.isEmpty()) {
				phone.status("error", "Spectate starts from a room you have scanned and built: finish a scan and wait for "
						+ "\"Built a ... house\" first.");
				close("there is no built scan to spectate", true);
				return;
			}
			built = b.get();
			SpectateSession old = ACTIVE.put(player.getUUID(), this);
			if (old != null) old.close("a new spectate session started", true);
			if (closed) return;
			// Both sides must say yes. The game said yes first (/everything_cubed spectate): start now.
			Invite invite = INVITES.remove(player.getUUID());
			if (invite != null && invite.folder().equals(folderOf(built))) {
				start();
				return;
			}
			// The phone asked first: the player says yes or no in game.
			askedAt = System.currentTimeMillis();
			phone.send(StreamMessage.of("waiting").with("text", "Allow spectating in game (click Allow in chat)."));
			player.sendSystemMessage(Component.literal("Your phone (" + phone.device() + ") wants to spectate "
					+ folderOf(built) + ": you go into spectator mode and furniture follows the real room. ")
					.withStyle(ChatFormatting.AQUA)
					.append(button("[Allow]", "/everything_cubed spectate allow", ChatFormatting.GREEN))
					.append(Component.literal(" "))
					.append(button("[Deny]", "/everything_cubed spectate deny", ChatFormatting.RED)));
		});
		EverythingCubed.LOGGER.info("Spectate requested by {} for '{}'", phone.device(), folder);
	}

	private static Component button(String label, String command, ChatFormatting colour) {
		return Component.literal(label).withStyle(st -> st.withColor(colour).withBold(true)
				.withClickEvent(new ClickEvent.RunCommand(command)));
	}

	private static String folderOf(EverythingCubedCommand.Built b) {
		return b.plan().getParent().getFileName().toString();
	}

	/** Both sides said yes: spectator mode, the worker, and the phone may stream. Server thread. */
	private void start() {
		askedAt = 0;
		SpectateWorld w;
		try {
			w = new SpectateWorld(player, built);
		} catch (IOException e) {
			phone.status("error", "Could not read the scan's plan: " + e.getMessage());
			close("the scan's plan could not be read", true);
			return;
		}
		try {
			startWorker(built.plan());
		} catch (IOException e) {
			EverythingCubed.LOGGER.warn("Could not start spectating", e);
			phone.status("error", "Could not start the spectate analyzer: " + e.getMessage());
			world = w;   // so close() gives the player their game mode back
			close("the analyzer could not start", true);
			return;
		}
		world = w;
		phone.setReadTimeout(LOST_MS);   // the phone sends something every second: silence means it is gone
		player.sendSystemMessage(Component.literal("Spectating " + folderOf(built)
				+ ". Furniture you move in the room moves here; /everything_cubed spectate stop to exit.").withStyle(ChatFormatting.AQUA));
		phone.send(StreamMessage.of("ready").with("mode", "spectate"));
	}

	/** {@code /everything_cubed spectate}: asks the player's phone to spectate their last build. Server thread. */
	public static boolean invite(ServerPlayer player) {
		Optional<EverythingCubedCommand.Built> b = EverythingCubedCommand.lastBuild(player.getUUID());
		Optional<PhoneConnection> phone = PhoneConnection.of(player.getUUID());
		if (b.isEmpty() || phone.isEmpty()) {
			player.sendSystemMessage(Component.literal(b.isEmpty()
					? "Scan a room with the phone (LiDAR) and build it first; spectate follows that room."
					: "No phone is connected. Run /everything_cubed phone and connect the app first.").withStyle(ChatFormatting.RED));
			return false;
		}
		if (ACTIVE.containsKey(player.getUUID())) {
			player.sendSystemMessage(Component.literal("Already spectating; /everything_cubed spectate stop ends it.").withStyle(ChatFormatting.RED));
			return false;
		}
		String folder = folderOf(b.get());
		INVITES.put(player.getUUID(), new Invite(folder, System.currentTimeMillis()));
		phone.get().send(StreamMessage.of("spectate_invite").with("folder", folder));
		player.sendSystemMessage(Component.literal("Asked your phone to spectate " + folder + "; allow it on the phone.").withStyle(ChatFormatting.GRAY));
		return true;
	}

	/** {@code /everything_cubed spectate allow}: the player's yes to their phone's request. Server thread. */
	public static boolean allow(UUID player) {
		SpectateSession s = ACTIVE.get(player);
		if (s == null || s.askedAt == 0 || s.closed) return false;
		s.start();
		return true;
	}

	/** {@code /everything_cubed spectate deny}. Server thread. */
	public static boolean deny(UUID player) {
		SpectateSession s = ACTIVE.get(player);
		if (s == null || s.askedAt == 0) return false;
		s.close("you said no in game", true);
		return true;
	}

	/** The phone said no to the game's invite. Any thread. */
	static void declined(PhoneConnection phone, String why) {
		INVITES.remove(phone.player());
		phone.onServer(p -> p.ifPresent(sp -> sp.sendSystemMessage(
				Component.literal("Your phone did not start spectating: " + why + ".").withStyle(ChatFormatting.GRAY))));
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
		phone.setReadTimeout(PhoneConnection.IDLE_TIMEOUT_MS);
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

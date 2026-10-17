package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.command.EverythingCubedCommand;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;

/**
 * Mode {@code capture}: saves a LiDAR scan as it streams in, then builds it like
 * {@code /everything_cubed build <folder>}. The folder (under {@code everything_cubed/photos}) holds
 * <ul>
 *   <li>{@code frame_0001.jpg ...}: the photos, so the folder also works with the photo-only analyzer backends</li>
 *   <li>{@code depth/frame_0001.depth}, {@code .conf}: LiDAR depth (metres) and ARKit confidence per photo</li>
 *   <li>{@code room.json}: Apple RoomPlan's room model, when the phone sent one</li>
 *   <li>{@code capture.json}: camera intrinsics and poses per photo, read by the analyzer's {@code arkit} backend</li>
 * </ul>
 */
final class CaptureRecorder implements StreamSession {
	private static final int MAX_FRAMES = 400;
	/** Blob name -> file suffix. Frame blobs with other names are ignored. */
	private static final Map<String, String> FRAME_BLOBS = Map.of("depth", ".depth", "confidence", ".conf");

	private final PhoneConnection phone;
	private final Path dir;
	private final JsonObject capture = new JsonObject();
	private final JsonArray frames = new JsonArray();

	CaptureRecorder(PhoneConnection phone, StreamMessage begin) throws IOException {
		this.phone = phone;
		String name = begin.string("name", "scan").replaceAll("[^A-Za-z0-9_-]+", "_");
		if (name.isBlank()) name = "scan";
		this.dir = AnalyzerRunner.photosDir().resolve(name + "-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")));
		Files.createDirectories(dir.resolve("depth"));
		copyFields(begin, capture);
		capture.addProperty("format", "everything_cubed.arkit");
		capture.addProperty("version", 1);
		capture.addProperty("device", phone.device());
		capture.add("frames", frames);
		phone.send(StreamMessage.of("ready").with("folder", dir.getFileName().toString()));
		EverythingCubed.LOGGER.info("Recording phone scan to {}", dir);
	}

	@Override
	public void onMessage(StreamMessage msg) throws IOException {
		switch (msg.type()) {
			case "frame" -> frame(msg);
			case "room" -> {
				var room = msg.blob("room");
				if (room.isPresent()) {
					Files.write(dir.resolve("room.json"), room.get());
					capture.addProperty("room", "room.json");
				}
			}
			default -> EverythingCubed.LOGGER.debug("capture: ignoring '{}'", msg.type());
		}
	}

	private void frame(StreamMessage msg) throws IOException {
		if (frames.size() >= MAX_FRAMES) return;
		var image = msg.blob("image");
		if (image.isEmpty()) return;
		String base = String.format("frame_%04d", frames.size() + 1);
		JsonObject f = new JsonObject();
		copyFields(msg, f);
		Files.write(dir.resolve(base + ".jpg"), image.get());
		f.addProperty("image", base + ".jpg");
		for (JsonElement e : msg.header().getAsJsonArray("blobs")) {
			JsonObject b = e.getAsJsonObject();
			String suffix = FRAME_BLOBS.get(b.get("name").getAsString());
			if (suffix == null) continue;
			var data = msg.blob(b.get("name").getAsString());
			if (data.isEmpty()) continue;
			String file = "depth/" + base + suffix;
			Files.write(dir.resolve(file), data.get());
			JsonObject ref = new JsonObject();
			ref.addProperty("file", file);
			if (b.has("format")) ref.add("format", b.get("format"));
			f.add(b.get("name").getAsString(), ref);
		}
		frames.add(f);
		if (frames.size() % 10 == 0) phone.status("progress", frames.size() + " photos received");
	}

	@Override
	public void onEnd(StreamMessage msg) throws IOException {
		writeCapture();
		boolean build = !msg.header().has("build") || msg.header().get("build").getAsBoolean();
		Double scale = msg.header().has("scale") && !msg.header().get("scale").isJsonNull() ? msg.header().get("scale").getAsDouble() : null;
		String folder = dir.getFileName().toString();
		phone.status("success", "Saved " + frames.size() + " photos as " + folder);
		if (!build) return;
		phone.onServer(player -> {
			if (player.isEmpty()) {
				phone.status("error", "You are not in the game; build it later with /everything_cubed build " + folder);
				return;
			}
			var src = player.get().createCommandSourceStack().withSource(phone.mirror(player.get()));
			EverythingCubedCommand.start(src, dir, scale);
		});
	}

	@Override
	public void onDisconnect() {
		try {
			// Keep what arrived; it can still be built by hand.
			if (!frames.isEmpty()) writeCapture();
		} catch (IOException e) {
			EverythingCubed.LOGGER.warn("Could not save {}", dir, e);
		}
	}

	private void writeCapture() throws IOException {
		Files.writeString(dir.resolve("capture.json"), new GsonBuilder().setPrettyPrinting().create().toJson(capture));
	}

	private static void copyFields(StreamMessage from, JsonObject to) {
		for (var e : from.header().entrySet())
			if (!e.getKey().equals("type") && !e.getKey().equals("blobs")) to.add(e.getKey(), e.getValue());
	}
}

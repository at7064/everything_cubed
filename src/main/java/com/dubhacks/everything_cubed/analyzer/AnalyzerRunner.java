package com.dubhacks.everything_cubed.analyzer;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.stream.PhoneServer;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.loader.api.FabricLoader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * Runs the bundled Python analyzer (photos -> plan.json) as an external process.
 * The scripts ship inside the mod jar and are unpacked to {@code <game dir>/everything_cubed/analyzer}.
 * Which Python to use is configured in {@code config/everything_cubed.json}.
 */
public final class AnalyzerRunner {
	private static final String[] SCRIPTS = {"everything_cubed_analyzer.py", "perception.py", "roomplan.py", "palette.py", "walltex.py", "wallitems.py", "vanilla.py", "lidar_analyzer.py", "spectate_analyzer.py", "plan_viewer.py", "mesh_export.py", "photo_texture.py", "solid_export.py", "original_mesh_overlay.py", "video_to_3d.py",
			"requirements.txt", "README.md"};
	private static final Set<String> IMAGE_EXT = Set.of("jpg", "jpeg", "png", "webp", "bmp", "heic");
	private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

	private AnalyzerRunner() {
	}

	public static Path workDir() {
		return FabricLoader.getInstance().getGameDir().resolve(EverythingCubed.MOD_ID);
	}

	/** Where relative {@code /everything_cubed build} paths are looked up. */
	public static Path photosDir() {
		return workDir().resolve("photos");
	}

	static final int DEFAULT_KEEP_SCANS = 5;

	/** {@code "keepScans"} in the config: phone scans kept in the photos folder (the newest ones); 0 keeps them all. */
	public static int keepScans() {
		try {
			JsonObject cfg = config();
			return cfg.has("keepScans") ? Math.max(0, cfg.get("keepScans").getAsInt()) : DEFAULT_KEEP_SCANS;
		} catch (IOException | RuntimeException e) {
			return DEFAULT_KEEP_SCANS;
		}
	}

	/** {@code "exterior"} in the config: the houses' outside look, "auto" (from each room) or a style name. */
	public static String exterior() {
		try {
			JsonObject cfg = config();
			return cfg.has("exterior") ? cfg.get("exterior").getAsString() : "auto";
		} catch (IOException | RuntimeException e) {
			return "auto";
		}
	}

	/** Saves {@code style} ("auto" or a style name) as the config's {@code "exterior"}. */
	public static void setExterior(String style) throws IOException {
		JsonObject cfg = config();
		cfg.addProperty("exterior", style);
		Files.writeString(configFile(), GSON.toJson(cfg));
	}

	/** Called as a new phone scan starts recording into {@code current}: see {@link #pruneScans}. */
	public static void pruneOldScans(Path current) {
		int keep = keepScans();
		try {
			for (Path old : pruneScans(photosDir(), current, keep))
				EverythingCubed.LOGGER.info("Deleted old phone scan {} (keepScans = {})", old.getFileName(), keep);
		} catch (IOException e) {
			EverythingCubed.LOGGER.warn("Could not tidy old phone scans", e);
		}
	}

	/**
	 * Deletes the oldest phone scans in {@code photos} so that {@code keep} remain counting {@code current} (the scan
	 * starting now, which has no capture.json yet and is never deleted); {@code keep <= 0} keeps them all. Only folders
	 * the phone stream wrote (they hold a capture.json) are touched: photo folders made by hand are never deleted.
	 * Returns the deleted folders.
	 */
	static List<Path> pruneScans(Path photos, Path current, int keep) throws IOException {
		List<Path> deleted = new ArrayList<>();
		if (keep <= 0 || !Files.isDirectory(photos)) return deleted;
		List<Path> scans;
		try (Stream<Path> s = Files.list(photos)) {
			scans = new ArrayList<>(s.filter(p -> !p.equals(current) && Files.isRegularFile(p.resolve("capture.json"))).toList());
		}
		scans.sort((a, b) -> Long.compare(modified(b), modified(a)));  // newest first
		for (Path old : scans.subList(Math.min(keep - 1, scans.size()), scans.size())) {
			try (Stream<Path> files = Files.walk(old)) {
				for (Path f : files.sorted((a, b) -> b.getNameCount() - a.getNameCount()).toList()) Files.deleteIfExists(f);
			}
			deleted.add(old);
		}
		return deleted;
	}

	private static long modified(Path p) {
		try {
			return Files.getLastModifiedTime(p).toMillis();
		} catch (IOException e) {
			return 0;
		}
	}

	/** Whether folder builds should refresh a cached plan. Defaults to true. */
	public static boolean alwaysReanalyze() {
		try {
			JsonObject cfg = config();
			return !cfg.has("alwaysReanalyze") || cfg.get("alwaysReanalyze").getAsBoolean();
		} catch (IOException | RuntimeException e) {
			return true;
		}
	}

	/** Resolves a user-supplied path; relative paths are looked up in {@link #photosDir()}. */
	public static Path resolveInput(String raw) {
		String s = raw.trim();
		if (s.length() >= 2 && (s.startsWith("\"") && s.endsWith("\"") || s.startsWith("'") && s.endsWith("'")))
			s = s.substring(1, s.length() - 1);
		if (s.startsWith("~")) s = System.getProperty("user.home") + s.substring(1);
		Path p = Path.of(s);
		return p.isAbsolute() ? p : photosDir().resolve(s);
	}

	/** True when a plan exists that is newer than every photo, so the (slow) analysis can be skipped. */
	public static boolean planIsFresh(Path folder, Path plan) throws IOException {
		if (!Files.isRegularFile(plan)) return false;
		long t = Files.getLastModifiedTime(plan).toMillis();
		try (Stream<Path> s = Files.list(folder)) {
			return s.filter(AnalyzerRunner::isImage).allMatch(p -> {
				try {
					return Files.getLastModifiedTime(p).toMillis() <= t;
				} catch (IOException e) {
					return false;
				}
			});
		}
	}

	private static boolean isImage(Path p) {
		String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
		int i = n.lastIndexOf('.');
		// tex_*.png are wall textures written by older analyzer versions, not photos.
		return i > 0 && !n.startsWith(".") && !n.startsWith("tex_") && IMAGE_EXT.contains(n.substring(i + 1));
	}

	public static JsonObject readPlan(Path plan) throws IOException {
		JsonObject o = JsonParser.parseString(Files.readString(plan)).getAsJsonObject();
		if (!o.has("size") || !o.has("walls") || !o.has("floor"))
			throw new IOException("not an everything_cubed plan: " + plan);
		// Textures referenced by the plan live next to it.
		o.addProperty("_dir", plan.toAbsolutePath().getParent().toString());
		return o;
	}

	/** Blocking: runs the analyzer and returns the plan. Progress lines are passed to {@code progress}. */
	public static JsonObject analyze(Path folder, Double scale, Consumer<String> progress) throws IOException, InterruptedException {
		Path scripts = unpackScripts();
		JsonObject cfg = config();
		Path plan = folder.resolve("plan.json");

		List<String> cmd = new ArrayList<>();
		cmd.add(cfg.get("python").getAsString());
		cmd.add(scripts.resolve("everything_cubed_analyzer.py").toString());
		cmd.add("--images");
		cmd.add(folder.toString());
		cmd.add("--out");
		cmd.add(plan.toString());
		minecraftJar().ifPresent(j -> { cmd.add("--mc-jar"); cmd.add(j.toString()); });
		if (scale != null) { cmd.add("--scale"); cmd.add(Double.toString(scale)); }
		for (var e : cfg.getAsJsonArray("extraArgs")) cmd.add(e.getAsString());
		EverythingCubed.LOGGER.info("Running analyzer: {}", String.join(" ", cmd));

		Process proc = new ProcessBuilder(cmd).directory(scripts.toFile()).redirectErrorStream(true).start();
		List<String> tail = new ArrayList<>();
		try (BufferedReader r = new BufferedReader(new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
			for (String line; (line = r.readLine()) != null; ) {
				EverythingCubed.LOGGER.info("[analyzer] {}", line);
				if (line.startsWith("[")) progress.accept(line);
				tail.add(line);
				if (tail.size() > 6) tail.removeFirst();
			}
		}
		int code = proc.waitFor();
		if (code != 0)
			throw new IOException("analyzer exited with code " + code + ": " + String.join(" | ", tail)
					+ " (python: " + cmd.getFirst() + ", see " + scripts.resolve("README.md") + ")");
		return readPlan(plan);
	}

	/**
	 * Starts the spectate worker ({@code spectate_analyzer.py}) for a built LiDAR scan ({@code plan}, its plan.json,
	 * with room.json beside it). It reads the phone's RoomPlan updates and frames on stdin and writes one JSON line of
	 * furniture moves and people per message on stdout; stderr is its log. Same Python as the analyzer.
	 */
	public static Process startSpectate(Path plan) throws IOException {
		Path scripts = unpackScripts();
		List<String> cmd = new ArrayList<>();
		cmd.add(config().get("python").getAsString());
		cmd.add("-u");
		cmd.add(scripts.resolve("spectate_analyzer.py").toString());
		cmd.add("--plan");
		cmd.add(plan.toString());
		EverythingCubed.LOGGER.info("Starting spectate worker: {}", String.join(" ", cmd));
		return new ProcessBuilder(cmd).directory(scripts.toFile()).start();
	}

	/** TCP port for the iOS scanner app, {@code "phonePort"} in the config. */
	public static int phonePort() {
		try {
			JsonObject cfg = config();
			return cfg.has("phonePort") ? cfg.get("phonePort").getAsInt() : PhoneServer.DEFAULT_PORT;
		} catch (IOException | RuntimeException e) {
			return PhoneServer.DEFAULT_PORT;
		}
	}

	private static Optional<Path> minecraftJar() {
		return FabricLoader.getInstance().getModContainer("minecraft")
				.flatMap(c -> c.getOrigin().getPaths().stream()
						.filter(p -> p.toString().endsWith(".jar") && Files.isRegularFile(p))
						.findFirst());
	}

	private static Path unpackScripts() throws IOException {
		Path dir = workDir().resolve("analyzer");
		Files.createDirectories(dir);
		for (String name : SCRIPTS) {
			try (InputStream in = AnalyzerRunner.class.getResourceAsStream("/everything_cubed_analyzer/" + name)) {
				if (in == null) throw new IOException("analyzer resource missing from mod jar: " + name);
				Files.copy(in, dir.resolve(name), StandardCopyOption.REPLACE_EXISTING);
			}
		}
		return dir;
	}

	private static Path configFile() {
		return FabricLoader.getInstance().getConfigDir().resolve(EverythingCubed.MOD_ID + ".json");
	}

	private static JsonObject config() throws IOException {
		Path file = configFile();
		JsonObject cfg = Files.isRegularFile(file) ? JsonParser.parseString(Files.readString(file)).getAsJsonObject() : new JsonObject();
		boolean changed = false;
		String bare = isWindows() ? "python" : "python3";
		if (!cfg.has("python")) {
			cfg.addProperty("python", bare);
			changed = true;
		}
		if (bare.equals(cfg.get("python").getAsString())) {
			// Prefer a venv created by the analyzer README (the game dir, or ~/.minecraft for dev runs).
			String exe = isWindows() ? "Scripts/python.exe" : "bin/python";
			for (Path base : new Path[]{workDir(), Path.of(System.getProperty("user.home"), ".minecraft", EverythingCubed.MOD_ID)}) {
				Path venv = base.resolve("venv").resolve(exe);
				if (Files.isRegularFile(venv)) {
					cfg.addProperty("python", venv.toString());
					changed = true;
					break;
				}
			}
		}
		if (!cfg.has("extraArgs")) {
			cfg.add("extraArgs", new JsonArray());
			changed = true;
		}
		// Written out so the switches are easy to find.
		if (!cfg.has("alwaysReanalyze")) {
			cfg.addProperty("alwaysReanalyze", true);
			changed = true;
		}
		if (!cfg.has("keepScans")) {
			cfg.addProperty("keepScans", DEFAULT_KEEP_SCANS);
			changed = true;
		}
		if (!cfg.has("exterior")) {
			cfg.addProperty("exterior", "auto");
			changed = true;
		}
		if (changed) {
			Files.createDirectories(file.getParent());
			Files.writeString(file, GSON.toJson(cfg));
		}
		return cfg;
	}

	private static boolean isWindows() {
		return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
	}
}

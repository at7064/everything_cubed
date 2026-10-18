package com.dubhacks.everything_cubed.command;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.build.HouseBuilder;
import com.dubhacks.everything_cubed.build.UndoLog;
import com.dubhacks.everything_cubed.stream.PhoneServer;
import com.dubhacks.everything_cubed.stream.SpectateSession;
import com.google.gson.JsonObject;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.minecraft.ChatFormatting;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * /everything_cubed build <folder or plan.json>        analyse the photos (if needed) and build in front of you
 * /everything_cubed buildscaled <blocksPerMetre> <...> same, with a fixed scale
 * /everything_cubed undo                               revert the last build
 * /everything_cubed phone                              pair the iOS scanner app, which then builds in front of you
 * /everything_cubed spectate stop                      stop mirroring the phone's live view (started from the app)
 */
public final class EverythingCubedCommand {
	private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(r -> {
		Thread t = new Thread(r, "everything_cubed-analyzer");
		t.setDaemon(true);
		return t;
	});
	private static final AtomicBoolean BUSY = new AtomicBoolean();
	private static final int MAX_UNDO = 5;
	private static final Deque<UndoLog> HISTORY = new ArrayDeque<>();
	/** Each player's last build, for spectate. */
	private static final Map<UUID, Built> LAST = new ConcurrentHashMap<>();
	private static final Map<UndoLog, UUID> BUILT_BY = new IdentityHashMap<>();

	/** A build in the world: the plan file it came from and where its interior (0, 1, 0) block is, minus (0, 1, 0). */
	public record Built(ServerLevel level, BlockPos origin, Path plan, int w, int d, int h) {
	}

	/** The player's last build, if it is still there (not undone). */
	public static Optional<Built> lastBuild(UUID player) {
		return Optional.ofNullable(LAST.get(player));
	}

	private EverythingCubedCommand() {
	}

	public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
		dispatcher.register(Commands.literal("everything_cubed")
				.requires(Commands.hasPermission(Commands.LEVEL_GAMEMASTERS))
				.then(Commands.literal("build")
						.then(Commands.argument("path", StringArgumentType.greedyString())
								.executes(ctx -> start(ctx, null))))
				.then(Commands.literal("buildscaled")
						.then(Commands.argument("blocksPerMetre", DoubleArgumentType.doubleArg(0.5, 4.0))
								.then(Commands.argument("path", StringArgumentType.greedyString())
										.executes(ctx -> start(ctx, DoubleArgumentType.getDouble(ctx, "blocksPerMetre"))))))
				.then(Commands.literal("undo").executes(EverythingCubedCommand::undo))
				.then(Commands.literal("phone").executes(EverythingCubedCommand::phone))
				.then(Commands.literal("spectate")
						.executes(ctx -> {
							info(ctx.getSource(), "Run /everything_cubed phone, then choose Spectate in the iPhone app. "
									+ "/everything_cubed spectate stop ends it.");
							return 1;
						})
						.then(Commands.literal("stop").executes(EverythingCubedCommand::stopSpectate))));
	}

	private static int start(CommandContext<CommandSourceStack> ctx, Double scale) {
		return start(ctx.getSource(), AnalyzerRunner.resolveInput(StringArgumentType.getString(ctx, "path")), scale);
	}

	/** Analyses {@code input} (if needed) in the background and builds it in front of {@code src}. */
	public static int start(CommandSourceStack src, Path input, Double scale) {
		if (!Files.exists(input)) {
			src.sendFailure(Component.literal("Not found: " + input + " (relative paths are looked up in "
					+ AnalyzerRunner.photosDir() + ")"));
			return 0;
		}
		if (!BUSY.compareAndSet(false, true)) {
			src.sendFailure(Component.literal("An analysis is already running, please wait."));
			return 0;
		}
		// Capture where to build now, the player may walk around while the analysis runs.
		ServerLevel level = src.getLevel();
		BlockPos at = BlockPos.containing(src.getPosition());
		Entity e = src.getEntity();
		Direction facing = e != null ? e.getDirection() : Direction.SOUTH;
		MinecraftServer server = src.getServer();

		info(src, "Reading " + input + " ...");
		WORKER.submit(() -> {
			try {
				JsonObject plan = loadPlan(src, input, scale);
				Path planFile = Files.isRegularFile(input) ? input : input.resolve("plan.json");
				server.execute(() -> build(src, level, at, facing, plan, planFile));
			} catch (Exception ex) {
				EverythingCubed.LOGGER.error("everything_cubed failed", ex);
				server.execute(() -> src.sendFailure(Component.literal("everything_cubed failed: " + ex.getMessage())));
			} finally {
				BUSY.set(false);
			}
		});
		return 1;
	}

	/**
	 * Blocking. A plan.json given directly is built as it is. A folder is analysed again on every build while
	 * "alwaysReanalyze" is true in the config (the default); with it false, a plan.json newer than every photo
	 * is reused.
	 */
	private static JsonObject loadPlan(CommandSourceStack src, Path input, Double scale) throws IOException, InterruptedException {
		if (Files.isRegularFile(input)) return AnalyzerRunner.readPlan(input);
		Path existing = input.resolve("plan.json");
		if (scale == null && !AnalyzerRunner.alwaysReanalyze() && AnalyzerRunner.planIsFresh(input, existing)) {
			info(src, "Using existing plan.json (set \"alwaysReanalyze\": true in config/everything_cubed.json to re-analyse every time).");
			return AnalyzerRunner.readPlan(existing);
		}
		info(src, "Analysing photos - the first run downloads the vision models and can take a while.");
		MinecraftServer server = src.getServer();
		return AnalyzerRunner.analyze(input, scale, line -> server.execute(() -> progress(src, line)));
	}

	private static void build(CommandSourceStack src, ServerLevel level, BlockPos at, Direction facing, JsonObject plan, Path planFile) {
		try {
			JsonObject size = plan.getAsJsonObject("size");
			int w = size.get("x").getAsInt(), d = size.get("z").getAsInt();
			// Put the house a few blocks in front of the player, centred on their line of sight.
			int ahead = 4 + (facing.getAxis() == Direction.Axis.X ? w : d) / 2;
			int cx = at.getX() + facing.getStepX() * ahead;
			int cz = at.getZ() + facing.getStepZ() * ahead;
			BlockPos origin = new BlockPos(cx - w / 2, at.getY() - 1, cz - d / 2);
			UndoLog log = new HouseBuilder(level, origin, plan).build();
			remember(log);
			if (src.getEntity() instanceof ServerPlayer p) {
				Built b = new Built(level, origin, planFile, w, d, size.get("y").getAsInt());
				LAST.put(p.getUUID(), b);
				BUILT_BY.put(log, p.getUUID());
			}
			JsonObject m = plan.getAsJsonObject("metres");
			String metres = m == null ? "" : String.format(" (room %.1f x %.1f m)", m.get("x").getAsDouble(), m.get("z").getAsDouble());
			src.sendSuccess(() -> Component.literal("Built a " + w + " x " + d
					+ " x " + size.get("y").getAsInt() + " house" + metres + " at " + origin.toShortString()
					+ ", " + plan.getAsJsonArray("objects").size() + " furnishings. /everything_cubed undo to revert.")
					.withStyle(ChatFormatting.GREEN), true);
		} catch (Exception ex) {
			EverythingCubed.LOGGER.error("everything_cubed build failed", ex);
			src.sendFailure(Component.literal("Build failed: " + ex.getMessage()));
		}
	}

	/** Adds a finished change (a spectate session's blocks) to the undo history. Server thread. */
	public static void remember(UndoLog log) {
		HISTORY.push(log);
		while (HISTORY.size() > MAX_UNDO) HISTORY.removeLast();
	}

	private static int stopSpectate(CommandContext<CommandSourceStack> ctx) throws CommandSyntaxException {
		ServerPlayer player = ctx.getSource().getPlayerOrException();
		if (!SpectateSession.stop(player.getUUID(), "you stopped it")) {
			ctx.getSource().sendFailure(Component.literal("You are not spectating."));
			return 0;
		}
		return 1;
	}

	private static int undo(CommandContext<CommandSourceStack> ctx) {
		UndoLog log = HISTORY.poll();
		if (log == null) {
			ctx.getSource().sendFailure(Component.literal("Nothing to undo."));
			return 0;
		}
		log.revert();
		UUID by = BUILT_BY.remove(log);
		if (by != null) LAST.remove(by);   // that build is gone: nothing to spectate
		ctx.getSource().sendSuccess(() -> Component.literal("Reverted " + log.size() + " blocks."), true);
		return 1;
	}

	private static int phone(CommandContext<CommandSourceStack> ctx) throws CommandSyntaxException {
		CommandSourceStack src = ctx.getSource();
		ServerPlayer player = src.getPlayerOrException();
		int port;
		try {
			port = PhoneServer.start(src.getServer(), AnalyzerRunner.phonePort());
		} catch (IOException e) {
			src.sendFailure(Component.literal("Could not open the phone port: " + e.getMessage()));
			return 0;
		}
		String code = PhoneServer.pair(player.getUUID());
		String addr = String.join(", ", PhoneServer.addresses().stream().map(a -> a + ":" + port).toList());
		src.sendSuccess(() -> Component.literal("Open Everything Cubed Scanner on your iPhone (same Wi-Fi). Pairing code: ")
				.append(Component.literal(code).withStyle(ChatFormatting.GOLD, ChatFormatting.BOLD)), false);
		info(src, "If the app does not list this game, enter " + (addr.isEmpty() ? "this computer's IP and port " + port : addr) + ".");
		return 1;
	}

	private static void info(CommandSourceStack src, String msg) {
		src.getServer().execute(() -> src.sendSuccess(() -> Component.literal(msg).withStyle(ChatFormatting.GRAY), false));
	}

	private static void progress(CommandSourceStack src, String line) {
		src.sendSuccess(() -> Component.literal(line).withStyle(ChatFormatting.DARK_GRAY), false);
	}
}

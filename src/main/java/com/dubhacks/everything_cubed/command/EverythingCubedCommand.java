package com.dubhacks.everything_cubed.command;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.build.HouseBuilder;
import com.dubhacks.everything_cubed.build.UndoLog;
import com.google.gson.JsonObject;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.minecraft.ChatFormatting;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * /everything_cubed build <folder or plan.json>        analyse the photos (if needed) and build in front of you
 * /everything_cubed buildscaled <blocksPerMetre> <...> same, with a fixed scale
 * /everything_cubed undo                               revert the last build
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
				.then(Commands.literal("undo").executes(EverythingCubedCommand::undo)));
	}

	private static int start(CommandContext<CommandSourceStack> ctx, Double scale) {
		CommandSourceStack src = ctx.getSource();
		Path input = AnalyzerRunner.resolveInput(StringArgumentType.getString(ctx, "path"));
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
				server.execute(() -> build(src, level, at, facing, plan));
			} catch (Exception ex) {
				EverythingCubed.LOGGER.error("everything_cubed failed", ex);
				server.execute(() -> src.sendFailure(Component.literal("everything_cubed failed: " + ex.getMessage())));
			} finally {
				BUSY.set(false);
			}
		});
		return 1;
	}

	/** Blocking. Reads a plan.json directly, reuses a fresh one next to the photos, or runs the analyzer. */
	private static JsonObject loadPlan(CommandSourceStack src, Path input, Double scale) throws IOException, InterruptedException {
		if (Files.isRegularFile(input)) return AnalyzerRunner.readPlan(input);
		Path existing = input.resolve("plan.json");
		if (scale == null && AnalyzerRunner.planIsFresh(input, existing)) {
			info(src, "Using existing plan.json (delete it to re-analyse the photos).");
			return AnalyzerRunner.readPlan(existing);
		}
		info(src, "Analysing photos - the first run downloads the vision models and can take a while.");
		MinecraftServer server = src.getServer();
		return AnalyzerRunner.analyze(input, scale, line -> server.execute(() -> progress(src, line)));
	}

	private static void build(CommandSourceStack src, ServerLevel level, BlockPos at, Direction facing, JsonObject plan) {
		try {
			JsonObject size = plan.getAsJsonObject("size");
			int w = size.get("x").getAsInt(), d = size.get("z").getAsInt();
			// Put the house a few blocks in front of the player, centred on their line of sight.
			int ahead = 4 + (facing.getAxis() == Direction.Axis.X ? w : d) / 2;
			int cx = at.getX() + facing.getStepX() * ahead;
			int cz = at.getZ() + facing.getStepZ() * ahead;
			BlockPos origin = new BlockPos(cx - w / 2, at.getY() - 1, cz - d / 2);
			UndoLog log = new HouseBuilder(level, origin, plan).build();
			HISTORY.push(log);
			while (HISTORY.size() > MAX_UNDO) HISTORY.removeLast();
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

	private static int undo(CommandContext<CommandSourceStack> ctx) {
		UndoLog log = HISTORY.poll();
		if (log == null) {
			ctx.getSource().sendFailure(Component.literal("Nothing to undo."));
			return 0;
		}
		log.revert();
		ctx.getSource().sendSuccess(() -> Component.literal("Reverted " + log.size() + " blocks."), true);
		return 1;
	}

	private static void info(CommandSourceStack src, String msg) {
		src.getServer().execute(() -> src.sendSuccess(() -> Component.literal(msg).withStyle(ChatFormatting.GRAY), false));
	}

	private static void progress(CommandSourceStack src, String line) {
		src.sendSuccess(() -> Component.literal(line).withStyle(ChatFormatting.DARK_GRAY), false);
	}
}

package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.build.HouseBuilder;
import com.dubhacks.everything_cubed.build.UndoLog;
import com.dubhacks.everything_cubed.command.EverythingCubedCommand;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.BedBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BedPart;
import net.minecraft.world.phys.Vec3;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The game side of one spectate session, on a room that was scanned and built (the build is the starting state),
 * used on the server thread only. Puts the player into spectator mode (so nothing they do can disturb the house),
 * moves and adds whole pieces of furniture as the worker reports them from RoomPlan (never single blocks), and shows
 * the people the phone sees as villagers. {@link #finish()} gives the player their game mode and place back; the
 * changes stay and {@code /everything_cubed undo} reverts them.
 *
 * <p>The worker's coordinates are the plan's: objects are plan "objects" entries, people stand at plan block
 * coordinates from the build's origin.
 */
final class SpectateWorld {
	static final String TAG = "everything_cubed_spectate";
	private static final int OBJECTS_PER_TICK = 8;
	private static final double MIN_TURN = 0.05;

	private final ServerLevel level;
	private final ServerPlayer player;
	private final EverythingCubedCommand.Built build;
	private final GameType previousMode;
	private final Vec3 returnPos;
	private final float returnYaw, returnPitch;
	private final UndoLog undo;
	/** The plan the house was built from (HouseBuilder needs its wall and roof materials). */
	private final JsonObject plan;
	private final ArrayDeque<Change> pending = new ArrayDeque<>();
	private final Map<Integer, Villager> people = new HashMap<>();
	private int placed;

	/** A piece of furniture to take away ({@code from}, may be null) and to build ({@code to}, may be null). */
	private record Change(JsonObject from, JsonObject to) {
	}

	SpectateWorld(ServerPlayer player, EverythingCubedCommand.Built build) throws IOException {
		this.plan = AnalyzerRunner.readPlan(build.plan());
		this.player = player;
		this.build = build;
		this.level = build.level();
		this.previousMode = player.gameMode();
		this.returnPos = player.position();
		this.returnYaw = player.getYRot();
		this.returnPitch = player.getXRot();
		this.undo = new UndoLog(level);
		player.setGameMode(GameType.SPECTATOR);
	}

	ServerPlayer player() {
		return player;
	}

	UndoLog undo() {
		return undo;
	}

	int placed() {
		return placed;
	}

	/** Queues one worker update: {@code moves} [{from, to}], {@code adds} [object] and {@code people} [{id, x, y, z}]. */
	void update(JsonObject msg) {
		JsonArray moves = msg.getAsJsonArray("moves");
		if (moves != null)
			for (JsonElement e : moves)
				pending.add(new Change(e.getAsJsonObject().getAsJsonObject("from"), e.getAsJsonObject().getAsJsonObject("to")));
		JsonArray adds = msg.getAsJsonArray("adds");
		if (adds != null)
			for (JsonElement e : adds) pending.add(new Change(null, e.getAsJsonObject()));
		JsonArray ppl = msg.getAsJsonArray("people");
		if (ppl != null) people(ppl);
	}

	/** Moves or adds up to {@link #OBJECTS_PER_TICK} queued pieces of furniture. Called every server tick. */
	void tick() {
		for (int n = 0; n < OBJECTS_PER_TICK && !pending.isEmpty(); n++) {
			Change c = pending.poll();
			// A move takes the whole piece away and builds it again at its new place, as build would have.
			if (c.from() != null) HouseBuilder.removeObject(level, build.origin(), c.from(), undo);
			if (c.to() != null) HouseBuilder.placeObject(level, build.origin(), plan, c.to(), undo);
			placed++;
		}
	}

	private void people(JsonArray list) {
		Set<Integer> seen = new HashSet<>();
		BlockPos o = build.origin();
		for (JsonElement e : list) {
			JsonObject p = e.getAsJsonObject();
			int id = p.get("id").getAsInt();
			seen.add(id);
			Vec3 at = new Vec3(o.getX() + p.get("x").getAsDouble(), o.getY() + p.get("y").getAsDouble(), o.getZ() + p.get("z").getAsDouble());
			Villager v = people.get(id);
			if (v == null || v.isRemoved()) {
				v = new Villager(EntityTypes.VILLAGER, level);
				v.setNoAi(true);
				v.setSilent(true);
				v.setNoGravity(true);
				v.setPermanentlyInvulnerable(true);
				v.addTag(TAG);
				v.snapTo(at.x, at.y, at.z, 0, 0);
				level.addFreshEntity(v);
				people.put(id, v);
				EverythingCubed.LOGGER.info("[spectate] villager #{} appears at {} ({} villager(s) now)", id, at, people.size());
				continue;
			}
			if (sleep(id, v, p)) continue;
			Vec3 d = at.subtract(v.position());
			float yaw = v.getYRot();
			if (d.horizontalDistance() > MIN_TURN) yaw = (float) Math.toDegrees(Math.atan2(-d.x, d.z));
			v.snapTo(at.x, at.y, at.z, yaw, 0);
			v.setYHeadRot(yaw);
			v.setYBodyRot(yaw);
		}
		people.entrySet().removeIf(en -> {
			if (seen.contains(en.getKey())) return false;
			EverythingCubed.LOGGER.info("[spectate] villager #{} removed from {}", en.getKey(), en.getValue().position());
			remove(en.getValue());
			return true;
		});
	}

	/** Takes a villager away, waking it first so its bed is not left marked as occupied. */
	private static void remove(Villager v) {
		if (v.isSleeping()) v.stopSleeping();
		v.discard();
	}

	/**
	 * A person lying on a bed: the villager sleeps in that bed (its head block), and gets up when they do.
	 * Returns true while the villager sleeps, so it is not moved.
	 */
	private boolean sleep(int id, Villager v, JsonObject p) {
		JsonArray at = p.getAsJsonArray("sleep");
		BlockPos head = null;
		if (at != null) {
			BlockPos pos = build.origin().offset(at.get(0).getAsInt(), 1, at.get(1).getAsInt());
			BlockState bed = level.getBlockState(pos);
			if (bed.getBlock() instanceof BedBlock) {
				// Villagers sleep with their head on the bed's head block.
				head = bed.getValue(BedBlock.PART) == BedPart.FOOT ? pos.relative(bed.getValue(BedBlock.FACING)) : pos;
			} else if (noBedLogged.add(id)) {
				EverythingCubed.LOGGER.info("[spectate] person #{} lies on plan block ({}, {}) = {}, but that is {}, not a bed: not asleep",
						id, at.get(0).getAsInt(), at.get(1).getAsInt(), pos.toShortString(), bed);
			}
		}
		if (head == null) {
			if (v.isSleeping()) {
				v.stopSleeping();
				EverythingCubed.LOGGER.info("[spectate] villager #{} wakes, now at {}", id, v.position());
			}
			return false;
		}
		noBedLogged.remove(id);
		if (!v.isSleeping() || !v.getSleepingPos().map(head::equals).orElse(false)) {
			if (v.isSleeping()) v.stopSleeping();
			Vec3 before = v.position();
			v.startSleeping(head);
			EverythingCubed.LOGGER.info("[spectate] villager #{} sleeps in the bed with its head at {} ({}), house origin {}: "
							+ "moved from {} to {}, sleeping {}", id, head.toShortString(), level.getBlockState(head),
					build.origin().toShortString(), before, v.position(), v.isSleeping());
		}
		return true;
	}

	/** People already logged as lying where there is no bed (once each, until they are on one). */
	private final Set<Integer> noBedLogged = new HashSet<>();

	/** Ends spectating: the rest of the queued blocks are placed, villagers go, the player gets their mode back. */
	void finish() {
		while (!pending.isEmpty()) tick();
		people.values().forEach(SpectateWorld::remove);
		people.clear();
		if (!player.isRemoved()) {
			player.setGameMode(previousMode);
			player.teleportTo(level, returnPos.x, returnPos.y, returnPos.z, Set.of(), returnYaw, returnPitch, false);
		}
	}
}

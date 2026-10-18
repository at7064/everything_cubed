package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.build.FurnitureInteractions;
import com.dubhacks.everything_cubed.build.HouseBuilder;
import com.dubhacks.everything_cubed.build.UndoLog;
import com.dubhacks.everything_cubed.command.EverythingCubedCommand;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Display;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The game side of one spectate session, on a room that was scanned and built (the build is the starting state),
 * used on the server thread only. Puts the player into spectator mode (so nothing they do can disturb the mirrored
 * blocks), applies the worker's changes inside the house a few thousand per tick, and shows the people the phone
 * sees as villagers. {@link #finish()} gives the player their game mode and place back; the changes stay and
 * {@code /everything_cubed undo} reverts them.
 *
 * <p>The worker's coordinates are the plan's: interior blocks x in [0, W), z in [0, D), y in [1, H], from the
 * build's origin.
 */
final class SpectateWorld {
	static final String TAG = "everything_cubed_spectate";
	private static final int BLOCKS_PER_TICK = 4096;
	private static final double MIN_TURN = 0.05;

	private final ServerLevel level;
	private final ServerPlayer player;
	private final EverythingCubedCommand.Built build;
	private final GameType previousMode;
	private final Vec3 returnPos;
	private final float returnYaw, returnPitch;
	private final UndoLog undo;
	private final ArrayDeque<Change> pending = new ArrayDeque<>();
	private final Map<Integer, Villager> people = new HashMap<>();
	private final Map<String, BlockState> states = new HashMap<>();
	private int placed;

	private record Change(BlockPos pos, String block) {
	}

	SpectateWorld(ServerPlayer player, EverythingCubedCommand.Built build) {
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

	/**
	 * The blocks inside the house that hold something now, [[x, y, z], ...] in plan coordinates: solid blocks and
	 * blocks with a furniture model in them (a bin or a monitor has no real block). The worker starts from this.
	 */
	JsonArray baseline() {
		Set<BlockPos> solid = new HashSet<>();
		BlockPos o = build.origin();
		for (int x = 0; x < build.w(); x++)
			for (int y = 1; y <= build.h(); y++)
				for (int z = 0; z < build.d(); z++) {
					BlockState s = level.getBlockState(o.offset(x, y, z));
					if (!s.isAir() && !s.is(Blocks.LIGHT)) solid.add(new BlockPos(x, y, z));
				}
		AABB inside = new AABB(Vec3.atLowerCornerOf(o.offset(0, 1, 0)), Vec3.atLowerCornerOf(o.offset(build.w(), build.h() + 1, build.d())));
		for (Display e : level.getEntitiesOfClass(Display.class, inside)) {
			BlockPos p = e.blockPosition().subtract(o);
			if (p.getX() >= 0 && p.getX() < build.w() && p.getY() >= 1 && p.getY() <= build.h() && p.getZ() >= 0 && p.getZ() < build.d())
				solid.add(p);
		}
		JsonArray out = new JsonArray();
		for (BlockPos p : solid) {
			JsonArray c = new JsonArray();
			c.add(p.getX());
			c.add(p.getY());
			c.add(p.getZ());
			out.add(c);
		}
		return out;
	}

	/** Queues one worker update: {@code changes} [[x, y, z, block id], ...] and {@code people} [{id, x, y, z}]. */
	void update(JsonObject msg) {
		JsonArray changes = msg.getAsJsonArray("changes");
		if (changes != null)
			for (JsonElement e : changes) {
				JsonArray c = e.getAsJsonArray();
				pending.add(new Change(build.origin().offset(c.get(0).getAsInt(), c.get(1).getAsInt(), c.get(2).getAsInt()),
						c.get(3).getAsString()));
			}
		JsonArray ppl = msg.getAsJsonArray("people");
		if (ppl != null) people(ppl);
	}

	/** Places up to {@link #BLOCKS_PER_TICK} queued changes. Called every server tick. */
	void tick() {
		for (int n = 0; n < BLOCKS_PER_TICK && !pending.isEmpty(); n++) {
			Change c = pending.poll();
			boolean air = c.block().equals("minecraft:air");
			BlockState s = states.computeIfAbsent(c.block(), id -> air ? Blocks.AIR.defaultBlockState() : HouseBuilder.first(id));
			// A piece of furniture that is gone (or that something new now stands in) goes as a whole.
			FurnitureInteractions.removeModelsIn(level, c.pos());
			if (level.getBlockState(c.pos()) == s) continue;
			undo.remember(c.pos());
			level.setBlock(c.pos(), s, Block.UPDATE_CLIENTS);
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
				continue;
			}
			Vec3 d = at.subtract(v.position());
			float yaw = v.getYRot();
			if (d.horizontalDistance() > MIN_TURN) yaw = (float) Math.toDegrees(Math.atan2(-d.x, d.z));
			v.snapTo(at.x, at.y, at.z, yaw, 0);
			v.setYHeadRot(yaw);
			v.setYBodyRot(yaw);
		}
		people.entrySet().removeIf(en -> {
			if (seen.contains(en.getKey())) return false;
			en.getValue().discard();
			return true;
		});
	}

	/** Ends spectating: the rest of the queued blocks are placed, villagers go, the player gets their mode back. */
	void finish() {
		while (!pending.isEmpty()) tick();
		people.values().forEach(Villager::discard);
		people.clear();
		if (!player.isRemoved()) {
			player.setGameMode(previousMode);
			player.teleportTo(level, returnPos.x, returnPos.y, returnPos.z, Set.of(), returnYaw, returnPitch, false);
		}
	}
}

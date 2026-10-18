package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.build.HouseBuilder;
import com.dubhacks.everything_cubed.build.UndoLog;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The game side of one spectate session, used on the server thread only. Puts the player into spectator mode
 * (so nothing they do can disturb the mirrored blocks), applies the worker's block changes a few thousand per tick,
 * and shows the people the phone sees as villagers. {@link #finish()} gives the player their game mode and place
 * back; the mirrored blocks stay and {@code /everything_cubed undo} reverts them.
 *
 * <p>The worker's coordinates are blocks: y = 0 is the floor block, +z is where the phone looked first, +x is its
 * left. They are laid out from the player's position and facing when spectating started, so the phone's spot is
 * the player's spot and its view is theirs.
 */
final class SpectateWorld {
	static final String TAG = "everything_cubed_spectate";
	private static final int BLOCKS_PER_TICK = 4096;
	private static final double MIN_TURN = 0.05;

	private final ServerLevel level;
	private final ServerPlayer player;
	private final GameType previousMode;
	private final Vec3 returnPos;
	private final float returnYaw, returnPitch;
	private final double ox, oz;
	private final int floorY;
	private final Direction forward, left;
	private final UndoLog undo;
	private final ArrayDeque<Change> pending = new ArrayDeque<>();
	private final Map<Integer, Villager> people = new HashMap<>();
	private final Map<String, BlockState> states = new HashMap<>();
	private int placed;

	private record Change(BlockPos pos, String block) {
	}

	SpectateWorld(ServerPlayer player) {
		this.player = player;
		this.level = player.level();
		this.previousMode = player.gameMode();
		this.returnPos = player.position();
		this.returnYaw = player.getYRot();
		this.returnPitch = player.getXRot();
		this.ox = player.getX();
		this.oz = player.getZ();
		this.floorY = player.blockPosition().getY() - 1;   // like build: the floor is level with your feet
		this.forward = player.getDirection();
		this.left = forward.getCounterClockWise();          // facing south, east is on your left
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

	private Vec3 toWorld(double x, double y, double z) {
		return new Vec3(ox + forward.getStepX() * z + left.getStepX() * x, floorY + y,
				oz + forward.getStepZ() * z + left.getStepZ() * x);
	}

	/** Queues one worker update: {@code changes} [[x, y, z, block id], ...] and {@code people} [{id, x, y, z}]. */
	void update(JsonObject msg) {
		JsonArray changes = msg.getAsJsonArray("changes");
		if (changes != null)
			for (JsonElement e : changes) {
				JsonArray c = e.getAsJsonArray();
				Vec3 p = toWorld(c.get(0).getAsInt() + 0.5, c.get(1).getAsInt(), c.get(2).getAsInt() + 0.5);
				pending.add(new Change(BlockPos.containing(p), c.get(3).getAsString()));
			}
		JsonArray ppl = msg.getAsJsonArray("people");
		if (ppl != null) people(ppl);
	}

	/** Places up to {@link #BLOCKS_PER_TICK} queued changes. Called every server tick. */
	void tick() {
		for (int n = 0; n < BLOCKS_PER_TICK && !pending.isEmpty(); n++) {
			Change c = pending.poll();
			BlockState s = states.computeIfAbsent(c.block(),
					id -> id.equals("minecraft:air") ? Blocks.AIR.defaultBlockState() : HouseBuilder.first(id));
			if (level.getBlockState(c.pos()) == s) continue;
			undo.remember(c.pos());
			level.setBlock(c.pos(), s, Block.UPDATE_CLIENTS);
			placed++;
		}
	}

	private void people(JsonArray list) {
		Set<Integer> seen = new HashSet<>();
		for (JsonElement e : list) {
			JsonObject p = e.getAsJsonObject();
			int id = p.get("id").getAsInt();
			seen.add(id);
			Vec3 at = toWorld(p.get("x").getAsDouble(), p.get("y").getAsDouble(), p.get("z").getAsDouble());
			Villager v = people.get(id);
			if (v == null || v.isRemoved()) {
				v = new Villager(EntityTypes.VILLAGER, level);
				v.setNoAi(true);
				v.setSilent(true);
				v.setNoGravity(true);
				v.setPermanentlyInvulnerable(true);
				v.addTag(TAG);
				v.snapTo(at.x, at.y, at.z, player.getYRot() + 180, 0);   // facing the phone until they move
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

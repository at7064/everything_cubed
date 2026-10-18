package com.dubhacks.everything_cubed.build;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Remembers the original block states (and spawned entities) of a build so it can be reverted. */
public class UndoLog {
	private final ServerLevel level;
	private final Map<BlockPos, BlockState> original = new LinkedHashMap<>();
	private final List<Entity> entities = new ArrayList<>();

	public UndoLog(ServerLevel level) {
		this.level = level;
	}

	public void remember(BlockPos pos) {
		original.computeIfAbsent(pos.immutable(), level::getBlockState);
	}

	void rememberEntity(Entity e) {
		entities.add(e);
	}

	public int size() {
		return original.size();
	}

	/** Restores block states (block-entity contents placed by the build are dropped with them). */
	public void revert() {
		entities.forEach(e -> { if (!e.isRemoved()) e.discard(); });
		// Restore in reverse so two-block things (doors, beds) never see a half-built neighbour.
		List<Map.Entry<BlockPos, BlockState>> list = new ArrayList<>(original.entrySet());
		for (int i = list.size() - 1; i >= 0; i--)
			level.setBlock(list.get(i).getKey(), list.get(i).getValue(), Block.UPDATE_CLIENTS);
	}
}

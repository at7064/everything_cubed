package com.dubhacks.everything_cubed.build;

import net.fabricmc.fabric.api.event.player.AttackEntityCallback;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.fabricmc.fabric.api.event.player.UseEntityCallback;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.sounds.SoundEvents;
import net.minecraft.sounds.SoundSource;
import net.minecraft.util.Brightness;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.Display;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.Interaction;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.LightBlock;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.Vec3;

import java.util.List;
import java.util.UUID;

/**
 * Makes furniture models usable. Each piece of furniture is a group of entities sharing a group tag:
 * <ul>
 *   <li>right-click a seat (chair, sofa, toilet) to sit on it, sneak to get up;</li>
 *   <li>right-click a lamp to switch it on or off;</li>
 *   <li>right-click a trash can with something in your main hand to throw that stack away;</li>
 *   <li>punch a seat or lamp, or break the real block inside a model, to remove that piece of furniture.</li>
 * </ul>
 * Tags are saved with the entities, so this keeps working after a restart.
 */
public final class FurnitureInteractions {
	static final String SEAT = "everything_cubed_seat";
	static final String LAMP = "everything_cubed_lamp";
	static final String TRASH = "everything_cubed_trash";
	static final String GLOW = "everything_cubed_glow";
	private static final String GROUP = "everything_cubed_group:";
	/** On one display of a model, per real block inside it: breaking that block removes the model. */
	private static final String CORE = "everything_cubed_core:";
	/** How far from the used entity or block the rest of its group is searched for. */
	private static final double REACH = 16;

	private FurnitureInteractions() {
	}

	public static void register() {
		UseEntityCallback.EVENT.register(FurnitureInteractions::use);
		AttackEntityCallback.EVENT.register(FurnitureInteractions::attack);
		PlayerBlockBreakEvents.AFTER.register((level, player, pos, state, blockEntity) -> {
			if (level instanceof ServerLevel server) removeModelAround(server, pos);
		});
	}

	static String newGroup() {
		return GROUP + UUID.randomUUID();
	}

	static String coreTag(BlockPos pos) {
		return CORE + pos.asLong();
	}

	/** An invisible hitbox standing on {@code bottomCentre}. A seat's top is where the player sits. */
	static Interaction spawnHitbox(ServerLevel level, Vec3 bottomCentre, float width, float height, String kind, String group) {
		Interaction e = new Interaction(EntityTypes.INTERACTION, level);
		e.setPos(bottomCentre);
		e.setWidth(width);
		e.setHeight(height);
		e.setResponse(true); // swing the arm when used
		e.addTag(kind);
		e.addTag(group);
		level.addFreshEntity(e);
		return e;
	}

	/** The block a lamp hitbox lights: the one its shade is in. */
	static BlockPos lightPos(Interaction lamp) {
		return BlockPos.containing(lamp.getX(), lamp.getY() + lamp.getHeight() - 0.5, lamp.getZ());
	}

	private static InteractionResult use(Player player, Level level, InteractionHand hand, Entity entity, EntityHitResult hit) {
		// Tags only exist on the server; a client sees PASS and sends the click on to it.
		if (!(entity instanceof Interaction box) || !(level instanceof ServerLevel server)) return InteractionResult.PASS;
		if (box.entityTags().contains(SEAT)) {
			if (!box.isVehicle() && !player.isPassenger()) player.startRiding(box);
			return InteractionResult.SUCCESS;
		}
		if (box.entityTags().contains(LAMP)) {
			toggleLamp(server, box);
			return InteractionResult.SUCCESS;
		}
		if (box.entityTags().contains(TRASH)) {
			// Main hand only: an empty main hand must not throw away a shield or totem held in the off hand.
			if (hand != InteractionHand.MAIN_HAND) return InteractionResult.PASS;
			ItemStack held = player.getItemInHand(hand);
			if (held.isEmpty()) return InteractionResult.SUCCESS;
			player.setItemInHand(hand, ItemStack.EMPTY);
			server.playSound(null, box.blockPosition(), SoundEvents.COMPOSTER_FILL, SoundSource.BLOCKS, 1f, 1f);
			return InteractionResult.SUCCESS;
		}
		return InteractionResult.PASS;
	}

	private static InteractionResult attack(Player player, Level level, InteractionHand hand, Entity entity, EntityHitResult hit) {
		if (!(entity instanceof Interaction box) || !(level instanceof ServerLevel server)) return InteractionResult.PASS;
		String group = groupOf(box);
		if (group == null || !player.mayBuild()) return InteractionResult.PASS;
		removeGroup(server, group, box.position());
		return InteractionResult.SUCCESS;
	}

	private static void toggleLamp(ServerLevel level, Interaction lamp) {
		BlockPos pos = lightPos(lamp);
		boolean on = level.getBlockState(pos).is(Blocks.LIGHT);
		if (on) level.removeBlock(pos, false);
		else if (level.getBlockState(pos).isAir())
			level.setBlockAndUpdate(pos, Blocks.LIGHT.defaultBlockState().setValue(LightBlock.LEVEL, LightBlock.MAX_LEVEL));
		for (Entity e : members(level, groupOf(lamp), lamp.position()))
			if (e instanceof Display shade && e.entityTags().contains(GLOW))
				shade.setBrightnessOverride(on ? null : Brightness.FULL_BRIGHT);
	}

	/** Removes the model (if any) that had the real block at {@code pos} inside it. */
	private static void removeModelAround(ServerLevel level, BlockPos pos) {
		String core = coreTag(pos);
		for (Entity e : level.getEntitiesOfClass(Display.class, new AABB(pos).inflate(REACH), e -> e.entityTags().contains(core))) {
			String group = groupOf(e);
			if (group != null) removeGroup(level, group, e.position());
		}
	}

	private static void removeGroup(ServerLevel level, String group, Vec3 near) {
		for (Entity e : members(level, group, near)) {
			if (e instanceof Interaction lamp && e.entityTags().contains(LAMP) && level.getBlockState(lightPos(lamp)).is(Blocks.LIGHT))
				level.removeBlock(lightPos(lamp), false);
			e.ejectPassengers();
			e.discard();
		}
	}

	private static Iterable<Entity> members(ServerLevel level, String group, Vec3 near) {
		if (group == null) return List.of();
		return level.getEntitiesOfClass(Entity.class, new AABB(near, near).inflate(REACH), e -> e.entityTags().contains(group));
	}

	private static String groupOf(Entity e) {
		for (String tag : e.entityTags()) if (tag.startsWith(GROUP)) return tag;
		return null;
	}
}

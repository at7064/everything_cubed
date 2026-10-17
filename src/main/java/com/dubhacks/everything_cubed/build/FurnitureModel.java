package com.dubhacks.everything_cubed.build;

import com.mojang.math.Transformation;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.util.Brightness;
import net.minecraft.world.entity.Display;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;
import org.joml.Quaternionf;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

/**
 * A detailed furniture model made of scaled block displays (legs, drawers, cushions, ...).
 * Display entities have no hitbox, so whatever players use is a real block hidden inside the model, or an
 * interaction entity (see {@link FurnitureInteractions}).
 *
 * <p>Parts are boxes in the object's own frame, in blocks: x across its front (left to right as seen from the
 * front), z from its back (0) to its front ({@link #d}), y up from the floor.
 */
final class FurnitureModel {
	private record Part(double x0, double y0, double z0, double x1, double y1, double z1, BlockState block, boolean glow) {}

	/** Size in the model's own frame. */
	final double w, d, h;
	private final int x0, z0, sizeX, sizeZ;
	private final Direction front;
	private final List<Part> parts = new ArrayList<>();

	/** Footprint is house-relative cells x0..x0+sizeX-1 by z0..z0+sizeZ-1; {@code front} is the way the front faces. */
	FurnitureModel(int x0, int z0, int sizeX, int sizeZ, double h, Direction front) {
		this.x0 = x0;
		this.z0 = z0;
		this.sizeX = sizeX;
		this.sizeZ = sizeZ;
		this.h = h;
		this.front = front;
		boolean frontAlongZ = front.getAxis() == Direction.Axis.Z;
		this.w = frontAlongZ ? sizeX : sizeZ;
		this.d = frontAlongZ ? sizeZ : sizeX;
	}

	void box(double x0, double y0, double z0, double x1, double y1, double z1, BlockState block) {
		parts.add(new Part(x0, y0, z0, x1, y1, z1, block, false));
	}

	/** A part that renders at full brightness (a lit lamp shade). */
	void glow(double x0, double y0, double z0, double x1, double y1, double z1, BlockState block) {
		parts.add(new Part(x0, y0, z0, x1, y1, z1, block, true));
	}

	/** House-relative x/z of the cell at whole-block position (lx, lz) of the model frame. */
	int[] cell(int lx, int lz) {
		double[] p = toHouse(lx + 0.5, lz + 0.5);
		return new int[]{(int) Math.floor(p[0]), (int) Math.floor(p[1])};
	}

	/** House-relative x/z of a point in the model frame. */
	private double[] toHouse(double lx, double lz) {
		return switch (front) {
			case NORTH -> new double[]{x0 + sizeX - lx, z0 + sizeZ - lz};
			case EAST -> new double[]{x0 + lz, z0 + sizeZ - lx};
			case WEST -> new double[]{x0 + sizeX - lz, z0 + lx};
			default -> new double[]{x0 + lx, z0 + lz};
		};
	}

	/** Spawns every part as a block display carrying {@code tags}; {@code origin} is the house origin. */
	List<Display.BlockDisplay> spawn(ServerLevel level, BlockPos origin, String... tags) {
		// A display is lit by the light where the entity stands, and inside a solid core block that is
		// darkness. So every part stands in the air just above the model and is moved into place by its
		// transformation.
		double[] c = toHouse(w / 2, d / 2);
		Vec3 probe = new Vec3(origin.getX() + c[0], origin.getY() + 1 + h + 0.2, origin.getZ() + c[1]);
		List<Display.BlockDisplay> out = new ArrayList<>();
		for (int i = 0; i < parts.size(); i++) {
			Part p = parts.get(i);
			double eps = 0.002 * (1 + i % 4); // so flush faces of neighbouring parts don't z-fight
			double[] a = toHouse(p.x0, p.z0), b = toHouse(p.x1, p.z1);
			Vector3f min = new Vector3f(
					(float) (origin.getX() + Math.min(a[0], b[0]) - eps - probe.x),
					(float) (origin.getY() + 1 + p.y0 - eps - probe.y),
					(float) (origin.getZ() + Math.min(a[1], b[1]) - eps - probe.z));
			Vector3f size = new Vector3f(
					(float) (Math.abs(b[0] - a[0]) + 2 * eps),
					(float) (p.y1 - p.y0 + 2 * eps),
					(float) (Math.abs(b[1] - a[1]) + 2 * eps));

			Display.BlockDisplay e = new Display.BlockDisplay(EntityTypes.BLOCK_DISPLAY, level);
			e.setPos(probe);
			e.setBlockState(p.block);
			e.setTransformation(new Transformation(min, new Quaternionf(), size, new Quaternionf()));
			for (String tag : tags) e.addTag(tag);
			if (p.glow) {
				e.setBrightnessOverride(Brightness.FULL_BRIGHT);
				e.addTag(FurnitureInteractions.GLOW);
			}
			level.addFreshEntity(e);
			out.add(e);
		}
		return out;
	}
}

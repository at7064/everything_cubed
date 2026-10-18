package com.dubhacks.everything_cubed.build;

import net.minecraft.world.level.block.state.BlockState;

import static com.dubhacks.everything_cubed.build.HouseBuilder.first;

/**
 * The look of each piece of furniture, as parts of a {@link FurnitureModel} (sizes in blocks).
 * Where a real block does the work (a barrel, a bed, a furnace, ...), the model fully encloses that block so
 * only the model is seen; clicks pass through displays and still reach the block.
 */
final class FurnitureDesigns {
	private static final BlockState IRON = first("iron_block");
	private static final BlockState WHITE = first("white_concrete");
	private static final BlockState PORCELAIN = first("smooth_quartz");
	private static final BlockState LINEN = first("white_wool");
	private static final BlockState BLACK = first("black_concrete");
	private static final BlockState STEEL = first("light_gray_concrete");
	private static final BlockState DARK_STEEL = first("gray_concrete");
	private static final BlockState COUNTERTOP = first("polished_andesite");
	private static final BlockState GOLD = first("gold_block");
	private static final BlockState AMBER = first("shroomlight", "glowstone");

	/** Height of a vanilla bed's mattress. */
	private static final double BED_TOP = 9 / 16.0;

	private FurnitureDesigns() {
	}

	static BlockState planks(String wood) {
		return first(wood + "_planks", "oak_planks");
	}

	static BlockState strippedLog(String wood) {
		return first("stripped_" + wood + "_log", "stripped_" + wood + "_stem", "stripped_oak_log");
	}

	// ------------------------------------------------------------------------------------- around real blocks

	/** Frame, mattress, headboard, pillows and duvet over one or more beds side by side (head at the back). */
	static void bed(FurnitureModel m, BlockState frame, BlockState duvet) {
		double w = m.w, d = m.d;
		m.box(0, 0, 0, w, 0.3, d, frame);
		m.box(0, 0.3, 0, w, BED_TOP, d, LINEN);
		m.box(0, 0, 0, w, 1.15, 0.12, frame); // headboard
		for (int i = 0; i < (int) w; i++)
			m.box(i + 0.12, BED_TOP, 0.18, i + 0.88, BED_TOP + 0.14, 0.55, LINEN);
		m.box(-0.03, BED_TOP - 0.2, d * 0.4, w + 0.03, BED_TOP + 0.05, d + 0.03, duvet); // draped over the sides
		m.box(0.03, BED_TOP + 0.05, d * 0.4, w - 0.03, BED_TOP + 0.08, d * 0.4 + 0.3, LINEN); // folded edge
	}

	/** Upholstered sofa around a row of stairs: base, backrest, arm rests and one cushion per seat. */
	static void sofa(FurnitureModel m, BlockState fabric) {
		double w = m.w, d = m.d, arm = 0.18;
		m.box(0, 0, 0, w, 0.5, d, fabric);
		m.box(0, 0.5, 0, w, 1, d / 2, fabric);
		m.box(0, 0.5, 0, arm, 0.78, d, fabric);
		m.box(w - arm, 0.5, 0, w, 0.78, d, fabric);
		int seats = Math.max(1, (int) Math.round(w - 2 * arm));
		double cw = (w - 2 * arm) / seats;
		for (int i = 0; i < seats; i++) {
			double x = arm + i * cw;
			m.box(x + 0.02, 0.5, d / 2, x + cw - 0.02, 0.6, d - 0.02, fabric);
			m.box(x + 0.03, 0.55, d / 2 - 0.14, x + cw - 0.03, 0.95, d / 2, fabric);
		}
	}

	/** Legs, a skirt hiding the top slab, and an overhanging top. With {@code pedestal}, a drawer unit on the right. */
	static void table(FurnitureModel m, BlockState top, BlockState frame, boolean pedestal) {
		double w = m.w, d = m.d;
		legs(m, 0.5, 0.14, frame);
		m.box(0, 0.5, 0, w, 0.86, d, frame);
		m.box(-0.05, 0.86, -0.05, w + 0.05, 1, d + 0.05, top);
		if (pedestal) {
			m.box(w - 1, 0, 0, w, 0.86, d, top);
			drawers(m, w - 0.92, w - 0.08, 0.06, 0.8, 2, frame);
		}
	}

	/** Low table on a bottom slab. */
	static void coffeeTable(FurnitureModel m, BlockState top, BlockState base) {
		m.box(0, 0, 0, m.w, 0.4, m.d, base);
		m.box(-0.05, 0.4, -0.05, m.w + 0.05, 0.5, m.d + 0.05, top);
	}

	/** Chest of drawers around barrels: a column of drawers with knobs per block. */
	static void cabinet(FurnitureModel m, BlockState body, BlockState fronts) {
		double w = m.w, d = m.d, h = m.h;
		m.box(0, 0, 0, w, h, d, body);
		m.box(-0.03, h - 0.06, -0.03, w + 0.03, h, d + 0.03, fronts);
		int n = Math.max(2, (int) Math.round(h * 2));
		for (int c = 0; c < (int) w; c++) drawers(m, c + 0.06, c + 0.94, 0.06, h - 0.1, n, fronts);
	}

	/** Wardrobe around barrels: two tall doors with handles and a cornice. */
	static void wardrobe(FurnitureModel m, BlockState body, BlockState doors) {
		double w = m.w, d = m.d, h = m.h;
		m.box(0, 0, 0, w, h, d, body);
		m.box(0.05, 0.08, d, w / 2 - 0.02, h - 0.12, d + 0.04, doors);
		m.box(w / 2 + 0.02, 0.08, d, w - 0.05, h - 0.12, d + 0.04, doors);
		m.box(w / 2 - 0.12, h * 0.4, d + 0.04, w / 2 - 0.07, h * 0.6, d + 0.08, IRON);
		m.box(w / 2 + 0.07, h * 0.4, d + 0.04, w / 2 + 0.12, h * 0.6, d + 0.08, IRON);
		m.box(-0.04, h - 0.08, -0.04, w + 0.04, h, d + 0.06, body);
	}

	/** Washing machine around barrels: control strip and a round porthole door per block. */
	static void washer(FurnitureModel m) {
		double w = m.w, d = m.d, h = m.h, cy = Math.min(h, 1) * 0.4;
		m.box(0, 0, 0, w, h, d, WHITE);
		m.box(0.06, h - 0.16, d, w - 0.06, h - 0.05, d + 0.02, STEEL);
		for (int c = 0; c < (int) w; c++) {
			double cx = c + 0.5;
			m.box(cx - 0.32, cy - 0.32, d, cx + 0.32, cy + 0.32, d + 0.03, STEEL);
			m.box(cx - 0.24, cy - 0.24, d + 0.03, cx + 0.24, cy + 0.24, d + 0.05, DARK_STEEL);
		}
	}

	/** Newly Added Furniture */
	/** Podium: a slanted reading stand for presentations with a base and angled top. */
	static void podium(FurnitureModel m, BlockState wood) {

		double w = m.w, d = m.d;

		double cx = w / 2.0;

		// Central vertical pillar/base (using dynamic depth 'd')

		m.box(cx - 0.25, 0, d * 0.25, cx + 0.25, 0.9, d * 0.75, wood);

		// Flat base plate on the floor for stability

		m.box(cx - 0.35, 0, d * 0.15, cx + 0.35, 0.08, d * 0.85, DARK_STEEL);

		// Slanted reading surface / desktop on top

		m.box(cx - 0.35, 0.9, d * 0.15, cx + 0.35, 1.05, d * 0.85, wood);

		// Small lip/ledge at the bottom of the slant so notes don't slide off

		m.box(cx - 0.35, 0.92, d * 0.12, cx + 0.35, 1.02, d * 0.18, IRON);

	}

/** Newly Added Furniture */
	/** Laptop on a desk: base, open screen, and keyboard detailing. */

	static void laptop(FurnitureModel m) {

		double w = m.w;

		// Center the laptop nicely on whatever width the desk/surface has

		double cx = w / 2.0;

		// Laptop base (keyboard tray) lying flat on the surface

		m.box(cx - 0.28, 0.82, 0.3, cx + 0.28, 0.85, 0.65, STEEL);
		
		// Open screen angled upward at the back edge

		m.box(cx - 0.28, 0.85, 0.3, cx + 0.28, 1.22, 0.34, BLACK);
	}

/** Newly Added Furniture */
	/**A whiteboard on a wall */

	static void whiteboard(FurnitureModel m) {

		double w = m.w, h = m.h;

		// having the whiteboard on a wall with z = 0
		m.box(0, 0, 0, w, h, 0.04, STEEL);

		// White writing surface inside the frame

		m.box(0.04, 0.04, 0.02, w - 0.04, h - 0.04, 0.05, WHITE);

		// Marker tray running along the bottom edge

		m.box(0.1, 0.04, 0.05, w - 0.1, 0.1, 0.18, DARK_STEEL);

	}

/** Newly Added Furniture */
	/** Modern indoor plant pot with foliage. */

	static void pottedPlant(FurnitureModel m) {

		// Ceramic pot base

		m.box(0.3, 0, 0.3, 0.7, 0.4, 0.7, WHITE);

		// Soil

		m.box(0.32, 0.35, 0.32, 0.68, 0.4, 0.68, BLACK);

		// Plant leaves / greenery

		m.box(0.2, 0.4, 0.2, 0.8, 1.1, 0.8, planks("oak"));

	}

	/** Kitchen counter around barrels: cupboard doors with handles under a stone worktop. */
	static void counter(FurnitureModel m, BlockState body, BlockState doors) {
		double w = m.w, d = m.d;
		m.box(0, 0, 0, w, 0.92, d, body);
		m.box(-0.03, 0.92, -0.03, w + 0.03, 1, d + 0.05, COUNTERTOP);
		for (int c = 0; c < (int) w; c++) cupboardDoor(m, c, d, doors);
	}

	/** Fridge around stacked shulker boxes: freezer and fridge doors with handles. */
	static void fridge(FurnitureModel m) {
		double w = m.w, d = m.d, h = m.h, split = h * 0.62;
		m.box(0, 0, 0, w, h, d, WHITE);
		m.box(0.03, 0.04, d, w - 0.03, split - 0.02, d + 0.04, PORCELAIN);
		m.box(0.03, split + 0.02, d, w - 0.03, h - 0.04, d + 0.04, PORCELAIN);
		m.box(w - 0.16, split - 0.5, d + 0.04, w - 0.11, split - 0.08, d + 0.1, IRON);
		m.box(w - 0.16, split + 0.08, d + 0.04, w - 0.11, split + 0.45, d + 0.1, IRON);
	}

	/** Cooker around furnaces: four burners on top, oven door, handle and knobs per block. */
	static void stove(FurnitureModel m) {
		double w = m.w, d = m.d;
		m.box(0, 0, 0, w, 0.92, d, STEEL);
		m.box(-0.01, 0.92, -0.01, w + 0.01, 1, d + 0.01, BLACK);
		for (int c = 0; c < (int) w; c++) {
			for (double bx : new double[]{0.28, 0.72})
				for (double bz : new double[]{0.3, 0.7})
					m.box(c + bx - 0.12, 1, d * bz - 0.12, c + bx + 0.12, 1.02, d * bz + 0.12, DARK_STEEL);
			m.box(c + 0.08, 0.1, d, c + 0.92, 0.72, d + 0.03, BLACK);
			m.box(c + 0.14, 0.76, d, c + 0.86, 0.8, d + 0.07, IRON);
			for (int k = 0; k < 4; k++) {
				double kx = c + 0.2 + k * 0.2;
				m.box(kx - 0.03, 0.84, d, kx + 0.03, 0.89, d + 0.04, BLACK);
			}
		}
	}

	/** Microwave around a smoker: dark window and a control panel. */
	static void microwave(FurnitureModel m) {
		double d = m.d;
		m.box(0, 0, 0, m.w, m.h, d, WHITE);
		m.box(0.08, 0.18, d, 0.68, 0.82, d + 0.02, BLACK);
		m.box(0.74, 0.18, d, 0.92, 0.82, d + 0.02, DARK_STEEL);
	}

	/** Sink cabinet around water cauldrons: the water stays visible as the basin. */
	static void sink(FurnitureModel m, BlockState body, BlockState doors) {
		basin(m, body, COUNTERTOP, 0.03);
		for (int c = 0; c < (int) m.w; c++) cupboardDoor(m, c, m.d + 0.03, doors);
		tap(m);
	}

	/** Bathtub around water cauldrons. */
	static void bathtub(FurnitureModel m) {
		basin(m, PORCELAIN, PORCELAIN, 0.06);
		tap(m);
	}

	/** Side panels and a top around chiseled bookshelves, leaving their (usable) book slots in view. */
	static void bookshelf(FurnitureModel m, BlockState body) {
		double w = m.w, d = m.d, h = m.h;
		m.box(-0.08, 0, 0, 0, h, d + 0.03, body);
		m.box(w, 0, 0, w + 0.08, h, d + 0.03, body);
		m.box(-0.08, h, 0, w + 0.08, h + 0.08, d + 0.03, body);
	}

	// ------------------------------------------------------------------------------------- display-only

	/** Wooden chair with legs and a backrest. */
	static void chair(FurnitureModel m, BlockState seat, BlockState frame) {
		legs(m, 0.46, 0.1, frame);
		m.box(0.06, 0.44, 0.06, 0.94, 0.52, 0.94, seat);
		m.box(0.1, 0.52, 0.06, 0.2, 1.15, 0.16, frame);
		m.box(0.8, 0.52, 0.06, 0.9, 1.15, 0.16, frame);
		m.box(0.1, 0.8, 0.07, 0.9, 1.1, 0.15, seat);
	}

	/** Toilet: bowl, seat, cistern and flush button. */
	static void toilet(FurnitureModel m) {
		m.box(0.28, 0, 0.32, 0.72, 0.38, 0.92, PORCELAIN);
		m.box(0.22, 0.38, 0.28, 0.78, 0.44, 0.98, WHITE);
		m.box(0.15, 0.38, 0, 0.85, 0.95, 0.24, PORCELAIN);
		m.box(0.4, 0.95, 0.06, 0.6, 0.98, 0.18, IRON);
	}

	/** Floor lamp: base, pole and a glowing shade at the top. */
	static void lamp(FurnitureModel m) {
		double h = m.h;
		m.box(0.32, 0, 0.32, 0.68, 0.05, 0.68, IRON);
		m.box(0.47, 0.05, 0.47, 0.53, h - 0.5, 0.53, IRON);
		m.glow(0.26, h - 0.55, 0.26, 0.74, h - 0.1, 0.74, LINEN);
	}

	/**
	 * Chandelier: a lantern body between heights {@code y0} and {@code y1} (blocks above the floor), hung from
	 * the ceiling at {@code top} on a rod. Dark iron frame and crown, gold bands, a glowing amber core.
	 */
	static void chandelier(FurnitureModel m, double y0, double y1, double top) {
		double w = m.w, d = m.d, cx = w / 2, cz = d / 2, in = 0.15, post = 0.08;
		if (top - y1 > 0.05) {
			m.box(cx - 0.04, y1, cz - 0.04, cx + 0.04, top, cz + 0.04, BLACK);       // rod
			m.box(cx - 0.2, top - 0.06, cz - 0.2, cx + 0.2, top, cz + 0.2, BLACK);   // ceiling rose
		}
		m.box(in + 0.1, y1 - 0.12, in + 0.1, w - in - 0.1, y1, d - in - 0.1, BLACK);  // crown
		m.box(in, y1 - 0.2, in, w - in, y1 - 0.12, d - in, GOLD);                     // upper rim
		for (double[] c : new double[][]{{in, in}, {w - in - post, in}, {in, d - in - post}, {w - in - post, d - in - post}})
			m.box(c[0], y0 + 0.15, c[1], c[0] + post, y1 - 0.2, c[1] + post, BLACK);  // frame posts
		m.glow(in + 0.08, y0 + 0.2, in + 0.08, w - in - 0.08, y1 - 0.22, d - in - 0.08, AMBER);
		double mid = (y0 + y1) / 2;
		m.box(in - 0.02, mid - 0.03, in - 0.02, w - in + 0.02, mid + 0.03, d - in + 0.02, GOLD); // middle band
		m.box(in + 0.05, y0 + 0.08, in + 0.05, w - in - 0.05, y0 + 0.15, d - in - 0.05, GOLD); // bowl
		m.box(cx - 0.08, y0, cz - 0.08, cx + 0.08, y0 + 0.08, cz + 0.08, BLACK);     // finial
	}

	// ------------------------------------------------------------------------------------- parts

	private static void legs(FurnitureModel m, double top, double t, BlockState block) {
		double w = m.w, d = m.d, in = 0.06;
		m.box(in, 0, in, in + t, top, in + t, block);
		m.box(w - in - t, 0, in, w - in, top, in + t, block);
		m.box(in, 0, d - in - t, in + t, top, d - in, block);
		m.box(w - in - t, 0, d - in - t, w - in, top, d - in, block);
	}

	/** {@code n} drawer fronts with knobs stacked between y0 and y1 on the front face. */
	private static void drawers(FurnitureModel m, double x0, double x1, double y0, double y1, int n, BlockState face) {
		double d = m.d, dh = (y1 - y0) / n, cx = (x0 + x1) / 2;
		for (int i = 0; i < n; i++) {
			double a = y0 + i * dh + 0.02, b = y0 + (i + 1) * dh - 0.02, cy = (a + b) / 2;
			m.box(x0, a, d, x1, b, d + 0.04, face);
			m.box(cx - 0.07, cy - 0.025, d + 0.04, cx + 0.07, cy + 0.025, d + 0.08, IRON);
		}
	}

	/** A cupboard door with a handle on the front face (at {@code front}) of block column {@code c}. */
	private static void cupboardDoor(FurnitureModel m, int c, double front, BlockState door) {
		m.box(c + 0.06, 0.1, front, c + 0.94, 0.82, front + 0.03, door);
		m.box(c + 0.4, 0.68, front + 0.03, c + 0.6, 0.72, front + 0.07, IRON);
	}

	/**
	 * Walls of thickness {@code t} around cauldrons and a rim over their edges (also between neighbouring
	 * cauldrons), so only the water shows from above.
	 */
	private static void basin(FurnitureModel m, BlockState walls, BlockState rim, double t) {
		double w = m.w, d = m.d, r = 0.14, top = 1.04;
		m.box(0, 0, -t, w, 1, 0, walls);
		m.box(0, 0, d, w, 1, d + t, walls);
		m.box(-t, 0, -t, 0, 1, d + t, walls);
		m.box(w, 0, -t, w + t, 1, d + t, walls);
		m.box(-t, 1, -t, w + t, top, r, rim);
		m.box(-t, 1, d - r, w + t, top, d + t, rim);
		m.box(-t, 1, r, r, top, d - r, rim);
		m.box(w - r, 1, r, w + t, top, d - r, rim);
		for (int c = 1; c < (int) w; c++) m.box(c - r, 1, r, c + r, top, d - r, rim);
		for (int c = 1; c < (int) d; c++) m.box(r, 1, c - r, w - r, top, c + r, rim);
	}

	/** A tap at the middle of the back edge. */
	private static void tap(FurnitureModel m) {
		double cx = m.w / 2;
		m.box(cx - 0.04, 1.04, 0, cx + 0.04, 1.4, 0.08, IRON);
		m.box(cx - 0.04, 1.32, 0, cx + 0.04, 1.4, 0.32, IRON);
	}
}

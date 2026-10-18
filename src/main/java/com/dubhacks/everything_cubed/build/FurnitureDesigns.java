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
	private static final BlockState RED = first("red_concrete");
	private static final BlockState EXIT_GREEN = first("lime_concrete");

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
		bed(m, 0, frame, duvet);
	}

	/** {@link #bed} with its frame's bottom at {@code y} (a raised bed's base). */
	private static void bed(FurnitureModel m, double y, BlockState frame, BlockState duvet) {
		double w = m.w, d = m.d;
		m.box(0, y, 0, w, y + 0.3, d, frame);
		m.box(0, y + 0.3, 0, w, y + BED_TOP, d, LINEN);
		m.box(0, y, 0, w, y + 1.15, 0.12, frame); // headboard
		for (int i = 0; i < (int) w; i++)
			m.box(i + 0.12, y + BED_TOP, 0.18, i + 0.88, y + BED_TOP + 0.14, 0.55, LINEN);
		m.box(-0.03, y + BED_TOP - 0.2, d * 0.4, w + 0.03, y + BED_TOP + 0.05, d + 0.03, duvet); // draped over the sides
		m.box(0.03, y + BED_TOP + 0.05, d * 0.4, w - 0.03, y + BED_TOP + 0.08, d * 0.4 + 0.3, LINEN); // folded edge
	}

	/** Height of a raised bed's head posts above its top bed's base. */
	static final double RAISED_POST = 1.15;

	/**
	 * A bunk bed (beds at every level in {@code levels}, model y of each base) or a loft bed (one raised level, open
	 * underneath), for one or more beds side by side: a post at every corner, guard rails along the top bed's sides
	 * and a low footboard, a reading lamp, a caddy and fairy lights. The ladder past each foot is real ladder blocks
	 * (see HouseBuilder); the model only draws its rungs below {@code rungsTo} (model y), where a ladder block would
	 * lean on the lower bed.
	 */
	static void raisedBed(FurnitureModel m, double[] levels, BlockState frame, BlockState duvet, double rungsTo) {
		double w = m.w, d = m.d, p = 0.12, top = levels[levels.length - 1];
		for (double y : levels) bed(m, y, frame, duvet);
		for (int i = 0; i <= (int) w; i++) {
			double x = Math.min(Math.max(i - p / 2, 0), w - p);
			m.box(x, 0, 0, x + p, top + RAISED_POST, p, frame);   // head posts, as high as the headboard
			m.box(x, 0, d - p, x + p, top + 0.9, d, frame);        // foot posts
		}
		// Guard rails on both long sides of the top bed, and a low footboard to step over from the ladder.
		m.box(-0.02, top + 0.75, 0, 0.06, top + 0.85, d, frame);
		m.box(w - 0.06, top + 0.75, 0, w + 0.02, top + 0.85, d, frame);
		m.box(0, top, d - 0.08, w, top + 0.35, d, frame);
		// Dorm touches on the top bed: a clip-on reading lamp on a head post, a caddy hanging on one rail with a book
		// in it, and fairy lights along the other rail.
		m.box(w - 0.14, top + 1.0, 0.04, w - 0.06, top + 1.06, 0.3, DARK_STEEL);   // clamp arm
		m.glow(w - 0.22, top + 0.86, 0.24, w - 0.02, top + 1.02, 0.42, LINEN);     // shade
		m.box(w + 0.02, top + 0.35, 0.55, w + 0.12, top + 0.85, Math.min(d - 0.2, 1.05), DARK_STEEL);
		m.box(w + 0.04, top + 0.6, 0.62, w + 0.1, top + 0.95, 0.72, RED);
		for (int k = 0; 0.15 + k * 0.22 < d - 0.1; k++) {
			double z = 0.15 + k * 0.22;
			m.glow(-0.06, top + 0.83, z, -0.01, top + 0.88, z + 0.05, k % 2 == 0 ? AMBER : LINEN);
		}
		for (int i = 0; i < (int) w && rungsTo > 0; i++) {
			m.box(i + 0.2, 0, d, i + 0.26, rungsTo, d + 0.06, frame);
			m.box(i + 0.74, 0, d, i + 0.8, rungsTo, d + 0.06, frame);
			for (double y = 0.25; y < rungsTo; y += 0.3) m.box(i + 0.26, y, d + 0.01, i + 0.74, y + 0.05, d + 0.05, frame);
		}
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
		m.box(cx - 0.22, 1.04, d * 0.32, cx + 0.22, 1.32, d * 0.36, BLACK);
		m.box(cx - 0.19, 1.07, d * 0.36, cx + 0.19, 1.29, d * 0.37, first("cyan_concrete"));

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

	/** Whiteboard (or chalkboard) on the wall behind it, from height y0 to y1: frame, writing surface, marker tray. */
	static void whiteboard(FurnitureModel m, double y0, double y1, BlockState surface) {
		double w = m.w;
		m.box(0, y0, 0, w, y1, 0.04, STEEL);                                  // aluminium frame
		m.box(0.05, y0 + 0.05, 0.04, w - 0.05, y1 - 0.05, 0.06, surface);     // writing surface
		m.box(0.05, y0 + 0.02, 0.04, w - 0.05, y0 + 0.06, 0.14, STEEL);       // marker tray
	}

	/** Exit sign high on the wall in row y0 (blocks above the floor): white housing, glowing green face. */
	static void exitSign(FurnitureModel m, double y0) {
		m.box(0.15, y0 + 0.55, 0, 0.85, y0 + 0.92, 0.08, WHITE);
		m.glow(0.2, y0 + 0.6, 0.08, 0.8, y0 + 0.87, 0.1, EXIT_GREEN);
	}

	/** Fire extinguisher on a wall bracket in row y0: red cylinder, valve, handle and hose. */
	static void fireExtinguisher(FurnitureModel m, double y0) {
		m.box(0.38, y0 + 0.3, 0, 0.62, y0 + 0.36, 0.08, DARK_STEEL);         // bracket
		m.box(0.4, y0 + 0.12, 0.06, 0.6, y0 + 0.74, 0.26, RED);               // cylinder
		m.box(0.45, y0 + 0.74, 0.11, 0.55, y0 + 0.81, 0.21, DARK_STEEL);      // valve
		m.box(0.43, y0 + 0.81, 0.1, 0.63, y0 + 0.85, 0.22, BLACK);            // handle
		m.box(0.6, y0 + 0.3, 0.14, 0.64, y0 + 0.8, 0.18, BLACK);              // hose
	}

	/** Ceiling projector in row y0 hanging from the ceiling at {@code top}, lens on its front. */
	static void projector(FurnitureModel m, double y0, double top) {
		double b0 = y0 + 0.55, b1 = y0 + 0.78;
		if (top - b1 > 0.05) m.box(0.47, b1, 0.47, 0.53, top, 0.53, BLACK); // mount rod
		m.box(0.2, b0, 0.15, 0.8, b1, 0.85, WHITE);                          // body
		m.box(0.55, b0 + 0.05, 0.85, 0.72, b1 - 0.05, 0.9, BLACK);           // lens
		m.box(0.25, b1 - 0.02, 0.2, 0.4, b1, 0.35, DARK_STEEL);              // vents
	}

	/**
	 * Trash can standing on the floor, in {@code body} colour: a rounded body (two crossed boxes), a rim at the top
	 * and the dark opening inside it.
	 */
	static void trashCan(FurnitureModel m, BlockState body) {
		double top = Math.min(0.9, m.h);
		m.box(0.27, 0, 0.22, 0.73, top - 0.06, 0.78, body);
		m.box(0.22, 0, 0.27, 0.78, top - 0.06, 0.73, body);
		m.box(0.24, top - 0.06, 0.19, 0.76, top, 0.81, body);    // rim
		m.box(0.19, top - 0.06, 0.24, 0.81, top, 0.76, body);
		m.box(0.27, top - 0.02, 0.27, 0.73, top + 0.001, 0.73, BLACK); // opening
		m.box(0.25, 0, 0.25, 0.75, 0.03, 0.75, BLACK);           // base
	}

	/** Tower fan standing on the floor: round base, tall body, front grille. */
	static void fan(FurnitureModel m) {
		double h = m.h;
		m.box(0.2, 0, 0.2, 0.8, 0.06, 0.8, BLACK);
		m.box(0.33, 0.06, 0.33, 0.67, h - 0.05, 0.67, BLACK);
		m.box(0.38, 0.15, 0.67, 0.62, h - 0.15, 0.69, DARK_STEEL);
		m.box(0.35, h - 0.05, 0.35, 0.65, h, 0.65, STEEL);
	}

	static void wallClock(FurnitureModel m, double y) {
		double cx = m.w / 2, cy = y + 0.5, r = 0.36;
		for (int row = -4; row <= 4; row++) {
			double dy = row * 0.08, half = Math.sqrt(Math.max(0, 0.4 * 0.4 - dy * dy));
			m.box(cx - half, cy + dy - 0.04, 0, cx + half, cy + dy + 0.04, 0.06, BLACK);
			if (Math.abs(dy) < 0.32) {
				double inner = Math.sqrt(0.32 * 0.32 - dy * dy);
				m.box(cx - inner, cy + dy - 0.035, 0.06, cx + inner, cy + dy + 0.035, 0.08, WHITE);
			}
		}
		for (int i = 0; i < 12; i++) {
			double a = i * Math.PI / 6, px = cx + r * Math.sin(a), py = cy + r * Math.cos(a);
			m.box(px - 0.018, py - 0.018, 0.08, px + 0.018, py + 0.018, 0.095, BLACK);
		}
		m.box(cx - 0.018, cy, 0.09, cx + 0.018, cy + 0.24, 0.11, BLACK);
		m.box(cx, cy - 0.018, 0.11, cx + 0.17, cy + 0.018, 0.13, BLACK);
		m.box(cx - 0.035, cy - 0.035, 0.13, cx + 0.035, cy + 0.035, 0.15, GOLD);
	}

	static void curtain(FurnitureModel m, double y0, double y1, BlockState fabric) {
		m.box(0, y1 - 0.06, 0, m.w, y1, 0.1, IRON);
		for (int i = 0; i < (int) m.w * 4; i++) {
			double a = i * 0.25;
			m.box(a, y0, 0.08 + (i % 2) * 0.025, Math.min(m.w, a + 0.24), y1 - 0.06,
					0.15 + (i % 2) * 0.025, fabric);
		}
	}

	static void bulletinBoard(FurnitureModel m, double y0, double y1) {
		m.box(0, y0, 0, m.w, y1, 0.06, planks("oak"));
		m.box(0.06, y0 + 0.06, 0.06, m.w - 0.06, y1 - 0.06, 0.08, first("brown_terracotta"));
		for (int i = 0; i < Math.max(1, (int) m.w * 2); i++) {
			double x = 0.15 + i * 0.42, top = y1 - 0.18 - (i % 2) * 0.12;
			if (x + 0.27 >= m.w) break;
			m.box(x, top - 0.35, 0.08, x + 0.27, top, 0.09, WHITE);
			m.box(x + 0.12, top - 0.02, 0.09, x + 0.16, top + 0.02, 0.11, RED);
		}
	}

	static void mirror(FurnitureModel m, double y0, double y1) {
		m.box(0, y0, 0, m.w, y1, 0.05, planks("spruce"));
		m.box(0.06, y0 + 0.06, 0.05, m.w - 0.06, y1 - 0.06, 0.07, first("light_blue_stained_glass"));
	}

	static void projectionScreen(FurnitureModel m, double y0, double y1) {
		m.box(0, y1 - 0.12, 0, m.w, y1, 0.14, DARK_STEEL);
		m.box(0.06, y0 + 0.07, 0.08, m.w - 0.06, y1 - 0.12, 0.1, WHITE);
		m.box(0.02, y0, 0.09, m.w - 0.02, y0 + 0.07, 0.15, BLACK);
	}

	static void radiator(FurnitureModel m, double y0, double y1) {
		double top = Math.min(y1, y0 + 0.8);
		m.box(0.04, y0 + 0.12, 0, m.w - 0.04, top, 0.08, DARK_STEEL);
		for (int i = 0; i < (int) m.w * 5; i++) {
			double x = i * 0.2 + 0.04;
			m.box(x, y0 + 0.06, 0.08, Math.min(m.w, x + 0.12), top, 0.18, STEEL);
		}
	}

	static void rangeHood(FurnitureModel m, double y) {
		m.box(0.04, y + 0.12, 0, m.w - 0.04, y + 0.32, 0.55, STEEL);
		m.box(m.w / 2 - 0.2, y + 0.32, 0.05, m.w / 2 + 0.2, y + 0.95, 0.42, STEEL);
		m.box(0.1, y + 0.1, 0.1, m.w - 0.1, y + 0.13, 0.48, BLACK);
	}

	/**
	 * One wall shelf in block row {@code y}: a board on two iron brackets with a row of books (mixed colours and
	 * heights) held by a bookend, and a small plant at the left end of a longer shelf.
	 * Parts start 0.07 out from the wall so a wall map tile behind the shelf (an item frame) shows cleanly.
	 */
	static void wallShelf(FurnitureModel m, double y, BlockState board) {
		double w = m.w, b = y + 0.2;  // board top
		m.box(0, b - 0.07, 0.07, w, b, 0.4, board);
		for (double x : new double[]{0.15, w - 0.2}) {
			m.box(x, b - 0.3, 0.07, x + 0.05, b - 0.07, 0.11, IRON);
			m.box(x, b - 0.11, 0.11, x + 0.05, b - 0.07, 0.34, IRON);
		}
		BlockState[] covers = {RED, first("blue_concrete"), first("green_concrete"), first("brown_concrete"), BLACK,
				first("yellow_concrete"), first("cyan_concrete"), WHITE};
		double x = w >= 2 ? 0.45 : 0.08;
		if (w >= 2) {
			m.box(0.1, b, 0.14, 0.34, b + 0.22, 0.36, first("terracotta"));
			m.box(0.14, b + 0.22, 0.18, 0.3, b + 0.42, 0.32, first("green_concrete"));
		}
		for (int i = 0; x < w * 0.82; i++) {
			int k = (i * 7 + (int) (w * 3)) % 11;          // a fixed, varied pattern of widths and heights
			double t = 0.07 + 0.01 * (k % 4), hgt = 0.38 + 0.03 * k;
			m.box(x, b, 0.1, x + t, b + hgt, 0.36, covers[(i * 3 + k) % covers.length]);
			x += t + 0.005;
		}
		m.box(x + 0.01, b, 0.12, x + 0.04, b + 0.4, 0.34, DARK_STEEL);   // bookend
	}

	static void wallHooks(FurnitureModel m, double y) {
		m.box(0.06, y + 0.55, 0, m.w - 0.06, y + 0.67, 0.09, planks("oak"));
		for (int i = 0; i < (int) m.w * 3; i++) {
			double x = 0.2 + i * 0.3;
			if (x >= m.w) break;
			m.box(x, y + 0.4, 0.09, x + 0.04, y + 0.59, 0.26, IRON);
		}
	}

	static void monitor(FurnitureModel m) {
		m.box(0.22, 1.02, 0.25, 0.78, 1.08, 0.75, BLACK);
		m.box(0.47, 1.08, 0.46, 0.53, 1.25, 0.54, STEEL);
		m.box(0.14, 1.25, 0.42, 0.86, 1.78, 0.48, BLACK);
		m.box(0.18, 1.29, 0.48, 0.82, 1.73, 0.49, first("cyan_concrete"));
	}

	static void tabletop(FurnitureModel m, String type, BlockState color) {
		switch (type) {
			case "laptop" -> {
				m.box(0.15, 1.01, 0.25, 0.85, 1.06, 0.72, DARK_STEEL);
				m.box(0.17, 1.05, 0.25, 0.83, 1.1, 0.68, BLACK);
				m.box(0.16, 1.06, 0.2, 0.84, 1.66, 0.25, BLACK);
				m.glow(0.2, 1.11, 0.25, 0.8, 1.61, 0.26, first("cyan_concrete"));
			}
			case "book_stack" -> {
				for (int i = 0; i < 4; i++) {
					m.box(0.16, 1.01 + i * 0.09, 0.2, 0.8, 1.08 + i * 0.09, 0.72,
							i % 2 == 0 ? color : RED);
					m.box(0.22, 1.02 + i * 0.09, 0.23, 0.76, 1.07 + i * 0.09, 0.7, LINEN);
				}
			}
			case "mug" -> {
				m.box(0.32, 1.01, 0.32, 0.64, 1.34, 0.64, color);
				m.box(0.38, 1.32, 0.38, 0.58, 1.35, 0.58, BLACK);
				m.box(0.64, 1.12, 0.4, 0.75, 1.26, 0.56, color);
			}
			case "small_plant" -> {
				m.box(0.28, 1.01, 0.28, 0.72, 1.35, 0.72, color);
				m.box(0.34, 1.34, 0.34, 0.66, 1.42, 0.66, BLACK);
				m.box(0.37, 1.4, 0.37, 0.63, 1.83, 0.63, first("green_concrete"));
			}
			case "printer" -> {
				m.box(0.09, 1.01, 0.2, 0.91, 1.37, 0.8, STEEL);
				m.box(0.2, 1.38, 0.3, 0.8, 1.48, 0.7, WHITE);
				m.box(0.18, 1.14, 0.8, 0.82, 1.2, 0.84, BLACK);
			}
			case "desk_lamp" -> {
				m.box(0.32, 1.01, 0.32, 0.68, 1.07, 0.68, DARK_STEEL);
				m.box(0.47, 1.06, 0.47, 0.53, 1.5, 0.53, STEEL);
				m.glow(0.27, 1.48, 0.27, 0.73, 1.73, 0.73, LINEN);
			}
		}
	}

	static void ceilingFixture(FurnitureModel m, String type, double bottom, double top) {
		double cx = m.w / 2, cz = m.d / 2;
		switch (type) {
			case "led_panel" -> {
				m.box(0.04, top - 0.12, 0.04, m.w - 0.04, top, m.d - 0.04, STEEL);
				m.glow(0.1, top - 0.13, 0.1, m.w - 0.1, top - 0.11, m.d - 0.1, WHITE);
			}
			case "downlight" -> {
				m.box(cx - 0.26, top - 0.04, cz - 0.26, cx + 0.26, top, cz + 0.26, STEEL);
				m.glow(cx - 0.19, top - 0.045, cz - 0.19, cx + 0.19, top - 0.025, cz + 0.19, WHITE);
			}
			case "smoke_detector" -> {
				m.box(cx - 0.22, top - 0.1, cz - 0.22, cx + 0.22, top, cz + 0.22, WHITE);
				m.box(cx - 0.04, top - 0.105, cz + 0.12, cx + 0.04, top - 0.09, cz + 0.18, RED);
			}
			case "linear_light" -> {
				m.box(cx - 0.025, bottom + 0.13, cz - 0.025, cx + 0.025, top, cz + 0.025, STEEL);
				if (m.w >= m.d) {
					m.box(0.04, bottom, cz - 0.12, m.w - 0.04, bottom + 0.18, cz + 0.12, DARK_STEEL);
					m.glow(0.08, bottom - 0.025, cz - 0.09, m.w - 0.08, bottom, cz + 0.09, WHITE);
				} else {
					m.box(cx - 0.12, bottom, 0.04, cx + 0.12, bottom + 0.18, m.d - 0.04, DARK_STEEL);
					m.glow(cx - 0.09, bottom - 0.025, 0.08, cx + 0.09, bottom, m.d - 0.08, WHITE);
				}
			}
			case "pendant_lamp" -> {
				m.box(cx - 0.025, bottom + 0.35, cz - 0.025, cx + 0.025, top, cz + 0.025, BLACK);
				m.box(cx - 0.32, bottom + 0.08, cz - 0.32, cx + 0.32, bottom + 0.35, cz + 0.32, LINEN);
				m.glow(cx - 0.22, bottom, cz - 0.22, cx + 0.22, bottom + 0.1, cz + 0.22, WHITE);
			}
		}
	}

	static void wallDevice(FurnitureModel m, String type, double y) {
		BlockState housing = switch (type) {
			case "fire_alarm" -> RED;
			case "soap_dispenser", "sanitizer_dispenser" -> WHITE;
			default -> STEEL;
		};
		double z = type.endsWith("dispenser") ? 0.22 : 0.11;
		double width = type.equals("paper_towel_dispenser") ? 0.62 : 0.32;
		m.box(0.5 - width / 2, y + 0.15, 0.01, 0.5 + width / 2, y + 0.72, z, housing);
		switch (type) {
			case "light_switch" -> m.box(0.44, y + 0.35, z, 0.56, y + 0.54, z + 0.03, WHITE);
			case "outlet" -> {
				m.box(0.42, y + 0.3, z, 0.45, y + 0.39, z + 0.02, BLACK);
				m.box(0.55, y + 0.3, z, 0.58, y + 0.39, z + 0.02, BLACK);
			}
			case "thermostat" -> m.glow(0.37, y + 0.36, z, 0.63, y + 0.57, z + 0.02, first("cyan_concrete"));
			case "fire_alarm" -> {
				m.box(0.35, y + 0.43, z, 0.65, y + 0.57, z + 0.06, WHITE);
				m.box(0.39, y + 0.2, z, 0.61, y + 0.39, z + 0.07, DARK_STEEL);
			}
			case "soap_dispenser", "sanitizer_dispenser" -> m.box(0.4, y + 0.18, z, 0.6, y + 0.36, z + 0.07, BLACK);
			case "paper_towel_dispenser" -> m.box(0.35, y + 0.12, z, 0.65, y + 0.22, z + 0.08, WHITE);
		}
	}

	static void studentDesk(FurnitureModel m, BlockState wood) {
		chair(m, wood, DARK_STEEL);
		m.box(0.78, 0.45, 0.15, 0.88, 0.95, 0.22, DARK_STEEL);
		m.box(0.45, 0.84, 0.12, 1.28, 0.94, 0.73, wood);
	}

	static void chairStack(FurnitureModel m, BlockState wood) {
		for (int i = 0; i < 3; i++) {
			double y = i * 0.4;
			m.box(0.13, y + 0.35, 0.13, 0.87, y + 0.44, 0.87, wood);
			m.box(0.13, y + 0.44, 0.13, 0.87, y + 0.95, 0.21, wood);
		}
		legs(m, 0.45, 0.1, DARK_STEEL);
	}

	static void nightstand(FurnitureModel m, BlockState wood) {
		m.box(0.08, 0, 0.07, 0.92, 0.77, 0.92, wood);
		drawers(m, 0.15, 0.85, 0.12, 0.68, 2, planks("spruce"));
		m.box(0, 0.77, 0, 1, 0.86, 1, wood);
	}

	/**
	 * Dresser around barrels (its drawers are real storage): a dark plinth, three drawers per block with pulls, an
	 * overhanging top, and on top a mirror propped at the back and a stack of folded clothes on a wider one.
	 */
	static void dresser(FurnitureModel m, BlockState wood) {
		double w = m.w, d = m.d, h = m.h, t = h + 0.05;
		m.box(0, 0, 0, w, h, d, wood);
		m.box(0, 0, d, w, 0.1, d + 0.03, planks("dark_oak"));
		for (int i = 0; i < (int) w; i++) drawers(m, i + 0.08, i + 0.92, 0.15, h - 0.08, 3, planks("spruce"));
		m.box(-0.03, h - 0.03, -0.01, w + 0.03, t, d + 0.06, wood);
		double mx = w >= 2 ? 0.2 : 0.15, mw = Math.min(0.7, w - 0.3);
		m.box(mx, t, 0.06, mx + mw, t + 0.75, 0.11, planks("spruce"));
		m.box(mx + 0.05, t + 0.05, 0.11, mx + mw - 0.05, t + 0.7, 0.12, first("light_blue_stained_glass"));
		if (w >= 2) {
			BlockState[] cloth = {first("gray_wool"), first("light_blue_wool"), LINEN};
			for (int k = 0; k < cloth.length; k++)
				m.box(w - 0.7, t + k * 0.07, 0.3, w - 0.2, t + (k + 1) * 0.07, 0.75, cloth[k]);
		}
	}

	static void piano(FurnitureModel m) {
		m.box(0, 0, 0, m.w, 1.05, m.d, BLACK);
		m.box(0.05, 0.8, m.d, m.w - 0.05, 0.88, m.d + 0.34, WHITE);
		for (int i = 0; i < (int) (m.w * 8); i++) {
			double x = 0.08 + i * 0.12;
			m.box(x, 0.83, m.d + 0.13, x + 0.04, 0.9, m.d + 0.3, BLACK);
		}
		m.box(0.13, 1.05, 0.08, m.w - 0.13, 1.45, 0.18, BLACK);
	}

	static void shower(FurnitureModel m) {
		m.box(0, 0, 0, m.w, 0.2, m.d, PORCELAIN);
		m.box(0, 0.2, 0, 0.05, m.h, m.d, first("light_blue_stained_glass"));
		m.box(m.w - 0.05, 0.2, 0, m.w, m.h, m.d, first("light_blue_stained_glass"));
		m.box(0, 0.2, m.d - 0.05, m.w, m.h, m.d, first("light_blue_stained_glass"));
		m.box(m.w / 2 - 0.04, m.h - 0.35, 0.06, m.w / 2 + 0.04, m.h - 0.03, 0.12, IRON);
		m.box(m.w / 2 - 0.22, m.h - 0.4, 0.12, m.w / 2 + 0.22, m.h - 0.32, 0.3, IRON);
	}

	static void coatRack(FurnitureModel m) {
		m.box(0.18, 0, 0.18, 0.82, 0.08, 0.82, planks("dark_oak"));
		m.box(0.45, 0.08, 0.45, 0.55, m.h, 0.55, planks("dark_oak"));
		for (double x : new double[]{0.15, 0.58})
			m.box(x, m.h - 0.25, 0.45, x + 0.27, m.h - 0.17, 0.55, IRON);
		for (double z : new double[]{0.15, 0.58})
			m.box(0.45, m.h - 0.4, z, 0.55, m.h - 0.32, z + 0.27, IRON);
	}

	static void ceilingFan(FurnitureModel m, double y, double top) {
		double cx = m.w / 2, cz = m.d / 2;
		m.box(cx - 0.04, y + 0.18, cz - 0.04, cx + 0.04, top, cz + 0.04, DARK_STEEL);
		m.box(cx - 0.18, y, cz - 0.18, cx + 0.18, y + 0.2, cz + 0.18, BLACK);
		m.box(0.04, y + 0.12, cz - 0.12, m.w - 0.04, y + 0.18, cz + 0.12, planks("oak"));
		m.box(cx - 0.12, y + 0.12, 0.04, cx + 0.12, y + 0.18, m.d - 0.04, planks("oak"));
	}

	/** Backpack on the floor: a rounded bag in {@code fabric} with a front pocket, straps and a top handle. */
	static void backpack(FurnitureModel m, BlockState fabric) {
		m.box(0.3, 0, 0.38, 0.7, 0.55, 0.62, fabric);                       // body
		m.box(0.33, 0.55, 0.4, 0.67, 0.6, 0.6, fabric);                     // rounded top
		m.box(0.35, 0.08, 0.62, 0.65, 0.3, 0.68, fabric);                   // front pocket
		m.box(0.47, 0.6, 0.47, 0.53, 0.66, 0.53, BLACK);                    // handle
		m.box(0.34, 0.1, 0.34, 0.38, 0.5, 0.38, BLACK);                     // straps
		m.box(0.62, 0.1, 0.34, 0.66, 0.5, 0.38, BLACK);
	}

	/** Globe: blue sphere (crossed boxes) with green land on a brass axis and base, standing at height {@code y0}. */
	static void globe(FurnitureModel m, double y0) {
		BlockState ocean = first("blue_concrete"), land = first("lime_concrete");
		m.box(0.36, y0, 0.36, 0.64, y0 + 0.04, 0.64, GOLD);                // base
		m.box(0.48, y0 + 0.04, 0.48, 0.52, y0 + 0.18, 0.52, GOLD);          // stem
		double c = y0 + 0.42;
		m.box(0.3, c - 0.18, 0.34, 0.7, c + 0.18, 0.66, ocean);
		m.box(0.34, c - 0.22, 0.3, 0.66, c + 0.22, 0.7, ocean);
		m.box(0.34, c - 0.05, 0.69, 0.5, c + 0.14, 0.71, land);             // continents
		m.box(0.52, c - 0.16, 0.69, 0.62, c - 0.02, 0.71, land);
		m.box(0.29, c + 0.02, 0.42, 0.31, c + 0.16, 0.58, land);
		m.box(0.25, c - 0.02, 0.49, 0.29, c + 0.02, 0.51, GOLD);            // meridian ends
		m.box(0.71, c - 0.02, 0.49, 0.75, c + 0.02, 0.51, GOLD);
	}

	/** A bank of lockers in {@code paint}: per block column a tall door with vents, a handle and a number plate. */
	static void lockers(FurnitureModel m, BlockState paint) {
		double w = m.w, d = m.d, h = m.h;
		m.box(0, 0, 0, w, h, d, paint);
		for (int c = 0; c < (int) w; c++) {
			m.box(c + 0.05, 0.08, d, c + 0.95, h - 0.06, d + 0.02, paint);  // door
			for (int v = 0; v < 3; v++) {
				double vy = h - 0.25 - v * 0.08;
				m.box(c + 0.3, vy, d + 0.02, c + 0.7, vy + 0.03, d + 0.03, DARK_STEEL);  // vents
			}
			m.box(c + 0.78, h / 2 - 0.1, d + 0.02, c + 0.84, h / 2 + 0.1, d + 0.05, STEEL);  // handle
			m.box(c + 0.4, h - 0.5, d + 0.02, c + 0.6, h - 0.42, d + 0.03, STEEL);          // number plate
		}
		m.box(0, 0, -0.01, w, 0.08, d + 0.03, BLACK);                       // kick plate
	}

	/** Drinking fountain on the wall in row y0: steel basin with a spout and push button, a bottle filler above. */
	static void drinkingFountain(FurnitureModel m, double y0) {
		m.box(0.2, y0 + 0.5, 0, 0.8, y0 + 0.82, 0.45, STEEL);               // basin
		m.box(0.25, y0 + 0.78, 0.05, 0.75, y0 + 0.83, 0.4, DARK_STEEL);     // bowl
		m.box(0.46, y0 + 0.83, 0.25, 0.54, y0 + 0.88, 0.3, STEEL);          // spout
		m.box(0.62, y0 + 0.6, 0.45, 0.72, y0 + 0.7, 0.47, BLACK);           // button
		m.box(0.25, y0 + 1.0, 0, 0.75, y0 + 1.45, 0.18, STEEL);             // bottle filler
		m.box(0.32, y0 + 1.05, 0.18, 0.68, y0 + 1.15, 0.2, BLACK);          // its recess
	}

	/** Umbrella stand: a dark cylinder (crossed boxes) with two umbrella handles sticking out. */
	static void umbrellaStand(FurnitureModel m) {
		m.box(0.32, 0, 0.28, 0.68, 0.6, 0.72, BLACK);
		m.box(0.28, 0, 0.32, 0.72, 0.6, 0.68, BLACK);
		m.box(0.4, 0.6, 0.42, 0.44, 0.95, 0.46, first("red_concrete"));     // umbrella shafts
		m.box(0.55, 0.6, 0.52, 0.59, 0.9, 0.56, first("blue_concrete"));
		m.box(0.4, 0.95, 0.42, 0.5, 0.98, 0.46, BLACK);                     // handles
		m.box(0.55, 0.9, 0.52, 0.65, 0.93, 0.56, BLACK);
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

package com.dubhacks.everything_cubed.build;

import com.dubhacks.everything_cubed.EverythingCubed;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Display;
import net.minecraft.world.entity.Interaction;
import net.minecraft.world.entity.decoration.ItemFrame;
import net.minecraft.world.entity.decoration.painting.Painting;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.LightBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BedPart;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.level.block.state.properties.DoorHingeSide;
import net.minecraft.world.level.block.state.properties.DoubleBlockHalf;
import net.minecraft.world.level.block.state.properties.Half;
import net.minecraft.world.level.block.state.properties.Property;
import net.minecraft.world.level.block.state.properties.SlabType;
import net.minecraft.world.level.saveddata.maps.MapId;
import net.minecraft.world.level.saveddata.maps.MapItemSavedData;
import net.minecraft.world.phys.Vec3;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Places a plan produced by the analyzer (see analyzer/roomplan.py for the coordinate conventions).
 * Plan x/z map directly onto world x/z relative to {@code origin}, which is the first interior block
 * of the floor layer. Everything usable is a real, vanilla block (beds, barrels, furnaces, ...) or an
 * interaction entity (seats, lamps); most furniture also gets a detailed display model, see {@link FurnitureModel}.
 */
public class HouseBuilder {
	private static final String[] PLANTS = {"potted_fern", "potted_azalea_bush", "potted_oak_sapling",
			"potted_red_tulip", "potted_cactus", "potted_flowering_azalea_bush"};

	private final ServerLevel level;
	private final BlockPos origin;
	private final JsonObject plan;
	private final int W, D, H;
	private final UndoLog undo;
	/** Every block we set, so a final pass can recompute connections (panes, fences, stairs, doors). */
	private final Map<BlockPos, BlockState> placed = new LinkedHashMap<>();
	/** Interior cells already used by furniture or kept clear for a doorway. */
	private final Set<Long> occupied = new HashSet<>();
	private final String wallMain, wood;
	private final boolean ridgeAlongX;
	private int plantIndex;

	public HouseBuilder(ServerLevel level, BlockPos origin, JsonObject plan) {
		this.level = level;
		this.origin = origin;
		this.plan = plan;
		JsonObject size = plan.getAsJsonObject("size");
		this.W = size.get("x").getAsInt();
		this.D = size.get("z").getAsInt();
		this.H = size.get("y").getAsInt();
		this.undo = new UndoLog(level);
		this.wallMain = plan.get("wall_main").getAsString();
		JsonObject roof = plan.getAsJsonObject("roof");
		this.wood = roof.get("wood").getAsString();
		this.ridgeAlongX = "x".equals(roof.get("axis").getAsString());
	}

	public UndoLog build() {
		clearSite();
		buildShell();
		buildRoof();
		buildOpenings();
		buildObjects(); // lights are only the ones detected in the photos: a dark room stays dark
		finishShapes();
		hangWallTextures();
		return undo;
	}

	// ------------------------------------------------------------------------------------- blocks

	private static BlockState block(String id) {
		Identifier key = Identifier.tryParse(id.contains(":") ? id : "minecraft:" + id);
		if (key == null) return Blocks.STONE.defaultBlockState();
		return BuiltInRegistries.BLOCK.getOptional(key).map(Block::defaultBlockState).orElse(null);
	}

	/** First block id that exists in this game version. */
	static BlockState first(String... ids) {
		for (String id : ids) {
			BlockState s = block(id);
			if (s != null) return s;
		}
		return Blocks.STONE.defaultBlockState();
	}

	private static <T extends Comparable<T>> BlockState with(BlockState s, Property<T> p, T v) {
		return s.hasProperty(p) ? s.setValue(p, v) : s;
	}

	private static Direction dir(String name) {
		return switch (name == null ? "" : name) {
			case "north" -> Direction.NORTH;
			case "south" -> Direction.SOUTH;
			case "west" -> Direction.WEST;
			default -> Direction.EAST;
		};
	}

	private void set(int x, int y, int z, BlockState s) {
		BlockPos p = origin.offset(x, y, z);
		undo.remember(p);
		level.setBlock(p, s, Block.UPDATE_CLIENTS);
		placed.put(p, s);
	}

	private boolean isAir(int x, int y, int z) {
		return level.getBlockState(origin.offset(x, y, z)).isAir();
	}

	/** Marks a cell as used; false if something already claimed it. */
	private boolean claim(int x, int y, int z) {
		return occupied.add(BlockPos.asLong(x, y, z));
	}

	// ------------------------------------------------------------------------------------- shell

	private int roofHeight() {
		int across = ridgeAlongX ? D : W;
		return (across + 4) / 2 + 1;
	}

	private void clearSite() {
		BlockState air = Blocks.AIR.defaultBlockState();
		for (int x = -3; x <= W + 2; x++)
			for (int z = -3; z <= D + 2; z++)
				for (int y = 1; y <= H + 2 + roofHeight(); y++)
					if (!isAir(x, y, z)) set(x, y, z, air);
	}

	private void buildShell() {
		BlockState main = first(wallMain);
		BlockState pillar = with(first("stripped_" + wood + "_log", "stripped_oak_log"), BlockStateProperties.AXIS, Direction.Axis.Y);

		// Floor (y=0), matched cell by cell; the ring under the walls is part of the wall.
		JsonObject floor = plan.getAsJsonObject("floor");
		List<BlockState> fPal = palette(floor.getAsJsonArray("palette"));
		JsonArray fGrid = floor.getAsJsonArray("grid");
		for (int z = 0; z < D; z++) {
			JsonArray row = fGrid.get(z).getAsJsonArray();
			for (int x = 0; x < W; x++) set(x, 0, z, fPal.get(row.get(x).getAsInt()));
		}
		for (int x = -1; x <= W; x++) { set(x, 0, -1, main); set(x, 0, D, main); }
		for (int z = 0; z < D; z++) { set(-1, 0, z, main); set(W, 0, z, main); }

		// Walls (rows 0..H-1 -> y 1..H), each matched to how it looked in the photos.
		JsonObject walls = plan.getAsJsonObject("walls");
		for (String side : new String[]{"north", "south", "west", "east"}) {
			JsonObject w = walls.getAsJsonObject(side);
			List<BlockState> pal = palette(w.getAsJsonArray("palette"));
			JsonArray grid = w.getAsJsonArray("grid");
			for (int row = 0; row < H; row++) {
				JsonArray r = grid.get(row).getAsJsonArray();
				for (int u = 0; u < r.size(); u++) {
					int[] p = wallCell(side, u);
					set(p[0], row + 1, p[1], pal.get(r.get(u).getAsInt()));
				}
			}
		}
		// Corner pillars (only visible from outside).
		for (int y = 0; y <= H + 1; y++) {
			set(-1, y, -1, pillar); set(W, y, -1, pillar); set(-1, y, D, pillar); set(W, y, D, pillar);
		}
		// Ceiling layer.
		BlockState ceiling = first(plan.get("ceiling").getAsString());
		for (int x = -1; x <= W; x++)
			for (int z = -1; z <= D; z++)
				set(x, H + 1, z, (x < 0 || z < 0 || x >= W || z >= D) ? main : ceiling);
	}

	private List<BlockState> palette(JsonArray ids) {
		List<BlockState> out = new ArrayList<>();
		for (JsonElement e : ids) out.add(first(e.getAsString(), wallMain));
		return out;
	}

	/** World x/z of the wall block at position u along a side. */
	private int[] wallCell(String side, int u) {
		return switch (side) {
			case "north" -> new int[]{u, -1};
			case "south" -> new int[]{u, D};
			case "west" -> new int[]{-1, u};
			default -> new int[]{W, u};
		};
	}

	/** World x/z of the interior block directly in front of wall position u. */
	private int[] insideCell(String side, int u) {
		return switch (side) {
			case "north" -> new int[]{u, 0};
			case "south" -> new int[]{u, D - 1};
			case "west" -> new int[]{0, u};
			default -> new int[]{W - 1, u};
		};
	}

	// ------------------------------------------------------------------------------------- roof

	/** Gable roof of stairs over the long axis with a one block overhang; gable ends use the wall block. */
	private void buildRoof() {
		int along = ridgeAlongX ? W : D, across = ridgeAlongX ? D : W;
		BlockState stairs = first(wood + "_stairs", "dark_oak_stairs");
		BlockState slab = first(wood + "_slab", "dark_oak_slab");
		BlockState gable = first(wallMain);
		Direction up = ridgeAlongX ? Direction.SOUTH : Direction.EAST; // direction of increasing "across"
		int baseY = H + 1;
		for (int k = 0; ; k++) {
			int lo = -2 + k, hi = across + 1 - k, y = baseY + k;
			if (lo > hi) break;
			for (int a = -2; a <= along + 1; a++) {
				if (lo == hi) {
					put(a, lo, y, with(slab, BlockStateProperties.SLAB_TYPE, SlabType.BOTTOM));
					continue;
				}
				put(a, lo, y, with(stairs, BlockStateProperties.HORIZONTAL_FACING, up));
				put(a, hi, y, with(stairs, BlockStateProperties.HORIZONTAL_FACING, up.getOpposite()));
			}
			if (k >= 1)
				for (int b = lo + 1; b < hi; b++) {
					put(-1, b, y, gable);
					put(along, b, y, gable);
				}
		}
	}

	/** Sets a roof block by (along the ridge, across the ridge) coordinates. */
	private void put(int a, int b, int y, BlockState s) {
		if (ridgeAlongX) set(a, y, b, s);
		else set(b, y, a, s);
	}

	// ------------------------------------------------------------------------------------- openings

	private void buildOpenings() {
		for (JsonElement e : plan.getAsJsonArray("openings")) {
			JsonObject o = e.getAsJsonObject();
			String type = o.get("type").getAsString(), side = o.get("side").getAsString();
			int u0 = o.get("u").getAsInt(), w = o.get("w").getAsInt();
			int y0 = o.get("y").getAsInt(), h = o.get("h").getAsInt();
			Direction out = dir(side);
			switch (type) {
				case "window" -> {
					BlockState pane = Blocks.GLASS_PANE.defaultBlockState();
					for (int u = u0; u < u0 + w; u++)
						for (int r = y0; r < Math.min(H, y0 + h); r++) {
							int[] c = wallCell(side, u);
							set(c[0], r + 1, c[1], pane);
						}
				}
				case "door" -> {
					String mat = o.has("material") ? o.get("material").getAsString() : "oak";
					BlockState door = with(first(mat + "_door", "oak_door"), BlockStateProperties.HORIZONTAL_FACING, out.getOpposite());
					for (int i = 0; i < w; i++) {
						int[] c = wallCell(side, u0 + i);
						DoorHingeSide hinge = (w == 2 && i == 1) ? DoorHingeSide.RIGHT : DoorHingeSide.LEFT;
						BlockState d = with(door, BlockStateProperties.DOOR_HINGE, hinge);
						set(c[0], 1, c[1], with(d, BlockStateProperties.DOUBLE_BLOCK_HALF, DoubleBlockHalf.LOWER));
						set(c[0], 2, c[1], with(d, BlockStateProperties.DOUBLE_BLOCK_HALF, DoubleBlockHalf.UPPER));
						// Keep the way in clear and add a step outside.
						int[] in = insideCell(side, u0 + i);
						claim(in[0], 1, in[1]);
						claim(in[0], 2, in[1]);
						int ox = c[0] + out.getStepX(), oz = c[1] + out.getStepZ();
						set(ox, 0, oz, with(first(wood + "_slab"), BlockStateProperties.SLAB_TYPE, SlabType.TOP));
					}
					// A torch above the front door.
					int[] c = wallCell(side, u0);
					set(c[0] + out.getStepX(), 3, c[1] + out.getStepZ(),
							with(Blocks.WALL_TORCH.defaultBlockState(), BlockStateProperties.HORIZONTAL_FACING, out));
				}
				case "painting" -> {
					int[] in = insideCell(side, u0 + w / 2);
					BlockPos pos = origin.offset(in[0], 1 + Math.min(H - 1, y0 + h / 2), in[1]);
					Painting.create(level, pos, out.getOpposite()).ifPresent(p -> {
						level.addFreshEntity(p);
						undo.rememberEntity(p);
					});
				}
				case "tv" -> {
					for (int u = u0; u < u0 + w; u++)
						for (int r = y0; r < Math.min(H - 1, y0 + Math.max(1, h)); r++) {
							int[] in = insideCell(side, u);
							if (claim(in[0], r + 1, in[1])) set(in[0], r + 1, in[1], first("black_concrete"));
						}
				}
				case "whiteboard", "exit_sign", "fire_extinguisher", "clock", "curtain", "mirror",
						"bulletin_board", "projection_screen", "radiator", "range_hood", "wall_hooks" -> {
					// Flat things on the wall: a model in the interior cells along it, its back against the wall.
					int[] a = insideCell(side, u0), b = insideCell(side, u0 + w - 1);
					int x0 = Math.min(a[0], b[0]), z0 = Math.min(a[1], b[1]);
					int top = Math.min(H - 1, y0 + Math.max(1, h));
					FurnitureModel m = new FurnitureModel(x0, z0, Math.abs(b[0] - a[0]) + 1, Math.abs(b[1] - a[1]) + 1, top, out.getOpposite());
					switch (type) {
						case "whiteboard" -> {
							String col = o.has("color") ? o.get("color").getAsString() : "white";
							FurnitureDesigns.whiteboard(m, y0, y0 + Math.max(1, h), first(col + "_concrete", "white_concrete"));
						}
						case "exit_sign" -> {
							FurnitureDesigns.exitSign(m, y0);
							// An exit sign is a light: a dim one, like the real thing.
							if (claim(a[0], y0 + 1, a[1]))
								set(a[0], y0 + 1, a[1], Blocks.LIGHT.defaultBlockState().setValue(LightBlock.LEVEL, 6));
						}
						case "clock" -> FurnitureDesigns.wallClock(m, y0);
						case "curtain" -> FurnitureDesigns.curtain(m, y0, y0 + Math.max(1, h),
								first((o.has("color") ? o.get("color").getAsString() : "white") + "_wool", "white_wool"));
						case "mirror" -> FurnitureDesigns.mirror(m, y0, y0 + Math.max(1, h));
						case "bulletin_board" -> FurnitureDesigns.bulletinBoard(m, y0, y0 + Math.max(1, h));
						case "projection_screen" -> FurnitureDesigns.projectionScreen(m, y0, y0 + Math.max(1, h));
						case "radiator" -> FurnitureDesigns.radiator(m, y0, y0 + Math.max(1, h));
						case "range_hood" -> FurnitureDesigns.rangeHood(m, y0);
						case "wall_hooks" -> FurnitureDesigns.wallHooks(m, y0);
						default -> FurnitureDesigns.fireExtinguisher(m, y0);
					}
					furnish(m, List.of());
				}
				case "sconce" -> {
					int[] in = insideCell(side, u0);
					if (claim(in[0], y0 + 1, in[1]))
						set(in[0], y0 + 1, in[1], with(Blocks.WALL_TORCH.defaultBlockState(), BlockStateProperties.HORIZONTAL_FACING, out.getOpposite()));
				}
				default -> { }
			}
		}
	}

	// ------------------------------------------------------------------------------------- objects

	private void buildObjects() {
		List<JsonObject> rugs = new ArrayList<>();
		for (JsonElement e : plan.getAsJsonArray("objects")) {
			JsonObject o = e.getAsJsonObject();
			if ("rug".equals(o.get("type").getAsString())) rugs.add(o);
			else buildObject(o);
		}
		for (JsonObject o : rugs) buildObject(o); // carpets only go where nothing else stands
	}

	private void buildObject(JsonObject o) {
		String type = o.get("type").getAsString();
		int x0 = o.get("x").getAsInt(), z0 = o.get("z").getAsInt();
		int w = o.get("w").getAsInt(), d = o.get("d").getAsInt(), h = Math.max(1, Math.min(H - 1, o.get("h").getAsInt()));
		Direction facing = dir(o.has("facing") ? o.get("facing").getAsString() : "south");
		String mat = o.has("material") ? o.get("material").getAsString() : "oak";
		String color = o.has("color") ? o.get("color").getAsString() : "white";

		BlockState planks = FurnitureDesigns.planks(mat), log = FurnitureDesigns.strippedLog(mat);
		switch (type) {
			case "bed" -> {
				BlockState bed = with(first(color + "_bed", "red_bed"), BlockStateProperties.HORIZONTAL_FACING, facing);
				boolean alongX = facing.getAxis() == Direction.Axis.X;
				int count = alongX ? d : w;
				List<BlockPos> core = new ArrayList<>();
				for (int i = 0; i < count; i++) {
					int fx, fz;
					if (alongX) { fx = x0 + (facing == Direction.EAST ? 0 : 1); fz = z0 + i; }
					else { fx = x0 + i; fz = z0 + (facing == Direction.SOUTH ? 0 : 1); }
					int hx = fx + facing.getStepX(), hz = fz + facing.getStepZ();
					if (!inside(fx, fz) || !inside(hx, hz) || !claim(fx, 1, fz) || !claim(hx, 1, hz)) continue;
					set(fx, 1, fz, with(bed, BlockStateProperties.BED_PART, BedPart.FOOT));
					set(hx, 1, hz, with(bed, BlockStateProperties.BED_PART, BedPart.HEAD));
					core.add(origin.offset(fx, 1, fz));
					core.add(origin.offset(hx, 1, hz));
				}
				if (core.size() == 2 * count) {
					// The head is at the model's back, so its front faces the foot end.
					FurnitureModel m = alongX ? new FurnitureModel(x0, z0, 2, count, 1, facing.getOpposite())
							: new FurnitureModel(x0, z0, count, 2, 1, facing.getOpposite());
					FurnitureDesigns.bed(m, FurnitureDesigns.planks(wood), first(color + "_wool", "white_wool"));
					furnish(m, core);
				}
			}
			case "chair" -> {
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					// No block: a chair's legs leave room to walk through it, like a real one.
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 1, facing.getOpposite());
					FurnitureDesigns.chair(m, planks, log);
					seat(x0, z0, 0.8f, 0.5f, furnish(m, List.of()));
				}
			}
			case "student_desk" -> {
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 1, facing.getOpposite());
					FurnitureDesigns.studentDesk(m, planks);
					seat(x0, z0, 0.8f, 0.5f, furnish(m, List.of()));
				}
			}
			case "chair_stack" -> {
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 2, facing);
					FurnitureDesigns.chairStack(m, planks);
					furnish(m, List.of());
				}
			}
			case "sofa" -> {
				// One row of stairs along the back of the footprint, each one a seat.
				boolean rowAlongX = facing.getAxis() == Direction.Axis.Z;
				int len = rowAlongX ? w : d;
				int bx = facing == Direction.EAST ? x0 + w - 1 : x0, bz = facing == Direction.SOUTH ? z0 + d - 1 : z0;
				List<BlockPos> core = new ArrayList<>();
				for (int i = 0; i < len; i++) {
					int x = rowAlongX ? x0 + i : bx, z = rowAlongX ? bz : z0 + i;
					core.addAll(fill(x, z, 1, 1, 1, stairs(mat, facing)));
				}
				if (core.size() == len) {
					FurnitureModel m = rowAlongX ? new FurnitureModel(x0, bz, len, 1, 1, facing.getOpposite())
							: new FurnitureModel(bx, z0, 1, len, 1, facing.getOpposite());
					FurnitureDesigns.sofa(m, first(color + "_wool", "white_wool"));
					String group = furnish(m, core);
					for (BlockPos p : core) seat(p.getX() - origin.getX(), p.getZ() - origin.getZ(), 1.02f, 0.6f, group);
				}
			}
			case "table", "desk" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1, with(first(mat + "_slab"), BlockStateProperties.SLAB_TYPE, SlabType.TOP));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, 1, facing);
					boolean pedestal = type.equals("desk") && m.w >= 2;
					if (pedestal) { // the desk's drawers are a barrel at its right end
						BlockState barrel = with(Blocks.BARREL.defaultBlockState(), BlockStateProperties.FACING, facing);
						for (int lz = 0; lz < (int) m.d; lz++) {
							int[] c = m.cell((int) m.w - 1, lz);
							set(c[0], 1, c[1], barrel);
						}
					}
					FurnitureDesigns.table(m, planks, log, pedestal);
					furnish(m, core);
				}
			}
			case "coffee_table" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1, with(first(mat + "_slab"), BlockStateProperties.SLAB_TYPE, SlabType.BOTTOM));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, 1, facing);
					FurnitureDesigns.coffeeTable(m, planks, log);
					furnish(m, core);
				}
			}
			case "counter" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1, with(Blocks.BARREL.defaultBlockState(), BlockStateProperties.FACING, Direction.UP));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, 1, facing);
					FurnitureDesigns.counter(m, planks, log);
					furnish(m, core);
				}
			}
			case "cabinet", "dresser", "nightstand", "wardrobe", "washer" -> {
				int height = Math.min(H, type.equals("wardrobe") ? Math.max(2, h) : h);
				List<BlockPos> core = fill(x0, z0, w, d, height, with(Blocks.BARREL.defaultBlockState(), BlockStateProperties.FACING, facing));
				if (core.size() == w * d * height) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, height, facing);
					switch (type) {
						case "cabinet" -> FurnitureDesigns.cabinet(m, planks, log);
						case "dresser" -> FurnitureDesigns.dresser(m, planks);
						case "nightstand" -> FurnitureDesigns.nightstand(m, planks);
						case "wardrobe" -> FurnitureDesigns.wardrobe(m, planks, log);
						default -> FurnitureDesigns.washer(m);
					}
					furnish(m, core);
				}
			}
			case "podium" -> {
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					BlockState lectern = with(first("lectern"), BlockStateProperties.HORIZONTAL_FACING, facing);
					set(x0, 1, z0, lectern);
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 1.2, facing);
					FurnitureDesigns.podium(m, planks);
					furnish(m, List.of(origin.offset(x0, 1, z0)));
				}
			}
			case "piano" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1, first("black_concrete"));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, 1.5, facing);
					FurnitureDesigns.piano(m);
					furnish(m, core);
				}
			}
			case "storage" -> fill(x0, z0, w, d, 1, with(Blocks.BARREL.defaultBlockState(), BlockStateProperties.FACING, Direction.UP));
			case "bookshelf" -> {
				List<BlockPos> core = fill(x0, z0, w, d, h, with(Blocks.CHISELED_BOOKSHELF.defaultBlockState(), BlockStateProperties.HORIZONTAL_FACING, facing));
				if (core.size() == w * d * h) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, h, facing);
					FurnitureDesigns.bookshelf(m, planks);
					furnish(m, core);
				}
			}
			case "shelf" -> {
				BlockState shelf = with(first(mat + "_shelf", "oak_shelf", "chiseled_bookshelf"), BlockStateProperties.HORIZONTAL_FACING, facing);
				int y = Math.min(H, Math.max(1, h));
				for (int x = x0; x < x0 + w; x++)
					for (int z = z0; z < z0 + d; z++)
						if (claim(x, y, z)) set(x, y, z, shelf);
			}
			case "fridge" -> {
				int height = Math.min(H, Math.max(2, h));
				List<BlockPos> core = fill(x0, z0, w, d, height, with(first("white_shulker_box"), BlockStateProperties.FACING, facing));
				if (core.size() == w * d * height) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, height, facing);
					FurnitureDesigns.fridge(m);
					furnish(m, core);
				}
			}
			case "stove" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1, with(Blocks.FURNACE.defaultBlockState(), BlockStateProperties.HORIZONTAL_FACING, facing));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, 1, facing);
					FurnitureDesigns.stove(m);
					furnish(m, core);
				}
			}
			case "microwave" -> {
				List<BlockPos> core = fill(x0, z0, 1, 1, 1, with(Blocks.SMOKER.defaultBlockState(), BlockStateProperties.HORIZONTAL_FACING, facing));
				if (core.size() == 1) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 1, facing);
					FurnitureDesigns.microwave(m);
					furnish(m, core);
				}
			}
			case "sink", "bathtub" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1, with(Blocks.WATER_CAULDRON.defaultBlockState(), BlockStateProperties.LEVEL_CAULDRON, 3));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, 1, facing);
					if (type.equals("sink")) FurnitureDesigns.sink(m, planks, log);
					else FurnitureDesigns.bathtub(m);
					furnish(m, core);
				}
			}
			case "toilet" -> {
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 1, facing);
					FurnitureDesigns.toilet(m);
					seat(x0, z0, 0.7f, 0.44f, furnish(m, List.of()));
				}
			}
			case "trash" -> {
				// Vanilla has no trash can (this was a composter): a small bin model in the colour seen in the photo.
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 0.9, facing);
					FurnitureDesigns.trashCan(m, first(color + "_concrete", "gray_concrete"));
					String group = furnish(m, List.of());
					// Right-click it holding something to throw that away; punch it to remove the bin.
					undo.rememberEntity(FurnitureInteractions.spawnHitbox(level, Vec3.atBottomCenterOf(origin.offset(x0, 1, z0)),
							0.62f, 0.9f, FurnitureInteractions.TRASH, group));
				}
			}
			case "fireplace" -> {
				fill(x0, z0, w, d, 1, Blocks.BRICKS.defaultBlockState());
				int cx = x0 + w / 2, cz = z0 + d / 2;
				set(cx, 1, cz, Blocks.CAMPFIRE.defaultBlockState());
				for (int y = 2; y <= Math.min(H, 1 + Math.max(2, h)); y++)
					for (int x = x0; x < x0 + w; x++)
						for (int z = z0; z < z0 + d; z++)
							if (claim(x, y, z)) set(x, y, z, Blocks.BRICKS.defaultBlockState());
			}
			case "tv" -> fill(x0, z0, w, d, Math.min(2, h), first("black_concrete"));
			case "monitor" -> {
				int y = Math.max(2, Math.min(H, o.has("y") ? o.get("y").getAsInt() : 2));
				if (inside(x0, z0) && claim(x0, y, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 1.8, facing);
					FurnitureDesigns.monitor(m);
					furnish(m, List.of());
				}
			}
			case "shower" -> {
				List<BlockPos> core = fill(x0, z0, w, d, 1,
						with(Blocks.WATER_CAULDRON.defaultBlockState(), BlockStateProperties.LEVEL_CAULDRON, 3));
				if (core.size() == w * d) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, Math.max(2, h), facing);
					FurnitureDesigns.shower(m);
					furnish(m, core);
				}
			}
			case "coat_rack" -> {
				if (inside(x0, z0) && claim(x0, 1, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, Math.max(1.5, h), facing);
					FurnitureDesigns.coatRack(m);
					furnish(m, List.of());
				}
			}
			case "lamp" -> {
				if (inside(x0, z0) && claim(x0, 1, z0) && claim(x0, 2, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, 2, Direction.SOUTH);
					FurnitureDesigns.lamp(m);
					String group = furnish(m, List.of());
					Interaction lamp = FurnitureInteractions.spawnHitbox(level, Vec3.atBottomCenterOf(origin.offset(x0, 1, z0)),
							0.6f, 2f, FurnitureInteractions.LAMP, group);
					undo.rememberEntity(lamp);
					BlockPos light = FurnitureInteractions.lightPos(lamp);
					set(light.getX() - origin.getX(), light.getY() - origin.getY(), light.getZ() - origin.getZ(),
							Blocks.LIGHT.defaultBlockState().setValue(LightBlock.LEVEL, LightBlock.MAX_LEVEL));
				}
			}
			case "plant" -> fill(x0, z0, 1, 1, 1, first(PLANTS[plantIndex++ % PLANTS.length], "flower_pot"));
			case "projector" -> {
				int y = Math.max(1, Math.min(H, o.has("y") ? o.get("y").getAsInt() : H));
				if (inside(x0, z0) && claim(x0, y, z0)) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, H - 1, facing);
					FurnitureDesigns.projector(m, y - 1, H);
					furnish(m, List.of());
				}
			}
			case "fan" -> {
				int fh = Math.max(1, Math.min(2, h));
				boolean free = inside(x0, z0);
				for (int y = 1; free && y <= fh; y++) free = claim(x0, y, z0);
				if (free) {
					FurnitureModel m = new FurnitureModel(x0, z0, 1, 1, fh * 0.85, facing);
					FurnitureDesigns.fan(m);
					furnish(m, List.of());
				}
			}
			case "chandelier" -> {
				// Hangs from the ceiling: body rows y..y+h-1, rod up to the ceiling. The model's own height is H - 1
				// so its light probe sits in the top air layer, not inside the ceiling.
				int y = Math.max(1, Math.min(H, o.has("y") ? o.get("y").getAsInt() : H));
				int body = Math.max(1, Math.min(H - y + 1, h));
				int cx = x0 + (w - 1) / 2, cz = z0 + (d - 1) / 2;
				if (inside(cx, cz) && claim(cx, y, cz)) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, H - 1, Direction.SOUTH);
					FurnitureDesigns.chandelier(m, y - 1, y - 1 + body, H);
					furnish(m, List.of());
					set(cx, y, cz, Blocks.LIGHT.defaultBlockState().setValue(LightBlock.LEVEL, LightBlock.MAX_LEVEL));
				}
			}
			case "ceiling_fan" -> {
				int y = Math.max(1, Math.min(H, o.has("y") ? o.get("y").getAsInt() : H));
				int cx = x0 + w / 2, cz = z0 + d / 2;
				if (inside(cx, cz) && claim(cx, y, cz)) {
					FurnitureModel m = new FurnitureModel(x0, z0, w, d, H - 1, Direction.SOUTH);
					FurnitureDesigns.ceilingFan(m, y - 1, H);
					furnish(m, List.of());
				}
			}
			case "ceiling_light" -> {
				if (inside(x0, z0) && claim(x0, H, z0))
					set(x0, H, z0, with(Blocks.LANTERN.defaultBlockState(), BlockStateProperties.HANGING, true));
			}
			case "rug" -> {
				BlockState carpet = first(color + "_carpet", "white_carpet");
				for (int x = x0; x < x0 + w; x++)
					for (int z = z0; z < z0 + d; z++)
						if (inside(x, z) && claim(x, 1, z)) set(x, 1, z, carpet);
			}
			case "vanilla" -> {
				// A detected thing that is a vanilla block: a full cube fills the footprint (a bookshelf), anything
				// else is one block at its place, facing into the room, hanging if it hangs (a lantern).
				BlockState b = first(o.get("block").getAsString(), "stone");
				boolean cube = o.has("cube") && o.get("cube").getAsBoolean();
				int y = Math.max(1, Math.min(H, o.has("y") ? o.get("y").getAsInt() : 1));
				if (cube) fill(x0, z0, w, d, h, b);
				else if (inside(x0, z0) && claim(x0, y, z0)) {
					b = with(b, BlockStateProperties.HORIZONTAL_FACING, facing);
					if (o.has("hanging") && o.get("hanging").getAsBoolean()) b = with(b, BlockStateProperties.HANGING, true);
					set(x0, y, z0, b);
				}
			}
			// No template yet ("generic" from the detector, or a type this version does not know): a plain box
			// over the object's footprint, so it is still there and shows which templates are missing.
			default -> {
				String name = o.has("name") ? o.get("name").getAsString() : type;
				EverythingCubed.LOGGER.info("No template for '{}', building a plain {}x{}x{} box", name, w, d, h);
				fill(x0, z0, w, d, h, first(color + "_concrete", "white_concrete"));
			}
		}
	}

	/**
	 * Spawns a furniture model and returns its group tag. {@code core} are the real blocks inside it:
	 * breaking one of them removes the model.
	 */
	private String furnish(FurnitureModel m, List<BlockPos> core) {
		String group = FurnitureInteractions.newGroup();
		List<Display.BlockDisplay> parts = m.spawn(level, origin, group);
		parts.forEach(undo::rememberEntity);
		if (!parts.isEmpty()) for (BlockPos p : core) parts.getFirst().addTag(FurnitureInteractions.coreTag(p));
		return group;
	}

	/** A seat hitbox on interior cell (x, z); players sit at {@code height} above the floor. */
	private void seat(int x, int z, float width, float height, String group) {
		undo.rememberEntity(FurnitureInteractions.spawnHitbox(level, Vec3.atBottomCenterOf(origin.offset(x, 1, z)),
				width, height, FurnitureInteractions.SEAT, group));
	}

	private BlockState stairs(String mat, Direction back) {
		BlockState s = with(first(mat + "_stairs", "oak_stairs"), BlockStateProperties.HORIZONTAL_FACING, back);
		return with(s, BlockStateProperties.HALF, Half.BOTTOM);
	}

	private boolean inside(int x, int z) {
		return x >= 0 && z >= 0 && x < W && z < D;
	}

	/**
	 * Fill a footprint up to height h, skipping cells another object (or a doorway) already uses.
	 * Returns the positions that were filled.
	 */
	private List<BlockPos> fill(int x0, int z0, int w, int d, int h, BlockState s) {
		List<BlockPos> filled = new ArrayList<>();
		for (int x = x0; x < x0 + w; x++)
			for (int z = z0; z < z0 + d; z++)
				for (int y = 1; y <= Math.min(h, H); y++)
					if (inside(x, z) && claim(x, y, z)) {
						set(x, y, z, s);
						filled.add(origin.offset(x, y, z));
					}
		return filled;
	}

	// ------------------------------------------------------------------------------------- finishing


	// ------------------------------------------------------------------------------------- photo walls

	/**
	 * Covers every wall block with its piece of the wall texture (flat paint plus whatever hangs on the wall):
	 * an invisible item frame holding a locked map. The paint is exactly the colour of the wall block behind
	 * it, so frames and any bare wall block (behind furniture, past a stair) look the same.
	 * Players can punch the frame off (or break the wall) like any other frame. Only blocks that are a
	 * window or door, or have a painting or furniture directly in front, cannot hold a frame.
	 */
	private void hangWallTextures() {
		if (!plan.has("textures") || !plan.has("_dir")) return;
		JsonObject tex = plan.getAsJsonObject("textures");
		int px = tex.get("px").getAsInt();
		boolean dither = !tex.has("dither") || tex.get("dither").getAsBoolean();
		Path dir = Path.of(plan.get("_dir").getAsString());
		int frames = 0;
		for (Map.Entry<String, JsonElement> entry : tex.getAsJsonObject("walls").entrySet()) {
			String side = entry.getKey();
			JsonObject w = entry.getValue().getAsJsonObject();
			boolean northOrSouth = side.equals("north") || side.equals("south");
			int len = northOrSouth ? W : D;
			boolean flip = w.get("flip").getAsBoolean();
			JsonArray tiles = w.getAsJsonArray("tiles");
			byte[] pixels;
			try {
				// Inside faces of north/south walls point along Z (shaded x0.8), of west/east walls along X (x0.6).
				float shade = northOrSouth ? 0.8f : 0.6f;
				// A wall may override the room's dithering (a patterned wall next to flat-colour wall items).
				boolean wallDither = w.has("dither") ? w.get("dither").getAsBoolean() : dither;
				pixels = WallTextures.convert(dir.resolve(w.get("image").getAsString()), len * 128, H * 128, wallDither, shade);
			} catch (Exception ex) {
				EverythingCubed.LOGGER.warn("Skipping photo texture for the {} wall", side, ex);
				continue;
			}
			Direction into = dir(side).getOpposite();
			for (int row = 0; row < H; row++) {
				JsonArray r = tiles.get(row).getAsJsonArray();
				for (int u = 0; u < len; u++) {
					if (r.get(u).getAsInt() == 0) continue;
					int[] wc = wallCell(side, u), in = insideCell(side, u);
					BlockPos wallPos = origin.offset(wc[0], row + 1, wc[1]);
					if (!level.getBlockState(wallPos).isCollisionShapeFullBlock(level, wallPos)) continue; // window / door
					BlockPos framePos = origin.offset(in[0], row + 1, in[1]);
					BlockState inFront = level.getBlockState(framePos);
				if (!inFront.isAir() && !inFront.is(Blocks.LIGHT)) continue;
					// A regular frame is lit by the world exactly like the wall blocks around it.
					ItemFrame frame = new ItemFrame(level, framePos, into);
					if (!frame.survives()) continue; // painting or another frame already there
					int col = flip ? len - 1 - u : u, imgRow = H - 1 - row;
					frame.setItem(mapOf(pixels, len * 128, col * 128, imgRow * 128), false);
					frame.setInvisible(true);
					level.addFreshEntity(frame);
					undo.rememberEntity(frame);
					frames++;
				}
			}
		}
		EverythingCubed.LOGGER.info("Hung {} photo wall tiles (texture {} px/block)", frames, px);
	}

	/** A locked filled map showing the 128x128 window of {@code pixels} starting at (x0, y0). */
	private ItemStack mapOf(byte[] pixels, int stride, int x0, int y0) {
		MapId id = level.getFreeMapId();
		MapItemSavedData data = MapItemSavedData.createFresh(0, 0, (byte) 0, false, false, level.dimension()).locked();
		for (int y = 0; y < 128; y++)
			for (int x = 0; x < 128; x++)
				data.setColor(x, y, pixels[(y0 + y) * stride + x0 + x]);
		level.setMapData(id, data);
		ItemStack stack = new ItemStack(Items.FILLED_MAP);
		stack.set(DataComponents.MAP_ID, id);
		return stack;
	}

	/** Recompute shape-dependent states (pane/fence connections, stair corners) once everything exists. */
	private void finishShapes() {
		for (Map.Entry<BlockPos, BlockState> e : placed.entrySet()) {
			BlockState cur = level.getBlockState(e.getKey());
			if (cur.isAir()) continue;
			BlockState upd = Block.updateFromNeighbourShapes(cur, level, e.getKey());
			if (upd != cur && !upd.isAir()) level.setBlock(e.getKey(), upd, Block.UPDATE_CLIENTS);
		}
	}
}

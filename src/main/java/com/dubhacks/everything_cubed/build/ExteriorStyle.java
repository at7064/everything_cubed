package com.dubhacks.everything_cubed.build;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * The look of a house from outside: an outer layer one block outside the walls (the photo-matched walls inside are
 * untouched) and a roof. Chosen by name or by vibe words ({@link #fromWords}), or from the room itself
 * ({@link #auto}). Blocks are ids; {@link HouseBuilder} turns them into block states.
 *
 * @param base     the bottom row, at ground level
 * @param wall     the outer wall
 * @param corner   corner posts (and timber posts, see {@code postEvery}), upright
 * @param trim     the top band and the frame around windows and doors
 * @param postEvery a timber post every this many blocks along a wall (0: none)
 * @param logWall  the wall block is a log laid along the wall (a log cabin)
 * @param roof     roof stairs and slabs (wood or stone name, e.g. "spruce", "deepslate_tile"); null: a flat roof
 * @param gable    the gable ends under a pitched roof
 * @param chimney  a stone chimney with a smoking campfire at one gable end
 */
public record ExteriorStyle(String name, String base, String wall, String corner, String trim, int postEvery,
		boolean logWall, String roof, String gable, boolean chimney) {

	public static final ExteriorStyle COTTAGE = new ExteriorStyle("cottage", "cobblestone", "calcite",
			"stripped_spruce_log", "stripped_spruce_log", 4, false, "spruce", "spruce_planks", false);
	public static final ExteriorStyle MODERN = new ExteriorStyle("modern", "light_gray_concrete", "white_concrete",
			"white_concrete", "black_concrete", 0, false, null, null, false);
	public static final ExteriorStyle BRICK = new ExteriorStyle("brick", "stone_bricks", "bricks",
			"stone_bricks", "stone_bricks", 0, false, "deepslate_tile", "bricks", false);
	public static final ExteriorStyle CABIN = new ExteriorStyle("cabin", "cobblestone", "spruce_log",
			"dark_oak_log", "stripped_spruce_log", 0, true, "dark_oak", "spruce_planks", true);

	/** Every style, by name, in the order ties are broken. */
	public static final Map<String, ExteriorStyle> ALL = new LinkedHashMap<>();

	static {
		for (ExteriorStyle s : new ExteriorStyle[]{COTTAGE, MODERN, BRICK, CABIN}) ALL.put(s.name, s);
	}

	/** Vibe words per style (other names for it count too). */
	private static final Map<String, Set<String>> WORDS = Map.of(
			"cottage", Set.of("cottage", "cozy", "cosy", "cute", "garden", "flower", "flowers", "fairy", "storybook",
					"village", "countryside", "country", "farm", "farmhouse", "english", "tudor", "timber", "quaint",
					"charming", "hobbit", "medieval"),
			"modern", Set.of("modern", "minimal", "minimalist", "clean", "sleek", "white", "contemporary", "futuristic",
					"glass", "city", "urban", "scandinavian", "simple", "office", "apartment", "loft", "new"),
			"brick", Set.of("brick", "bricks", "campus", "college", "school", "university", "uni", "dorm", "classroom",
					"classic", "library", "academic", "red", "victorian", "historic", "old", "hall", "uw"),
			"cabin", Set.of("cabin", "log", "logs", "wood", "wooden", "forest", "woods", "mountain", "lodge", "rustic",
					"camping", "camp", "winter", "snow", "lake", "chalet", "warm", "fireplace", "chimney"));

	/** Furniture that says the room is a dorm, classroom or other campus room. */
	private static final Set<String> CAMPUS = Set.of("whiteboard", "student_desk", "podium", "projector",
			"projection_screen", "exit_sign", "locker", "chair_stack", "fire_extinguisher", "drinking_fountain");
	private static final Set<String> PLAIN_WHITE = Set.of("white_concrete", "light_gray_concrete", "smooth_quartz",
			"quartz_block", "calcite");

	/**
	 * The style named or best described by {@code words} (a style name, or vibe words such as "cozy cabin in the
	 * woods"), or null when no word matches. The style with the most matching words wins; ties go in {@link #ALL}
	 * order.
	 */
	public static ExteriorStyle fromWords(String words) {
		ExteriorStyle best = null;
		int bestScore = 0;
		String[] tokens = words.toLowerCase(Locale.ROOT).split("[^a-z]+");
		for (ExteriorStyle s : ALL.values()) {
			int score = 0;
			for (String t : tokens) if (WORDS.get(s.name).contains(t)) score += t.equals(s.name) ? 3 : 1;
			if (score > bestScore) {
				best = s;
				bestScore = score;
			}
		}
		return best;
	}

	/** A style that suits the room in {@code plan}, and why (for chat). */
	public static Pick auto(JsonObject plan) {
		for (String list : new String[]{"objects", "openings"}) {
			if (!plan.has(list)) continue;
			for (JsonElement e : plan.getAsJsonArray(list)) {
				JsonObject o = e.getAsJsonObject();
				String type = o.has("type") ? o.get("type").getAsString() : "";
				if (CAMPUS.contains(type)) return new Pick(BRICK, "campus furniture (" + type.replace('_', ' ') + ")");
				if (type.equals("bed") && o.has("style")) return new Pick(BRICK, "dorm loft beds");
			}
		}
		String wall = plan.has("wall_main") ? bare(plan.get("wall_main").getAsString()) : "";
		if (PLAIN_WHITE.contains(wall)) return new Pick(MODERN, "plain white walls");
		String floor = "";
		if (plan.has("floor")) {
			JsonArray pal = plan.getAsJsonObject("floor").getAsJsonArray("palette");
			if (pal != null && !pal.isEmpty()) floor = bare(pal.get(0).getAsString());
		}
		if (floor.startsWith("dark_oak") || floor.startsWith("spruce") || floor.contains("log"))
			return new Pick(CABIN, "a dark wood floor");
		return new Pick(COTTAGE, "the default");
	}

	/** The style set in the config ({@code setting}: "auto" or a style name) for this room. */
	public static Pick choose(String setting, JsonObject plan) {
		ExteriorStyle s = setting == null ? null : ALL.get(setting.toLowerCase(Locale.ROOT));
		return s != null ? new Pick(s, null) : auto(plan);
	}

	private static String bare(String id) {
		return id.startsWith("minecraft:") ? id.substring("minecraft:".length()) : id;
	}

	/** A chosen style; {@code reason} says why when it was picked automatically (else null). */
	public record Pick(ExteriorStyle style, String reason) {
	}
}

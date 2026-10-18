package com.dubhacks.everything_cubed.build;

import com.google.gson.JsonParser;
import com.google.gson.JsonObject;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** ExteriorStyle: vibe words to a style, and the automatic pick from a room's plan. Run: ./gradlew test */
class ExteriorStyleTest {
	private static JsonObject plan(String json) {
		return JsonParser.parseString(json).getAsJsonObject();
	}

	@Test
	void namesAndVibeWordsPickTheClosestStyle() {
		assertSame(ExteriorStyle.MODERN, ExteriorStyle.fromWords("modern"));
		assertSame(ExteriorStyle.BRICK, ExteriorStyle.fromWords("Campus"));
		assertSame(ExteriorStyle.CABIN, ExteriorStyle.fromWords("a cozy log cabin in the woods"), "cabin beats cozy");
		assertSame(ExteriorStyle.COTTAGE, ExteriorStyle.fromWords("cute cottage with a flower garden"));
		assertSame(ExteriorStyle.MODERN, ExteriorStyle.fromWords("sleek, minimal & white!"));
		assertSame(ExteriorStyle.BRICK, ExteriorStyle.fromWords("old red brick university hall"));
	}

	@Test
	void wordsThatMatchNothingGiveNull() {
		assertNull(ExteriorStyle.fromWords("spaceship"));
		assertNull(ExteriorStyle.fromWords(""));
	}

	@Test
	void autoPicksFromTheRoom() {
		assertSame(ExteriorStyle.BRICK, ExteriorStyle.auto(plan(
				"{\"objects\":[{\"type\":\"bed\",\"style\":\"loft\"}],\"openings\":[]}")).style(), "dorm lofts");
		assertSame(ExteriorStyle.BRICK, ExteriorStyle.auto(plan(
				"{\"objects\":[],\"openings\":[{\"type\":\"whiteboard\"}]}")).style(), "classroom");
		assertSame(ExteriorStyle.MODERN, ExteriorStyle.auto(plan(
				"{\"objects\":[{\"type\":\"bed\"}],\"wall_main\":\"minecraft:white_concrete\"}")).style());
		assertSame(ExteriorStyle.CABIN, ExteriorStyle.auto(plan(
				"{\"wall_main\":\"white_terracotta\",\"floor\":{\"palette\":[\"spruce_planks\"]}}")).style());
		ExteriorStyle.Pick p = ExteriorStyle.auto(plan("{\"wall_main\":\"white_terracotta\",\"floor\":{\"palette\":[\"oak_planks\"]}}"));
		assertSame(ExteriorStyle.COTTAGE, p.style());
		assertNotNull(p.reason(), "an automatic pick says why");
	}

	@Test
	void aChosenStyleWinsOverAuto() {
		JsonObject dorm = plan("{\"objects\":[{\"type\":\"bed\",\"style\":\"loft\"}]}");
		ExteriorStyle.Pick p = ExteriorStyle.choose("cabin", dorm);
		assertSame(ExteriorStyle.CABIN, p.style());
		assertNull(p.reason());
		assertSame(ExteriorStyle.BRICK, ExteriorStyle.choose("auto", dorm).style());
		assertSame(ExteriorStyle.BRICK, ExteriorStyle.choose("no-such-style", dorm).style(), "unknown names fall back to auto");
	}
}

package com.dubhacks.everything_cubed.analyzer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** AnalyzerRunner.pruneScans: which phone scans are deleted when a new one starts. Run: ./gradlew test */
class PruneScansTest {
	@TempDir
	Path photos;

	/** A finished phone scan (it has a capture.json, a photo and a nested folder), last modified at minute {@code age} ago. */
	private Path scan(String name, int age) throws IOException {
		Path d = photos.resolve(name);
		Files.createDirectories(d.resolve("points"));
		Files.writeString(d.resolve("capture.json"), "{}");
		Files.writeString(d.resolve("frame_0001.jpg"), "x");
		Files.writeString(d.resolve("points/frame_0001.points"), "x");
		Files.setLastModifiedTime(d, FileTime.fromMillis(1_000_000_000_000L - age * 60_000L));
		return d;
	}

	/** The scan starting now: its folder exists but its capture.json is only written when it finishes. */
	private Path starting() throws IOException {
		return Files.createDirectories(photos.resolve("room-new"));
	}

	@Test
	void keepsNewestCountingTheScanStartingNow() throws IOException {
		Path[] old = new Path[7];
		for (int i = 0; i < 7; i++) old[i] = scan("room-" + i, i + 1);  // room-0 newest
		Path current = starting();
		List<Path> deleted = AnalyzerRunner.pruneScans(photos, current, 5);
		assertEquals(List.of(old[4], old[5], old[6]), deleted);
		for (int i = 0; i < 7; i++) assertEquals(i < 4, Files.exists(old[i]), "room-" + i);
		assertTrue(Files.exists(current));
		try (var s = Files.list(photos)) {
			assertEquals(5, s.count(), "4 earlier scans + the new one");
		}
	}

	@Test
	void neverTouchesHandMadeFoldersOrLooseFiles() throws IOException {
		Path hand = Files.createDirectories(photos.resolve("classroom"));
		Files.writeString(hand.resolve("IMG_1.jpg"), "x");
		Files.setLastModifiedTime(hand, FileTime.fromMillis(0));  // oldest of all
		Path loose = Files.writeString(photos.resolve("notes.txt"), "x");
		for (int i = 0; i < 3; i++) scan("room-" + i, i + 1);
		List<Path> deleted = AnalyzerRunner.pruneScans(photos, starting(), 2);
		assertEquals(List.of(photos.resolve("room-1"), photos.resolve("room-2")), deleted);
		assertTrue(Files.exists(hand.resolve("IMG_1.jpg")));
		assertTrue(Files.exists(loose));
	}

	@Test
	void neverDeletesTheCurrentScanEvenIfItHasACaptureAndIsOldest() throws IOException {
		Path current = scan("room-current", 999);
		scan("room-a", 1);
		scan("room-b", 2);
		assertEquals(List.of(photos.resolve("room-b")), AnalyzerRunner.pruneScans(photos, current, 2));
		assertTrue(Files.exists(current.resolve("capture.json")));
	}

	@Test
	void zeroKeepsEverythingAndFewScansDeleteNothing() throws IOException {
		for (int i = 0; i < 3; i++) scan("room-" + i, i + 1);
		assertTrue(AnalyzerRunner.pruneScans(photos, starting(), 0).isEmpty());
		assertTrue(AnalyzerRunner.pruneScans(photos, starting(), 4).isEmpty());
		assertTrue(AnalyzerRunner.pruneScans(photos.resolve("missing"), starting(), 5).isEmpty());
		try (var s = Files.list(photos)) {
			assertEquals(4, s.count());
		}
	}

	@Test
	void keepOneLeavesOnlyTheNewScan() throws IOException {
		for (int i = 0; i < 3; i++) scan("room-" + i, i + 1);
		Path current = starting();
		assertEquals(3, AnalyzerRunner.pruneScans(photos, current, 1).size());
		try (var s = Files.list(photos)) {
			assertEquals(List.of(current), s.toList());
		}
	}
}

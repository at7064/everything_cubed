package com.dubhacks.everything_cubed.build;

import net.minecraft.world.level.material.MapColor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Converts wall textures into Minecraft map pixels, matched against every colour a map can display in this
 * game version. Optional Floyd–Steinberg dithering runs over the whole wall image, so tiles have no seams.
 */
final class WallTextures {
	private static int[] paletteRgb;
	private static byte[] paletteId;
	/** 5-bit-per-channel lookup of the nearest palette index. */
	private static short[] nearestCache;

	private WallTextures() {
	}

	private static synchronized void initPalette() {
		if (paletteRgb != null) return;
		List<int[]> cols = new ArrayList<>();
		for (int id = 1; id < 64; id++) {
			MapColor c = MapColor.byId(id);
			if (c == null || c == MapColor.NONE || c.col == 0) continue;
			for (MapColor.Brightness b : MapColor.Brightness.values()) {
				cols.add(new int[]{c.calculateARGBColor(b), c.getPackedId(b)});
			}
		}
		paletteRgb = new int[cols.size()];
		paletteId = new byte[cols.size()];
		for (int i = 0; i < cols.size(); i++) {
			paletteRgb[i] = cols.get(i)[0] & 0xFFFFFF;
			paletteId[i] = (byte) cols.get(i)[1];
		}
		// calculateARGBColor's channel order differs between versions; detect it from a known colour.
		int red = MapColor.COLOR_RED.calculateARGBColor(MapColor.Brightness.HIGH) & 0xFFFFFF;
		if ((red & 0xFF) > ((red >> 16) & 0xFF)) { // stored as ABGR, swap to RGB
			for (int i = 0; i < paletteRgb.length; i++) {
				int c = paletteRgb[i];
				paletteRgb[i] = ((c & 0xFF) << 16) | (c & 0xFF00) | ((c >> 16) & 0xFF);
			}
		}
		nearestCache = new short[32 * 32 * 32];
		Arrays.fill(nearestCache, (short) -1);
	}

	private static int nearest(int r, int g, int b) {
		int key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
		short cached = nearestCache[key];
		if (cached >= 0) return cached;
		int best = 0;
		long bestD = Long.MAX_VALUE;
		for (int i = 0; i < paletteRgb.length; i++) {
			int c = paletteRgb[i];
			int dr = ((c >> 16) & 0xFF) - r, dg = ((c >> 8) & 0xFF) - g, db = (c & 0xFF) - b;
			int rm = (((c >> 16) & 0xFF) + r) / 2; // "redmean" perceptual weighting
			long d = (long) (512 + rm) * dr * dr / 256 + 4L * dg * dg + (long) (767 - rm) * db * db / 256;
			if (d < bestD) { bestD = d; best = i; }
		}
		nearestCache[key] = (short) best;
		return best;
	}

	/**
	 * Map colour ids for the whole image, row-major, scaled (nearest neighbour) to outWidth x outHeight.
	 * Transparent pixels become map colour 0. {@code shade} darkens the picture the way Minecraft shades a
	 * block face pointing that way (maps in frames get no such shading), so frames match bare wall blocks.
	 */
	static byte[] convert(Path imageFile, int outWidth, int outHeight, boolean dither, float shade) throws IOException {
		initPalette();
		BufferedImage src = ImageIO.read(imageFile.toFile());
		if (src == null) throw new IOException("unreadable texture " + imageFile);
		int w = outWidth, h = outHeight;
		float[] err = new float[w * h * 3];
		boolean[] clear = new boolean[w * h];
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				int argb = src.getRGB(x * src.getWidth() / w, y * src.getHeight() / h), i = (y * w + x) * 3;
				clear[y * w + x] = (argb >>> 24) < 128;
				err[i] = ((argb >> 16) & 0xFF) * shade;
				err[i + 1] = ((argb >> 8) & 0xFF) * shade;
				err[i + 2] = (argb & 0xFF) * shade;
			}
		byte[] out = new byte[w * h];
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				if (clear[y * w + x]) continue; // stays 0 = transparent
				int i = (y * w + x) * 3;
				int r = clamp(err[i]), g = clamp(err[i + 1]), b = clamp(err[i + 2]);
				int p = nearest(r, g, b);
				out[y * w + x] = paletteId[p];
				if (!dither) continue;
				int c = paletteRgb[p];
				float er = r - ((c >> 16) & 0xFF), eg = g - ((c >> 8) & 0xFF), eb = b - (c & 0xFF);
				spread(err, w, h, x + 1, y, er, eg, eb, 7 / 16f);
				spread(err, w, h, x - 1, y + 1, er, eg, eb, 3 / 16f);
				spread(err, w, h, x, y + 1, er, eg, eb, 5 / 16f);
				spread(err, w, h, x + 1, y + 1, er, eg, eb, 1 / 16f);
			}
		return out;
	}

	private static void spread(float[] err, int w, int h, int x, int y, float er, float eg, float eb, float k) {
		if (x < 0 || y < 0 || x >= w || y >= h) return; // (error spread into transparent pixels is simply unused)
		int i = (y * w + x) * 3;
		err[i] += er * k;
		err[i + 1] += eg * k;
		err[i + 2] += eb * k;
	}

	private static int clamp(float v) {
		return v < 0 ? 0 : v > 255 ? 255 : (int) v;
	}
}

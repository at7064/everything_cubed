package com.dubhacks.everything_cubed.stream;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Optional;

/**
 * One message of the phone stream protocol, shared with the iOS scanner app.
 * On the wire: u32 header length, UTF-8 JSON header, u32 payload length, payload (all big-endian).
 * The header always has {@code "type"}; binary attachments are listed in {@code "blobs"}
 * as {@code {"name", "length", "format"}} and stored back to back in the payload.
 */
public record StreamMessage(JsonObject header, byte[] payload) {
	public static final int PROTOCOL = 1;
	private static final int MAX_HEADER = 1 << 20;
	private static final int MAX_PAYLOAD = 64 << 20;

	public String type() {
		return header.has("type") ? header.get("type").getAsString() : "";
	}

	public String string(String key, String fallback) {
		JsonElement e = header.get(key);
		return e == null || e.isJsonNull() ? fallback : e.getAsString();
	}

	/** The named attachment, if the message carries one. */
	public Optional<byte[]> blob(String name) {
		JsonArray blobs = header.getAsJsonArray("blobs");
		if (blobs == null) return Optional.empty();
		int offset = 0;
		for (JsonElement e : blobs) {
			JsonObject b = e.getAsJsonObject();
			int len = b.get("length").getAsInt();
			if (b.get("name").getAsString().equals(name)) {
				if (offset + len > payload.length) return Optional.empty();
				return Optional.of(Arrays.copyOfRange(payload, offset, offset + len));
			}
			offset += len;
		}
		return Optional.empty();
	}

	public static StreamMessage of(String type) {
		JsonObject h = new JsonObject();
		h.addProperty("type", type);
		return new StreamMessage(h, new byte[0]);
	}

	public StreamMessage with(String key, String value) {
		header.addProperty(key, value);
		return this;
	}

	public StreamMessage with(String key, Number value) {
		header.addProperty(key, value);
		return this;
	}

	public StreamMessage with(String key, JsonElement value) {
		header.add(key, value);
		return this;
	}

	public static StreamMessage read(DataInputStream in) throws IOException {
		int hl = in.readInt();
		if (hl <= 0 || hl > MAX_HEADER) throw new IOException("bad header length " + hl);
		byte[] h = in.readNBytes(hl);
		if (h.length != hl) throw new IOException("connection closed");
		int pl = in.readInt();
		if (pl < 0 || pl > MAX_PAYLOAD) throw new IOException("bad payload length " + pl);
		byte[] p = in.readNBytes(pl);
		if (p.length != pl) throw new IOException("connection closed");
		JsonObject header = JsonParser.parseString(new String(h, StandardCharsets.UTF_8)).getAsJsonObject();
		return new StreamMessage(header, p);
	}

	public void write(DataOutputStream out) throws IOException {
		byte[] h = header.toString().getBytes(StandardCharsets.UTF_8);
		out.writeInt(h.length);
		out.write(h);
		out.writeInt(payload.length);
		out.write(payload);
		out.flush();
	}
}

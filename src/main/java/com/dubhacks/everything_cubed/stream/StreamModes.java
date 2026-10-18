package com.dubhacks.everything_cubed.stream;

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Session modes a phone can ask for in {@code begin}. {@code capture} records a scan and builds it;
 * {@code spectate} mirrors the room into the world live.
 * New modes register here without touching the connection code.
 */
public final class StreamModes {
	private static final Map<String, StreamSession.Factory> MODES = new ConcurrentHashMap<>();

	static {
		register("capture", CaptureRecorder::new);
		register("spectate", SpectateSession::new);
	}

	private StreamModes() {
	}

	public static void register(String mode, StreamSession.Factory factory) {
		MODES.put(mode, factory);
	}

	public static Optional<StreamSession.Factory> get(String mode) {
		return Optional.ofNullable(MODES.get(mode));
	}

	public static Set<String> names() {
		return Set.copyOf(MODES.keySet());
	}
}

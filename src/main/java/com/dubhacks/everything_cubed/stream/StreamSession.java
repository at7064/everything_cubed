package com.dubhacks.everything_cubed.stream;

import java.io.IOException;

/**
 * What a phone does after {@code begin}: every following message goes to the session made for the
 * requested mode (see {@link StreamModes}). Called on the connection's own thread, never the server thread.
 */
public interface StreamSession {
	void onMessage(StreamMessage msg) throws IOException;

	/** The phone sent {@code end}. The session is finished after this. */
	void onEnd(StreamMessage msg) throws IOException;

	/** The connection dropped before {@code end}. */
	default void onDisconnect() {
	}

	@FunctionalInterface
	interface Factory {
		StreamSession open(PhoneConnection phone, StreamMessage begin) throws IOException;
	}
}

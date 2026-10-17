package com.dubhacks.everything_cubed.stream;

import com.dubhacks.everything_cubed.EverythingCubed;
import net.minecraft.server.MinecraftServer;

import javax.jmdns.JmDNS;
import javax.jmdns.ServiceInfo;
import java.io.IOException;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * TCP server the iOS app streams to. Started by {@code /everything_cubed phone}, stopped with the game server.
 * Advertised on the LAN over Bonjour as {@value #SERVICE_TYPE} so the app finds it without typing an address.
 */
public final class PhoneServer {
	public static final String SERVICE_TYPE = "_everythingcubed._tcp.local.";
	public static final int DEFAULT_PORT = 25599;
	private static final SecureRandom RANDOM = new SecureRandom();
	/** Pairing code -> player. A player's code stays valid until they ask for a new one. */
	private static final Map<String, UUID> CODES = new ConcurrentHashMap<>();
	private static ServerSocket socket;
	private static final List<JmDNS> BONJOUR = new ArrayList<>();

	private PhoneServer() {
	}

	/** Starts listening if needed. Returns the port. */
	public static synchronized int start(MinecraftServer server, int port) throws IOException {
		if (socket != null && !socket.isClosed()) return socket.getLocalPort();
		ServerSocket ss = new ServerSocket(port);
		socket = ss;
		Thread accept = new Thread(() -> acceptLoop(ss, server), "everything_cubed-phone");
		accept.setDaemon(true);
		accept.start();
		// Bonjour setup can take a few seconds per interface; keep it off the server thread.
		Thread bonjour = new Thread(() -> advertise(ss.getLocalPort()), "everything_cubed-bonjour");
		bonjour.setDaemon(true);
		bonjour.start();
		EverythingCubed.LOGGER.info("Phone server listening on port {}", ss.getLocalPort());
		return ss.getLocalPort();
	}

	public static synchronized void stop() {
		CODES.clear();
		for (JmDNS j : BONJOUR) {
			try {
				j.unregisterAllServices();
				j.close();
			} catch (IOException ignored) {
			}
		}
		BONJOUR.clear();
		if (socket != null) {
			try {
				socket.close();
			} catch (IOException ignored) {
			}
			socket = null;
		}
	}

	/** A fresh 6-digit code for {@code player}; their previous code stops working. */
	public static String pair(UUID player) {
		CODES.values().removeIf(player::equals);
		String code;
		do code = String.format("%06d", RANDOM.nextInt(1_000_000)); while (CODES.containsKey(code));
		CODES.put(code, player);
		return code;
	}

	static Optional<UUID> claim(String code) {
		return Optional.ofNullable(CODES.get(code.trim()));
	}

	/** This machine's LAN IPv4 addresses, for typing into the app when Bonjour is blocked. */
	public static List<String> addresses() {
		List<String> out = new ArrayList<>();
		for (InetAddress a : lanAddresses()) out.add(a.getHostAddress());
		return out;
	}

	private static void acceptLoop(ServerSocket ss, MinecraftServer server) {
		while (!ss.isClosed()) {
			try {
				Socket s = ss.accept();
				s.setTcpNoDelay(true);
				s.setSoTimeout(5 * 60_000);
				Thread t = new Thread(new PhoneConnection(s, server), "everything_cubed-phone-" + s.getInetAddress().getHostAddress());
				t.setDaemon(true);
				t.start();
			} catch (IOException e) {
				if (!ss.isClosed()) EverythingCubed.LOGGER.warn("Phone accept failed", e);
			}
		}
	}

	private static void advertise(int port) {
		String name = "Everything Cubed on " + hostName();
		for (InetAddress a : lanAddresses()) {
			try {
				JmDNS j = JmDNS.create(a, hostName());
				j.registerService(ServiceInfo.create(SERVICE_TYPE, name, port, "protocol=" + StreamMessage.PROTOCOL));
				synchronized (PhoneServer.class) {
					BONJOUR.add(j);
				}
			} catch (IOException e) {
				EverythingCubed.LOGGER.warn("Bonjour advertising on {} failed; the app can still connect by address", a, e);
			}
		}
	}

	private static String hostName() {
		try {
			return InetAddress.getLocalHost().getHostName().replaceFirst("\\.local$", "");
		} catch (IOException e) {
			return "Minecraft";
		}
	}

	private static List<InetAddress> lanAddresses() {
		List<InetAddress> out = new ArrayList<>();
		try {
			for (NetworkInterface n : Collections.list(NetworkInterface.getNetworkInterfaces())) {
				if (!n.isUp() || n.isLoopback() || n.isVirtual() || n.isPointToPoint()) continue;
				for (InetAddress a : Collections.list(n.getInetAddresses()))
					if (a instanceof Inet4Address && !a.isLinkLocalAddress()) out.add(a);
			}
		} catch (SocketException e) {
			EverythingCubed.LOGGER.warn("Could not list network interfaces", e);
		}
		return out;
	}
}

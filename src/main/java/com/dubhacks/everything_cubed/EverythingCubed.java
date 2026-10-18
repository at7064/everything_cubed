package com.dubhacks.everything_cubed;

import com.dubhacks.everything_cubed.analyzer.AnalyzerRunner;
import com.dubhacks.everything_cubed.build.FurnitureInteractions;
import com.dubhacks.everything_cubed.command.EverythingCubedCommand;
import net.fabricmc.api.ModInitializer;
import com.dubhacks.everything_cubed.stream.PhoneServer;
import com.dubhacks.everything_cubed.stream.SpectateSession;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;

public class EverythingCubed implements ModInitializer {
	public static final String MOD_ID = "everything_cubed";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
				EverythingCubedCommand.register(dispatcher));
		FurnitureInteractions.register();
		SpectateSession.register();
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			SpectateSession.stopAll();
			PhoneServer.stop();
		});
		// Where `/everything_cubed build <name>` looks for photo folders.
		try {
			Files.createDirectories(AnalyzerRunner.photosDir());
		} catch (IOException e) {
			LOGGER.warn("Could not create the everything_cubed/photos folder", e);
		}
	}
}

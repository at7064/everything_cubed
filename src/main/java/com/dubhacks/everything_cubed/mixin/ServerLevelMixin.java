package com.dubhacks.everything_cubed.mixin;

import com.dubhacks.everything_cubed.build.FurnitureInteractions;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.players.SleepStatus;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

import java.util.List;

/** Players lying in a bed in the daytime must not skip the clock to the next morning (and be woken by it). */
@Mixin(ServerLevel.class)
abstract class ServerLevelMixin {
	@WrapOperation(method = "tick", at = @At(value = "INVOKE",
			target = "Lnet/minecraft/server/players/SleepStatus;areEnoughDeepSleeping(ILjava/util/List;)Z"))
	private boolean everything_cubed$restersKeepTheDay(SleepStatus status, int percent, List<ServerPlayer> players,
			Operation<Boolean> original) {
		return original.call(status, percent, players)
				&& players.stream().noneMatch(p -> p.isSleeping() && p.entityTags().contains(FurnitureInteractions.RESTING));
	}
}

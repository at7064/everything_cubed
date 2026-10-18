package com.dubhacks.everything_cubed.mixin;

import com.dubhacks.everything_cubed.build.FurnitureInteractions;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import net.minecraft.world.attribute.BedRule;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.Level;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

/**
 * Lets a player lie in a built bed in the daytime ({@link FurnitureInteractions#RESTING}): every tick vanilla wakes a
 * sleeper whose bed may not be slept in now. Once it may (night falls), the rest becomes ordinary sleep.
 */
@Mixin(Player.class)
abstract class PlayerMixin {
	@WrapOperation(method = "tick", at = @At(value = "INVOKE",
			target = "Lnet/minecraft/world/attribute/BedRule;canSleep(Lnet/minecraft/world/level/Level;)Z"))
	private boolean everything_cubed$restInDaytime(BedRule rule, Level level, Operation<Boolean> original) {
		Player self = (Player) (Object) this;
		if (original.call(rule, level)) {
			self.removeTag(FurnitureInteractions.RESTING);
			return true;
		}
		return self.entityTags().contains(FurnitureInteractions.RESTING);
	}
}

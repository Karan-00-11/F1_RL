"""Reward shaping utilities for the F1 environment.

This module provides a multi-phase reward function with soft phase weights.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class RewardFunction:
    def __init__(
        self,
        ds: float,
        target_seg_len: float,
        progress_m: float,
        total_length_m: float,
        velocity_before: float,
        velocity_after: float,
        v_max_seg: float,
        curvature: float,
        ay_demand: float,
        ay_limit: float,
        throttle_eff: float,
        brake_eff: float,
        steering_cmd: float,
        soc_before: float,
        soc_after: float,
        battery_status: str,
        regen_intensity: float,
        deploy_level: float,
        tire_wear: float,
        slip_ratio: float,
        slip_angle_rad: float,
        step_count: int = 0,
        max_episode_steps: int = 1000,
    ):
        self.ds = float(ds)
        self.progress_m = float(progress_m)
        self.target_seg_len = float(target_seg_len)
        self.total_length_m = float(total_length_m)
        self.velocity_before = float(velocity_before)
        self.velocity_after = float(velocity_after)
        self.v_max_seg = float(v_max_seg)
        self.curvature = float(curvature)
        self.ay_demand = float(ay_demand)
        self.ay_limit = float(ay_limit)
        self.throttle_eff = float(throttle_eff)
        self.brake_eff = float(brake_eff)
        self.steering_cmd = float(steering_cmd)
        self.soc_before = float(soc_before)
        self.soc_after = float(soc_after)
        self.battery_status = str(battery_status).upper()
        self.regen_intensity = float(regen_intensity)
        self.deploy_level = float(deploy_level)
        self.tire_wear = float(tire_wear)
        self.slip_ratio = float(slip_ratio)
        self.slip_angle_rad = float(slip_angle_rad)
        self.step_count = int(step_count)
        self.max_episode_steps = int(max_episode_steps)

    @staticmethod
    def _sigmoid(z: float, k: float) -> float:
        return float(1.0 / (1.0 + np.exp(-k * z)))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def _phase_weights(self) -> Dict[str, float]:
        """
        Soft phase gating.

        The phase weights are mostly driven by track/vehicle state.
        Energy is handled as a small contextual modifier instead of a dominant phase.
        """
        v_ratio = self.velocity_after / max(self.v_max_seg, 1e-3)
        ay_ratio = self.ay_demand / max(self.ay_limit, 1e-3)
        acceleration = self.velocity_after - self.velocity_before

        g_curve = self._sigmoid(self.curvature - 0.01, 10.0)
        g_brake = self._sigmoid(self.brake_eff - 0.15, 15.0)
        g_throttle = self._sigmoid(self.throttle_eff - 0.30, 10.0)
        g_accel = self._sigmoid(acceleration - 0.05, 15.0)
        g_limit = self._sigmoid(ay_ratio - 0.85, 10.0)
        g_overspeed = self._sigmoid(v_ratio - 0.95, 10.0)

        raw_straight = (1.0 - g_curve) * (1.0 - g_brake)
        raw_entry = g_curve * g_brake * (0.6 + 0.4 * g_overspeed)
        raw_corner = g_curve * g_limit * (1.0 - 0.3 * g_throttle)
        raw_exit = g_curve * g_throttle * g_accel * (1.0 - g_brake)

        raw = np.array([raw_straight, raw_entry, raw_corner, raw_exit], dtype=float) + 1e-6
        tau = 0.8
        scaled_raw = raw / max(tau, 1e-6)
        scaled_raw -= np.max(scaled_raw)
        logits = np.exp(scaled_raw)
        phase_w = logits / np.sum(logits)

        # Energy opportunity is intentionally capped; it should never dominate the phase logic.
        deploy_opportunity = (1.0 - g_curve) * g_throttle * (1.0 - g_brake)
        regen_opportunity = g_brake * (1.0 - g_throttle) * (0.35 + 0.65 * g_curve)
        energy_mix = 0.55 * deploy_opportunity + 0.45 * regen_opportunity
        w_energy = self._clamp(0.08 + 0.08 * self._sigmoid(energy_mix - 0.25, 8.0), 0.08, 0.16)

        phase_scale = 1.0 - w_energy
        return {
            "straight": float(phase_w[0] * phase_scale),
            "entry": float(phase_w[1] * phase_scale),
            "corner": float(phase_w[2] * phase_scale),
            "exit": float(phase_w[3] * phase_scale),
            "energy": float(w_energy),
        }

    def _energy_reward(self, progress_ratio: float, lap_complete: bool) -> float:
        """
        Contextual energy strategy reward.

        Goals:
        - reward energy deployment only when it actually helps acceleration/progress,
        - reward regeneration only when braking or lift-off makes it meaningful,
        - discourage using the battery in the wrong phase,
        - discourage finishing the lap with large unused charge.
        """
        energy_used = max(self.soc_before - self.soc_after, 1e-6)
        progress_norm = self.ds / max(self.target_seg_len, 1e-6)
        speed_gain = max(0.0, self.velocity_after - self.velocity_before)

        # Efficiency: progress earned per unit energy spent.
        # Clamp to keep the scale bounded and stable.
        efficiency = self._clamp(progress_norm / max(energy_used, 1e-3), 0.0, 5.0)
        r_eff = 0.20 * efficiency

        deploy_context = (self.throttle_eff > 0.55) and (self.curvature < 0.02)
        regen_context = (self.brake_eff > 0.15) or (self.throttle_eff < 0.10)

        r_deploy = 0.0
        if self.battery_status == "DEPLOY" and deploy_context:
            r_deploy = 0.50 * self.deploy_level * speed_gain

        r_regen = 0.0
        if self.battery_status == "REGEN" and regen_context:
            r_regen = 0.30 * self.regen_intensity * self.brake_eff

        r_bad_deploy = 0.0
        if self.battery_status == "DEPLOY" and (self.curvature > 0.02 or self.brake_eff > 0.10):
            r_bad_deploy = -0.40 * self.deploy_level

        r_bad_regen = 0.0
        if self.battery_status == "REGEN" and self.throttle_eff > 0.40:
            r_bad_regen = -0.20 * self.regen_intensity

        r_unused = 0.0
        if lap_complete:
            r_unused = -0.60 * max(self.soc_after, 0.0)

        # Small progress-sensitive encouragement to use energy only when the lap is still open.
        r_strategy = 0.10 * progress_ratio * (r_deploy + r_regen)

        return float(r_eff + r_deploy + r_regen + r_bad_deploy + r_bad_regen + r_unused + r_strategy)

    def compute(self) -> Tuple[float, Dict[str, float]]:
        weights = self._phase_weights()

        progress_ratio = self.progress_m / max(self.total_length_m, 1e-6)
        lap_complete = progress_ratio >= 0.999
        step_limit = self.step_count >= self.max_episode_steps

        v_ratio = self.velocity_after / max(self.v_max_seg, 1e-3)
        over_speed = max(0.0, self.velocity_after - self.v_max_seg)
        ay_ratio = self.ay_demand / max(self.ay_limit, 1e-3)
        over_lateral = max(0.0, ay_ratio - 1.0)
        accel = self.velocity_after - self.velocity_before

        progress_reward = self.ds / max(self.target_seg_len, 1.0)
        time_penalty = -0.01
        terminal_bonus = 10.0 if lap_complete else 0.0
        timeout_penalty = -2.0 if step_limit and not lap_complete else 0.0

        r_straight = 0.6 * min(v_ratio, 1.0) - 0.4 * max(0.0, v_ratio - 1.0)
        r_entry = 0.8 * self.brake_eff * max(0.0, v_ratio - 0.9) - 0.3 * self.throttle_eff
        r_corner = -1.2 * over_lateral - 0.2 * abs(self.slip_angle_rad)
        r_exit = 0.8 * max(0.0, accel) * self.throttle_eff - 0.4 * max(0.0, v_ratio - 1.0)

        r_energy = self._energy_reward(progress_ratio=progress_ratio, lap_complete=lap_complete)

        overlap_penalty = -0.20 * min(self.throttle_eff, self.brake_eff)
        wear_penalty = -0.05 * self.tire_wear
        slip_penalty = -0.03 * abs(self.slip_ratio)
        speed_penalty = -0.02 * over_speed
        steer_smooth_penalty = -0.03 * abs(self.steering_cmd) * max(0.0, v_ratio - 0.8)

        phase_reward = (
            weights["straight"] * r_straight
            + weights["entry"] * r_entry
            + weights["corner"] * r_corner
            + weights["exit"] * r_exit
            + weights["energy"] * r_energy
        )

        total_reward = float(
            progress_reward
            + time_penalty
            + phase_reward
            + speed_penalty
            + overlap_penalty
            + wear_penalty
            + slip_penalty
            + steer_smooth_penalty
            + terminal_bonus
            + timeout_penalty
        )

        breakdown = {
            "progress_reward": float(progress_reward),
            "time_penalty": float(time_penalty),
            "phase_reward": float(phase_reward),
            "speed_penalty": float(speed_penalty),
            "overlap_penalty": float(overlap_penalty),
            "wear_penalty": float(wear_penalty),
            "slip_penalty": float(slip_penalty),
            "steer_smooth_penalty": float(steer_smooth_penalty),
            "terminal_bonus": float(terminal_bonus),
            "timeout_penalty": float(timeout_penalty),
            "w_straight": float(weights["straight"]),
            "w_entry": float(weights["entry"]),
            "w_corner": float(weights["corner"]),
            "w_exit": float(weights["exit"]),
            "w_energy": float(weights["energy"]),
            "r_straight": float(r_straight),
            "r_entry": float(r_entry),
            "r_corner": float(r_corner),
            "r_exit": float(r_exit),
            "r_energy": float(r_energy),
        }
        return total_reward, breakdown

    def _phase_weight(self) -> Dict[str, float]:
        """Backward-compatible wrapper for older callsites."""
        return self._phase_weights()

    def _rewards(self) -> Tuple[float, Dict[str, float]]:
        """Backward-compatible wrapper for older callsites."""
        return self.compute()

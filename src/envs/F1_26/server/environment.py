from uuid import uuid4
import os
import numpy as np

from openenv.core.env_server import Environment, Action

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from model import F1Observation, F1State
from Track import gps_to_segments, compute_distance
from physics import calucate_grip, lateral_acceleration_limit, tire_degradation, update_speed
from rewards_updated import RewardFunction





class F1Env(Environment):
    def __init__(
        self,
        target_seg_len: int = 50,
        kappa_straight: float = 0.007,
        ay_max: float = 30.0,
        hs_speed_mps: float = 50.0,
        curv_percentile: int = 90,
        dt: float = 0.0125,
        initial_soc: float = 1.0,
        initial_velocity: float = 0.0,
        initial_tire_temp: float = 80.0,
        base_mu: float = 1.6,
    ):
        super().__init__()

        track_path = "../track/Melbourne.csv"
        if not os.path.exists(track_path):
            raise FileNotFoundError(f"Track file not found at {track_path}")

        self.target_seg_len = target_seg_len
        self.kappa_straight = kappa_straight
        self.ay_max = ay_max
        self.hs_speed_mps = hs_speed_mps
        self.curv_percentile = curv_percentile

        self.x, self.y, self.segments = gps_to_segments(
            track_path,
            target_seg_len=self.target_seg_len,
            kappa_straight=self.kappa_straight,
            ay_max=self.ay_max,
            hs_speed_mps=self.hs_speed_mps,
            curv_percentile=self.curv_percentile,
        )

        self._cum_dist = compute_distance(self.x, self.y)
        self._total_length = self._cum_dist[-1]
        self._total_length_m = float(self._total_length)
        self._segment_bounds_m = []

        for seg in self.segments:
            start_idx = int(seg["start_idx"])
            end_idx = int(seg["end_idx"])
            start_m = float(self._cum_dist[start_idx])
            end_m = float(self._cum_dist[end_idx])
            self._segment_bounds_m.append((start_m, max(start_m, end_m)))

        self.segments_end_m = self._segment_bounds_m[-1][1] if self._segment_bounds_m else 0.0
        self._segment_end_m = np.array([b[1] for b in self._segment_bounds_m], dtype=float)

        self.dt = dt

        self.soc = initial_soc
        self.initial_soc = initial_soc
        self.battery_status = "NEUTRAL"

        self.velocity = initial_velocity
        self.initial_velocity = initial_velocity

        self.tire_temp = initial_tire_temp
        self.initial_tire_temp = initial_tire_temp

        self.tire_wear = 0.0

        self.mu = base_mu
        self.base_mu = base_mu

        self.aero_mode = int(self.segments[0].get("aero_mode", 0))
        self.throttle = 0.0
        self.brake = 0.0
        self.slip_ratio = 0.0
        self.slip_angle_rad = 0.0
        self.steering = 0.0

        self._segment_idx = 0
        self.progress_m = 0.0
        self._progress_m = 0.0
        self._step_count = 0
        self.max_episode_steps = 1000
        self._episode_id = str(uuid4())
        self._done = False
        self._last_reward_breakdown = {}

        self._state = self._build_state()

    def reset(self) -> F1Observation:
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._done = False

        self._segment_idx = 0
        self.progress_m = 0.0
        self._progress_m = 0.0

        self.velocity = self.initial_velocity
        self.soc = self.initial_soc
        self.battery_status = "NEUTRAL"
        self.tire_wear = 0.0
        self.mu = self.base_mu
        self.tire_temp = self.initial_tire_temp
        self.aero_mode = int(self.segments[0].get("aero_mode", 0))

        self._state = self._build_state()
        return self._get_observation(
            reward=0.0,
            done=False,
            metadata={"segment_type": self.segments[0].get("type", "straight")},
        )

    def step(self, action: Action) -> F1Observation:
        if self._done:
            return self._get_observation(
                reward=0.0,
                done=True,
                metadata={"reason": "episode_already_done"},
            )

        throttle_cmd = float(np.clip(action.throttle, 0.0, 1.0))
        brake_cmd = float(np.clip(action.brake, 0.0, 1.0))
        steering_cmd = float(np.clip(action.steering, -1.0, 1.0))
        regen_intensity = float(np.clip(action.regen_intensity, 0.0, 1.0))
        deploy_level = float(np.clip(action.deploy_level, 0.0, 1.0))
        battery_status = str(action.battery_status).upper()

        self.aero_mode = int(self.segments[self._segment_idx].get("aero_mode", 0))

        throttle_eff, brake_eff = self._energy_strategy(
            throttle=throttle_cmd,
            brake=brake_cmd,
            battery_status=battery_status,
            deploy_level=deploy_level,
            regen_intensity=regen_intensity,
        )

        velocity_before = self.velocity
        soc_before = self.soc

        velocity_after, soc_after = update_speed(
            velocity=velocity_before,
            throttle=throttle_eff,
            soc=soc_before,
            mode=self.aero_mode,
            brake=brake_eff,
            mu=self.mu,
            dt=self.dt,
        )

        velocity_after = float(max(0.0, velocity_after))
        soc_after = float(np.clip(soc_after, 0.0, 1.0))

        slip_ratio = float(np.clip(throttle_eff - brake_eff, -1.0, 1.0))
        slip_angle_rad = float(np.radians(5.0 * steering_cmd) * min(1.0, velocity_after / self.hs_speed_mps))

        grip_mu = calucate_grip(
            mu=self.mu,
            slip_ratio=slip_ratio,
            slip_angle_rad=slip_angle_rad,
            velocity=velocity_after,
            tire_wear=1.0 - self.tire_wear,
            mode=self.aero_mode,
            temp=self.tire_temp,
        )

        mu_after, wear_after = tire_degradation(
            mu=grip_mu,
            velocity=velocity_after,
            throttle=throttle_eff,
            brake=brake_eff,
            temp=self.tire_temp,
            wear=self.tire_wear,
            steering=steering_cmd,
        )

        self.mu = float(np.clip(mu_after, 0.4, self.base_mu))
        self.tire_wear = float(np.clip(wear_after, 0.0, 1.0))

        heat_in = (abs(slip_angle_rad) * 180.0 / np.pi) * 0.03 + brake_eff * 0.7 + throttle_eff * 0.02
        heat_out = 0.08 * max(0.0, self.tire_temp - 80.0)

        self.tire_temp = float(np.clip(self.tire_temp + (heat_in - heat_out) * self.dt * 4.0, 70.0, 120.0))

        ds = max(0.0, 0.5 * (velocity_before + velocity_after) * self.dt)
        self.progress_m = min(self._total_length_m, self.progress_m + ds)
        self._progress_m = self.progress_m
        self._segment_idx = self._find_segment_idx(self.progress_m)

        segment = self.segments[self._segment_idx]
        self.aero_mode = int(segment.get("aero_mode", 0))
        curvature = float(max(0.0, segment.get("curvature_pctl", 0.0)))
        vmax_seg = float(segment.get("vmax_mps", self.hs_speed_mps))

        ay_demand = float((velocity_after ** 2) * curvature)
        ay_limit = float(
            lateral_acceleration_limit(
                velocity=velocity_after,
                mode=self.aero_mode,
                slip_ratio=slip_ratio,
                slip_angle_rad=slip_angle_rad,
                temp=self.tire_temp,
                mu=self.mu,
            )
        )

        self._step_count += 1
        lap_complete = self._progress_m >= (self._total_length_m - 1e-6)
        step_limit = self._step_count >= self.max_episode_steps
        self._done = lap_complete or step_limit

        reward_fn = RewardFunction(
            ds=ds,
            target_seg_len=self.target_seg_len,
            progress_m=self._progress_m,
            total_length_m=self._total_length_m,
            velocity_before=velocity_before,
            velocity_after=velocity_after,
            v_max_seg=vmax_seg,
            curvature=curvature,
            ay_demand=ay_demand,
            ay_limit=ay_limit,
            throttle_eff=throttle_eff,
            brake_eff=brake_eff,
            steering_cmd=steering_cmd,
            soc_before=soc_before,
            soc_after=soc_after,
            battery_status=battery_status,
            regen_intensity=regen_intensity,
            deploy_level=deploy_level,
            tire_wear=self.tire_wear,
            slip_ratio=slip_ratio,
            slip_angle_rad=slip_angle_rad,
            step_count=self._step_count,
            max_episode_steps=self.max_episode_steps,
        )
        reward, breakdown = reward_fn.compute()

        self.velocity = velocity_after
        self.soc = soc_after
        self.battery_status = battery_status
        self.throttle = throttle_eff
        self.brake = brake_eff
        self.slip_ratio = slip_ratio
        self.slip_angle_rad = slip_angle_rad
        self.steering = steering_cmd
        self._last_reward_breakdown = breakdown
        self._state = self._build_state()

        return self._get_observation(
            reward=reward,
            done=self._done,
            metadata={
                "segment_index": int(self._segment_idx),
                "segment_type": segment.get("type", "straight"),
                "vmax_segment_mps": float(vmax_seg),
                "lateral_accel_demand": float(ay_demand),
                "lateral_accel_limit": float(ay_limit),
                "lap_complete": bool(lap_complete),
                "reward_breakdown": breakdown,
            },
        )
    @property
    def state(self) -> F1State:
        return self._build_state()

    def _build_state(self) -> F1State:  
        progress_m = float(getattr(self, "_progress_m", self.progress_m))
        total_length_m = float(getattr(self, "_total_length_m", getattr(self, "_total_length", 0.0)))

        if not self.segments:
            return F1State(
                episode_id=str(self._episode_id or ""),
                step_count=int(self._step_count),
                speed=float(self.velocity),
                curvature_ahead=0.0,
                segment_progress=0.0,
                position_along_lap=(0.0, 0.0),
                battery_state_of_charge=float(np.clip(self.soc, 0.0, 1.0)),
                tire_wear=float(np.clip(self.tire_wear, 0.0, 1.0)),
                aero_mode=int(np.clip(self.aero_mode, 0, 1)),
                remaining_lap=float(max(0.0, total_length_m - progress_m)),
            )

        segment_idx = int(np.clip(self._segment_idx, 0, len(self.segments) - 1))
        segment = self.segments[segment_idx]

        return F1State(
            episode_id=str(self._episode_id or ""),
            step_count=int(self._step_count),
            speed=float(self.velocity),
            curvature_ahead=float(max(0.0, segment.get("curvature_pctl", 0.0))),
            segment_progress=self._segment_progress(segment_idx, progress_m),
            position_along_lap=self._position_from_progress(progress_m),
            battery_state_of_charge=float(np.clip(self.soc, 0.0, 1.0)),
            tire_wear=float(np.clip(self.tire_wear, 0.0, 1.0)),
            aero_mode=int(np.clip(self.aero_mode, 0, 1)),
            remaining_lap=float(max(0.0, total_length_m - progress_m)),
        )

    def _energy_strategy(
        self,
        throttle: float,
        brake: float,
        battery_status: str,
        deploy_level: float,
        regen_intensity: float,
    ) -> tuple[float, float]:
        if battery_status == "DEPLOY":
            throttle *= 0.90 + 0.50 * deploy_level
            brake = 0.0
        elif battery_status == "REGEN":
            throttle *= 1.0 - 0.60 * regen_intensity
            brake += 0.20 * regen_intensity * (1.0 - throttle)

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        return throttle, brake

    def _find_segment_idx(self, progress_m: float) -> int:
        idx = int(np.searchsorted(self._segment_end_m, progress_m, side="right"))

        if idx >= len(self.segments):
            idx = len(self.segments) - 1
        return idx

    def _segment_progress(self, segment_idx: int, progress_m: float) -> float:
        start_m, end_m = self._segment_bounds_m[segment_idx]
        span = max(end_m - start_m, 1e-6)
        return float(np.clip((progress_m - start_m) / span, 0.0, 1.0))

    def _position_from_progress(self, progress_m: float) -> tuple[float, float]:
        if progress_m <= 0.0:
            return float(self.x[0]), float(self.y[0])
        if progress_m >= self._total_length_m:
            return float(self.x[-1]), float(self.y[-1])

        idx = int(np.searchsorted(self._cum_dist, progress_m, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self._cum_dist) - 2))
        next_idx = idx + 1

        d_0 = float(self._cum_dist[idx])
        d_1 = float(self._cum_dist[next_idx])
        alpha = (progress_m - d_0) / max(d_1 - d_0, 1e-6)

        x = float((1.0 - alpha) * self.x[idx] + alpha * self.x[next_idx])
        y = float((1.0 - alpha) * self.y[idx] + alpha * self.y[next_idx])

        return x, y

    def _get_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
        metadata: dict | None = None,
    ) -> F1Observation:
        segment = self.segments[self._segment_idx]
        return F1Observation(
            speed=float(self.velocity),
            curvature_ahead=float(max(0.0, segment.get("curvature_pctl", 0.0))),
            battery_state_of_charge=float(self.soc),
            segment_progress=self._segment_progress(self._segment_idx, self._progress_m),
            tire_wear=float(self.tire_wear),
            position_along_lap=self._position_from_progress(self._progress_m),
            aero_mode=int(self.aero_mode),
            reward=float(reward),
            done=bool(done),
            metadata=metadata or {},
        )
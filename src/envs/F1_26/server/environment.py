from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from openenv.core.env_server import Environment

THIS_DIR = Path(__file__).resolve().parent
ENV_ROOT = THIS_DIR.parent
if str(ENV_ROOT) not in sys.path:
	sys.path.append(str(ENV_ROOT))

try:
	from .Track import compute_distance, gps_to_segments
	from .physics import (
		calucate_grip,
		lateral_acceleration_limit,
		tire_degradation,
		update_speed,
	)
except ImportError:
	from Track import compute_distance, gps_to_segments
	from physics import (
		calucate_grip,
		lateral_acceleration_limit,
		tire_degradation,
		update_speed,
	)

try:
	from ..model import F1Action, F1Observation, F1State
except ImportError:
	from model import F1Action, F1Observation, F1State


class F1Environment(Environment[F1Action, F1Observation, F1State]):
	"""OpenEnv-compatible F1 strategy environment."""

	SUPPORTS_CONCURRENT_SESSIONS = True

	def __init__(
		self,
		track_csv_path: Optional[str] = None,
		target_seg_len: float = 50.0,
		kappa_straight: float = 0.007,
		ay_max: float = 30.0,
		hs_speed_mps: float = 50.0,
		curv_percentile: float = 90.0,
		dt: float = 0.0125,
		max_steps: int = 20_000,
		initial_soc: float = 0.75,
		initial_speed_mps: float = 0.0,
		initial_tire_temp_c: float = 95.0,
		base_mu: float = 1.4,
	):
		super().__init__()

		track_path = track_csv_path or os.getenv("F1_TRACK_CSV")
		if not track_path:
			track_path = str((ENV_ROOT / "track" / "Melbourne.csv").resolve())

		path_obj = Path(track_path)
		if not path_obj.exists():
			raise FileNotFoundError(f"Track CSV not found: {path_obj}")

		self.track_csv_path = str(path_obj)
		self.target_seg_len = float(target_seg_len)
		self.dt = float(dt)
		self.max_steps = int(max_steps)
		self.initial_soc = float(np.clip(initial_soc, 0.05, 1.0))
		self.initial_speed_mps = float(max(0.0, initial_speed_mps))
		self.initial_tire_temp_c = float(initial_tire_temp_c)
		self.base_mu = float(np.clip(base_mu, 0.4, 1.6))

		self.x, self.y, self.segments = gps_to_segments(
			self.track_csv_path,
			target_seg_len=self.target_seg_len,
			kappa_straight=kappa_straight,
			ay_max=ay_max,
			hs_speed_mps=hs_speed_mps,
			curv_percentile=curv_percentile,
		)

		if len(self.x) < 2 or len(self.y) < 2:
			raise ValueError("Track must include at least two points.")
		if not self.segments:
			raise ValueError("Track segmentation produced no segments.")

		self._cum_dist = compute_distance(self.x, self.y)
		self._total_length_m = float(self._cum_dist[-1])
		if self._total_length_m <= 0.0:
			raise ValueError("Track length must be positive.")

		self._segment_bounds_m = []
		for seg in self.segments:
			start_idx = int(seg["start_idx"])
			end_idx = int(seg["end_idx"])
			start_m = float(self._cum_dist[start_idx])
			end_m = float(self._cum_dist[end_idx])
			self._segment_bounds_m.append((start_m, max(start_m, end_m)))
		self._segment_end_m = np.array([b[1] for b in self._segment_bounds_m], dtype=float)

		self._rng = np.random.default_rng()
		self._episode_id = str(uuid4())
		self._step_count = 0
		self._done = False

		self._progress_m = 0.0
		self._speed_mps = self.initial_speed_mps
		self._soc = self.initial_soc
		self._tire_wear = 0.0
		self._mu = self.base_mu
		self._temp_c = self.initial_tire_temp_c
		self._segment_idx = 0
		self._aero_mode = int(self.segments[0].get("aero_mode", 0))

		self._state = self._build_state()

	def reset(
		self,
		seed: Optional[int] = None,
		episode_id: Optional[str] = None,
		**kwargs: Any,
	) -> F1Observation:
		if seed is not None:
			self._rng = np.random.default_rng(seed)

		self._reset_rubric()

		self._episode_id = episode_id or str(uuid4())
		self._step_count = 0
		self._done = False

		self._progress_m = 0.0
		self._speed_mps = self.initial_speed_mps
		self._soc = self.initial_soc
		self._tire_wear = 0.0
		self._mu = self.base_mu
		self._temp_c = self.initial_tire_temp_c
		self._segment_idx = 0
		self._aero_mode = int(self.segments[0].get("aero_mode", 0))

		self._state = self._build_state()
		obs = self._build_observation(
			reward=0.0,
			done=False,
			metadata={"segment_type": self.segments[0].get("type", "straight")},
		)
		return self._apply_transform(obs)

	def step(
		self,
		action: F1Action,
		timeout_s: Optional[float] = None,
		**kwargs: Any,
	) -> F1Observation:
		del timeout_s
		del kwargs

		if isinstance(action, dict):
			action = F1Action.model_validate(action)

		if self._done:
			obs = self._build_observation(
				reward=0.0,
				done=True,
				metadata={"reason": "episode_already_done"},
			)
			return self._apply_transform(obs)

		throttle_cmd = float(np.clip(action.throttle, 0.0, 1.0))
		brake_cmd = float(np.clip(action.brake, 0.0, 1.0))
		steering_cmd = float(np.clip(action.steering, -1.0, 1.0))
		regen_intensity = float(np.clip(action.regen_intensity, 0.0, 1.0))
		deploy_level = float(np.clip(action.energy_deploy_level, 0.0, 1.0))
		battery_status = str(action.battery_status).upper()

		throttle_eff, brake_eff = self._apply_energy_strategy(
			throttle=throttle_cmd,
			brake=brake_cmd,
			regen_intensity=regen_intensity,
			deploy_level=deploy_level,
			battery_status=battery_status,
		)

		self._aero_mode = int(action.aero_mode)
		soc_before = self._soc
		speed_before = self._speed_mps

		speed_after, soc_after = update_speed(
			velocity=speed_before,
			throttle=throttle_eff,
			brake=brake_eff,
			mu=self._mu,
			soc=self._soc,
			mode=self._aero_mode,
			dt=self.dt,
		)
		speed_after = float(max(0.0, speed_after))
		soc_after = float(np.clip(soc_after, 0.0, 1.0))

		slip_ratio = float(np.clip(throttle_eff - brake_eff, -1.0, 1.0))
		slip_angle_rad = float(np.radians(5.0 * steering_cmd) * min(1.0, speed_after / 60.0))

		grip_mu = calucate_grip(
			mu=self._mu,
			mode=self._aero_mode,
			velocity=speed_after,
			temp=self._temp_c,
			slip_ratio=slip_ratio,
			slip_angle_rad=slip_angle_rad,
			tire_wear=1.0 - self._tire_wear,
		)

		mu_after, wear_after = tire_degradation(
			mu=grip_mu,
			wear=self._tire_wear,
			velocity=speed_after,
			temp=self._temp_c,
			throttle=throttle_eff,
			steering=steering_cmd,
			brake=brake_eff,
		)

		self._mu = float(np.clip(mu_after, 0.4, 1.6))
		self._tire_wear = float(np.clip(wear_after, 0.0, 0.98))

		heat_in = (abs(slip_angle_rad) * 180.0 / np.pi) * 0.03 + brake_eff * 0.8 + throttle_eff * 0.2
		heat_out = 0.08 * max(0.0, self._temp_c - 90.0)
		self._temp_c = float(np.clip(self._temp_c + (heat_in - heat_out) * self.dt * 4.0, 70.0, 130.0))

		ds = max(0.0, 0.5 * (speed_before + speed_after) * self.dt)
		self._progress_m = min(self._total_length_m, self._progress_m + ds)
		self._segment_idx = self._find_segment_idx(self._progress_m)

		segment = self.segments[self._segment_idx]
		curvature = float(max(0.0, segment.get("curvature_pctl", 0.0)))
		vmax_seg = float(max(1e-6, segment.get("vmax_mps", 1e6)))

		ay_demand = float((speed_after**2) * curvature)
		ay_limit = float(
			lateral_acceleration_limit(
				velocity=speed_after,
				slip_ratio=slip_ratio,
				slip_angle_rad=slip_angle_rad,
				temp=self._temp_c,
				mu=self._mu,
			)
		)

		over_speed = max(0.0, speed_after - vmax_seg)
		over_lateral = max(0.0, ay_demand - ay_limit)

		progress_reward = ds / max(self.target_seg_len, 1.0)
		time_penalty = -0.01
		speed_penalty = -0.02 * over_speed
		lateral_penalty = -0.03 * over_lateral
		soc_reward = 0.15 * (soc_after - soc_before)
		wear_penalty = -0.05 * self._tire_wear
		aero_reward = 0.02 if self._aero_mode == int(segment.get("aero_mode", 0)) else -0.02

		reward = (
			progress_reward
			+ time_penalty
			+ speed_penalty
			+ lateral_penalty
			+ soc_reward
			+ wear_penalty
			+ aero_reward
		)

		self._step_count += 1
		lap_complete = self._progress_m >= (self._total_length_m - 1e-6)
		step_limit = self._step_count >= self.max_steps
		self._done = lap_complete or step_limit

		if lap_complete:
			reward += 10.0
		elif step_limit:
			reward -= 2.0

		self._speed_mps = speed_after
		self._soc = soc_after
		self._state = self._build_state()

		obs = self._build_observation(
			reward=float(reward),
			done=self._done,
			metadata={
				"segment_index": self._segment_idx,
				"segment_type": segment.get("type", "straight"),
				"vmax_segment_mps": vmax_seg,
				"lateral_accel_demand": ay_demand,
				"lateral_accel_limit": ay_limit,
				"lap_complete": lap_complete,
			},
		)

		obs.reward = float(obs.reward or 0.0) + float(self._apply_rubric(action, obs))
		return self._apply_transform(obs)

	@property
	def state(self) -> F1State:
		return self._state

	def _apply_energy_strategy(
		self,
		throttle: float,
		brake: float,
		regen_intensity: float,
		deploy_level: float,
		battery_status: str,
	) -> tuple[float, float]:
		if battery_status == "DEPLOY":
			throttle *= 0.75 + 0.50 * deploy_level
			brake *= 1.0 - 0.50 * regen_intensity
		elif battery_status == "REGEN":
			throttle *= 1.0 - 0.90 * regen_intensity
			brake += 0.40 * regen_intensity * (1.0 - throttle)

		return float(np.clip(throttle, 0.0, 1.0)), float(np.clip(brake, 0.0, 1.0))

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

		d0 = float(self._cum_dist[idx])
		d1 = float(self._cum_dist[next_idx])
		alpha = (progress_m - d0) / max(d1 - d0, 1e-6)

		x_pos = float((1.0 - alpha) * self.x[idx] + alpha * self.x[next_idx])
		y_pos = float((1.0 - alpha) * self.y[idx] + alpha * self.y[next_idx])
		return x_pos, y_pos

	def _build_state(self) -> F1State:
		segment = self.segments[self._segment_idx]
		return F1State(
			episode_id=self._episode_id,
			step_count=self._step_count,
			speed=float(self._speed_mps),
			curvature_ahead=float(max(0.0, segment.get("curvature_pctl", 0.0))),
			segment_progress=self._segment_progress(self._segment_idx, self._progress_m),
			position_along_lap=self._position_from_progress(self._progress_m),
			battery_state_of_charge=float(self._soc),
			tire_wear=float(self._tire_wear),
			aero_mode=int(self._aero_mode),
			remaining_lap=float(max(0.0, self._total_length_m - self._progress_m)),
		)

	def _build_observation(
		self,
		reward: float,
		done: bool,
		metadata: Optional[dict[str, Any]] = None,
	) -> F1Observation:
		segment = self.segments[self._segment_idx]
		return F1Observation(
			speed=float(self._speed_mps),
			curvature_ahead=float(max(0.0, segment.get("curvature_pctl", 0.0))),
			battery_state_of_charge=float(self._soc),
			segment_progress=self._segment_progress(self._segment_idx, self._progress_m),
			tire_wear=float(self._tire_wear),
			position_along_lap=self._position_from_progress(self._progress_m),
			aero_mode=int(self._aero_mode),
			reward=float(reward),
			done=bool(done),
			metadata=metadata or {},
		)


F1Env = F1Environment

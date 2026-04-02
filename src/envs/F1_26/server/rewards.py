'''
Plannnig to implement a hybrid reward function that combines  general thing Lap Completeino, Time and 
intricate stuff like Straights (focusing on max speed), corner entry (focusing on brake efficiency), 
corner exit (acceleration), cornering (focuses on speed stays under the lateral speed limit) 
and finally energy-based (I want it to conserve energy and deploy energy at the right place without loosing lap time,
and also want agent to utilize it rather than being greedy for not regenerating or saving it all the way down)





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





'''

import numpy as np

class RewardFunction:
    def __init__(
        self, 
        ds, 
        target_seg_len,
        progress_m,
        total_length_m, 
        velocity_before, 
        velocity_after, 
        v_max_seg, 
        curvature, 
        ay_demand, 
        ay_limit, 
        throttle_eff,
        brake_eff,
        steering_cmd,
        soc_before,
        soc_after,
        battery_status,
        regen_intensity,
        deploy_level,
        tire_wear,
        slip_ratio,
        slip_angle_rad,
        ):

        self.ds = ds
        self.progress_m = progress_m
        self.target_seg_len = target_seg_len
        self.total_length_m = total_length_m
        self.velocity_before = velocity_before
        self.velocity_after = velocity_after
        self.v_max_seg = v_max_seg
        self.curvature = curvature
        self.ay_demand = ay_demand
        self.ay_limit = ay_limit
        self.throttle_eff = throttle_eff
        self.brake_eff = brake_eff
        self.steering_cmd = steering_cmd
        self.soc_before = soc_before
        self.soc_after = soc_after
        self.battery_status = battery_status
        self.regen_intensity = regen_intensity
        self.deploy_level = deploy_level
        self.tire_wear = tire_wear
        self.slip_ratio = slip_ratio
        self.slip_angle_rad = slip_angle_rad
        self.step_count = 0
        self.max_episode_steps = 1000

    def _sigmoid(self, z, k):
        return float(1.0/(1.0 + np.exp(-k * z)))
    
    def _phase_weight(self):


        v_ratio = self.velocity_after / max(self.v_max_seg, 1e-3)
        ay_ratio = self.ay_demand / max(self.ay_limit, 1e-3)
        acceleration = float(self.velocity_after - self.velocity_before)
        
        
        
        g_curve = self._sigmoid(self.curvature - 0.01, 10)
        g_brake = self._sigmoid(self.brake_eff - 0.15, 15)
        g_throttle = self._sigmoid(self.throttle_eff - 0.30, 10)
        g_accerelation = self._sigmoid(acceleration - 0.05, 15)
        g_limit = self._sigmoid(ay_ratio - 0.85, 10.0)
        g_overspeed = self._sigmoid(v_ratio - 0.95, 10.0)

        raw_straight = (1.0 - g_curve) * (1.0 - g_brake)
        raw_entry = g_curve * g_brake * (0.6 + 0.4 * g_overspeed)
        raw_corner = g_curve * g_limit * (1.0 - 0.3 * g_throttle)
        raw_exit = g_curve * g_throttle * g_accerelation * (1.0 - g_brake)

        raw = np.array([raw_straight, raw_entry, raw_corner, raw_exit])

        tau = 0.5
        logits = np.exp(raw / max(tau, 1e-6))
        w = logits / np.sum(logits)

        deploy = (1.0 - g_curve) * g_throttle * (1.0 - g_brake)
        regen = np.clip(g_brake + (1.0 - g_throttle) * (1.0 - g_brake), 0.0, 1.0)
        w_energy = float(np.clip(0.5 * (deploy + regen), 0.2, 0.1))

        return {
            "straight": w[0] * (1.0 - w_energy),
            "entry": w[1] * (1.0 - w_energy),
            "corner": w[2] * (1.0 - w_energy),
            "exit": w[3] * (1.0 - w_energy),
            "energy": w_energy
        }

    def _rewards(self):

    ### Multi-phase reward function

        total_reward = 0.0
        rewards = {}

        progress_ratio = self.progress_m / self.total_length_m
        lap_complete = progress_ratio >= 0.99
        step_limit = self.step_count >= self.max_episode_steps
        rewards["progress_reward"] = self.ds / max(self.target_seg_len, 1.0)
        # rewards["time_penalty"] = -0.01 Need to figure out how to implement this

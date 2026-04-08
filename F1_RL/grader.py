import numpy as np


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


class AgentGrader:

    def compeletion_based_grader(self, trajectory: list) -> float:
        """
        Focus: Did the agent complete the lap safely?
        """

        total_reward = sum(step["reward"] for step in trajectory)

        final_step = trajectory[-1]
        final_obs = final_step["obs"]
        metadata = final_step.get("metadata", {})

        lap_complete = metadata.get("lap_complete", False)

        final_soc = final_obs.battery_state_of_charge
        tire_wear = final_obs.tire_wear

        # Normalize reward
        norm_reward = clamp(total_reward / 100.0)

        # Completion
        completion_score = 1.0 if lap_complete else 0.0

        # SOC: prefer not empty
        soc_score = clamp(final_soc)

        # Tire wear
        tire_score = clamp(1.0 - tire_wear)

        # Safety
        penalties = 0
        for step in trajectory:
            rb = step["metadata"].get("reward_breakdown", {})
            penalties += (
                abs(rb.get("overlap_penalty", 0)) +
                abs(rb.get("slip_penalty", 0)) +
                abs(rb.get("speed_penalty", 0))
            )

        safety_score = clamp(1.0 - penalties / len(trajectory))

        score = (
            0.40 * norm_reward +
            0.25 * completion_score +
            0.15 * soc_score +
            0.10 * tire_score +
            0.10 * safety_score
        )

        return clamp(score)

    def completion_based_grader(self, trajectory: list) -> float:
        """Compatibility wrapper with corrected method name."""
        return self.compeletion_based_grader(trajectory)


    def energy_efficiency_grader(self, trajectory: list) -> float:
        """
        Focus: Is energy used in correct driving phases?
        """
        total_progress = 0.0
        energy_used = 0.0
        phase_alignment_score = 0.0
        stability_penalty = 0.0

        for step in trajectory:
            obs = step["obs"]
            md = step.get("metadata", {})
            rb = md.get("reward_breakdown", {})

            # Progress
            total_progress += rb.get("progress_reward", 0)

            # Energy usage proxy
            r_energy = rb.get("r_energy", 0)
            energy_used += abs(r_energy)

            # Phase alignment based on reward components
            phase_alignment_score += (
                rb.get("r_straight", 0) +
                rb.get("r_entry", 0) +
                rb.get("r_corner", 0) +
                rb.get("r_exit", 0)
            )

            # Stability penalties
            stability_penalty += (
                abs(rb.get("slip_penalty", 0)) +
                abs(rb.get("overlap_penalty", 0)) +
                abs(rb.get("steer_smooth_penalty", 0))
            )

        n = len(trajectory)

        progress_score = clamp(total_progress / n)
        energy_efficiency = clamp(progress_score / (energy_used + 1e-6))
        phase_score = clamp(phase_alignment_score / n)
        stability_score = clamp(1.0 - stability_penalty / n)

        final_step = trajectory[-1]
        lap_complete = final_step["metadata"].get("lap_complete", False)
        outcome_score = 1.0 if lap_complete else 0.0

        score = (
            0.25 * progress_score +
            0.20 * phase_score +
            0.15 * energy_efficiency +
            0.15 * stability_score +
            0.10 * outcome_score +
            0.15 * clamp(total_progress / 100.0)
        )

        return clamp(score)

    def consistency_grader(self, trajectory: list) -> float:
        """
        Focus: Is behavior physically AND strategically optimal?
        """
        energy_timing_score = 0.0
        physics_violation_penalty = 0.0
        tire_management_score = 0.0
        efficiency_score = 0.0

        prev_soc = None

        for step in trajectory:
            obs = step["obs"]
            md = step.get("metadata", {})
            rb = md.get("reward_breakdown", {})

            curvature = obs.curvature_ahead
            soc = obs.battery_state_of_charge
            speed = obs.speed

            # --- Energy timing logic
            # Prefer deploy on low curvature (straights)
            if rb.get("r_energy", 0) > 0:
                if curvature < 0.02:
                    energy_timing_score += 1.0
                else:
                    energy_timing_score -= 1.0

            # --- Physics violations
            ay_demand = md.get("lateral_accel_demand", 0)
            ay_limit = md.get("lateral_accel_limit", 1e-6)

            if ay_demand > ay_limit:
                physics_violation_penalty += (ay_demand - ay_limit)

            # --- Tire management
            tire_management_score += (1.0 - obs.tire_wear)

            # --- Efficiency (progress vs energy)
            if prev_soc is not None:
                energy_used = max(prev_soc - soc, 0)
                efficiency_score += rb.get("progress_reward", 0) / (energy_used + 1e-6)

            prev_soc = soc

        n = len(trajectory)

        energy_timing_score = clamp((energy_timing_score / n + 1) / 2)
        physics_score = clamp(1.0 - physics_violation_penalty / n)
        tire_score = clamp(tire_management_score / n)
        efficiency_score = clamp(efficiency_score / n)

        final_step = trajectory[-1]
        lap_complete = final_step["metadata"].get("lap_complete", False)
        completion_score = 1.0 if lap_complete else 0.0

        # --- Robustness (simple proxy: stability of rewards)
        rewards = [step["reward"] for step in trajectory]
        reward_std = np.std(rewards)
        robustness_score = clamp(1.0 - reward_std)

        score = (
            0.20 * completion_score +
            0.20 * energy_timing_score +
            0.15 * physics_score +
            0.15 * tire_score +
            0.15 * efficiency_score +
            0.15 * robustness_score
        )

        return clamp(score)


# # =========================================================
# # OPTIONAL WRAPPER
# # =========================================================
# class AgentGraderSuite:
#     """
#     Run all graders together.
#     """

#     def __init__(self):
#         self.low = LowDifficultyGrader()
#         self.medium = MediumDifficultyGrader()
#         self.high = HighDifficultyGrader()

#     def evaluate(self, trajectory: list) -> dict:
#         return {
#             "low": self.low.grade(trajectory),
#             "medium": self.medium.grade(trajectory),
#             "high": self.high.grade(trajectory),
#         }
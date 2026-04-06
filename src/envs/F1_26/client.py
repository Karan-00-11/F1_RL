from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .model import F1Actions, F1Observation, F1State
# If your package uses models.py, change .model to .models

class F1EnvClient(EnvClient[F1Actions, F1Observation, F1State]):
    def _step_payload(self, action: F1Actions) -> Dict[str, Any]:
        return {
            "throttle": float(action.throttle),
            "brake": float(action.brake),
            "steering": float(action.steering),
            "regen_intensity": float(action.regen_intensity),
            "deploy_level": float(action.deploy_level),
            "battery_status": str(action.battery_status).upper(),
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[F1Observation]:
        obs_data = payload.get("observation", {})
        observation = F1Observation.model_validate({
            **obs_data,
            "reward": payload.get("reward"),
            "done": payload.get("done", False),
        })
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> F1State:
        return F1State.model_validate(payload)
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """F1 Rl Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from .models import F1Actions, F1Observation, F1State


# class F1RlEnv(
#     EnvClient[F1Actions, F1Observation, F1State]
# ):
#     """
#     Client for the F1 Rl Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with F1RlEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(F1Actions(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = F1RlEnv.from_docker_image("F1_RL-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(F1Actions(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: F1Actions) -> Dict:
#         """
#         Convert F1Actions to JSON payload for step message.

#         Args:
#             action: F1Actions instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[F1Observation]:
#         """
#         Parse server response into StepResult[F1Observation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with F1Observation
#         """
#         obs_data = payload.get("observation", {})
#         observation = F1Observation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> F1State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return F1State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )


from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import F1Actions, F1Observation, F1State
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
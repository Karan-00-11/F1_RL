from typing import Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import AliasChoices, ConfigDict, Field

@dataclass
class F1Actions(Action):
    model_config = ConfigDict(populate_by_name=True)

    brake: float = Field(default=0.0, ge=0.0, le=1.0)
    steering: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        validation_alias=AliasChoices("steering", "Steering"),
    )
    throttle: float = Field(default=0.0, ge=0.0, le=1.0)
    regen_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    deploy_level: float = Field(default=0.0, ge=0.0, le=1.0)
    battery_status: Literal["REGEN", "NEUTRAL", "DEPLOY"] = "NEUTRAL"


@dataclass
class F1Observation(Observation):
    speed: float = Field(default=0.0, ge=0.0)
    curvature_ahead: float = Field(default=0.0, ge=0.0)
    battery_state_of_charge: float = Field(default=0.8, ge=0.0, le=1.0)
    segment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    tire_wear: float = Field(default=0.0, ge=0.0, le=1.0)
    position_along_lap: tuple[float, float] = (0.0, 0.0)
    aero_mode: Literal[0, 1] = 0

@dataclass
class F1State(State):
    speed: float = Field(default=0.0, ge=0.0)
    curvature_ahead: float = Field(default=0.0, ge=0.0)
    segment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    position_along_lap: tuple[float, float] = (0.0, 0.0)
    battery_state_of_charge: float = Field(default=0.8, ge=0.0, le=1.0)
    tire_wear: float = Field(default=0.0, ge=0.0, le=1.0)
    aero_mode: Literal[0, 1] = 0
    remaining_lap: float = Field(default=0.0, ge=0.0)




# @dataclass(slots=True)
# class F1Action:
#     """Action payload sent to the OpenEnv server."""

#     throttle: float = 0.0
#     brake: float = 0.0
#     steering: float = 0.0
#     regen_intensity: float = 0.0
#     deploy_level: float = 0.0
#     battery_status: str = "NEUTRAL"

#     def to_payload(self) -> JsonDict:
#         return {
#             "throttle": float(self.throttle),
#             "brake": float(self.brake),
#             "steering": float(self.steering),
#             "regen_intensity": float(self.regen_intensity),
#             "deploy_level": float(self.deploy_level),
#             "battery_status": str(self.battery_status).upper(),
#         }


# @dataclass(slots=True)
# class F1Observation:
#     """Observation returned by the environment after reset/step."""

#     speed: float
#     curvature_ahead: float
#     battery_state_of_charge: float
#     segment_progress: float
#     tire_wear: float
#     position_along_lap: Tuple[float, float]
#     aero_mode: int
#     reward: float = 0.0
#     done: bool = False
#     metadata: Dict[str, Any] = field(default_factory=dict)

#     @classmethod
#     def from_payload(cls, payload: Mapping[str, Any]) -> "F1Observation":
#         pos = payload.get("position_along_lap", (0.0, 0.0))
#         if isinstance(pos, list):
#             pos = tuple(pos)
#         if not isinstance(pos, tuple) or len(pos) != 2:
#             pos = (0.0, 0.0)

#         return cls(
#             speed=float(payload.get("speed", 0.0)),
#             curvature_ahead=float(payload.get("curvature_ahead", 0.0)),
#             battery_state_of_charge=float(payload.get("battery_state_of_charge", 0.0)),
#             segment_progress=float(payload.get("segment_progress", 0.0)),
#             tire_wear=float(payload.get("tire_wear", 0.0)),
#             position_along_lap=(float(pos[0]), float(pos[1])),
#             aero_mode=int(payload.get("aero_mode", 0)),
#             reward=float(payload.get("reward", 0.0)),
#             done=bool(payload.get("done", False)),
#             metadata=dict(payload.get("metadata", {}) or {}),
#         )


# @dataclass(slots=True)
# class F1State:
#     """Episode metadata returned by GET /state."""

#     episode_id: str
#     step_count: int
#     speed: float
#     curvature_ahead: float
#     segment_progress: float
#     position_along_lap: Tuple[float, float]
#     battery_state_of_charge: float
#     tire_wear: float
#     aero_mode: int
#     remaining_lap: float

#     @classmethod
#     def from_payload(cls, payload: Mapping[str, Any]) -> "F1State":
#         pos = payload.get("position_along_lap", (0.0, 0.0))
#         if isinstance(pos, list):
#             pos = tuple(pos)
#         if not isinstance(pos, tuple) or len(pos) != 2:
#             pos = (0.0, 0.0)

#         return cls(
#             episode_id=str(payload.get("episode_id", "")),
#             step_count=int(payload.get("step_count", 0)),
#             speed=float(payload.get("speed", 0.0)),
#             curvature_ahead=float(payload.get("curvature_ahead", 0.0)),
#             segment_progress=float(payload.get("segment_progress", 0.0)),
#             position_along_lap=(float(pos[0]), float(pos[1])),
#             battery_state_of_charge=float(payload.get("battery_state_of_charge", 0.0)),
#             tire_wear=float(payload.get("tire_wear", 0.0)),
#             aero_mode=int(payload.get("aero_mode", 0)),
#             remaining_lap=float(payload.get("remaining_lap", 0.0)),
#         )


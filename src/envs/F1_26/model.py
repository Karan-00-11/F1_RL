from typing import Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import AliasChoices, ConfigDict, Field


class F1Action(Action):
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

class F1Observation(Observation):
    speed: float = Field(default=0.0, ge=0.0)
    curvature_ahead: float = Field(default=0.0, ge=0.0)
    battery_state_of_charge: float = Field(default=0.8, ge=0.0, le=1.0)
    segment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    tire_wear: float = Field(default=0.0, ge=0.0, le=1.0)
    position_along_lap: tuple[float, float] = (0.0, 0.0)
    aero_mode: Literal[0, 1] = 0


class F1State(State):
    speed: float = Field(default=0.0, ge=0.0)
    curvature_ahead: float = Field(default=0.0, ge=0.0)
    segment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    position_along_lap: tuple[float, float] = (0.0, 0.0)
    battery_state_of_charge: float = Field(default=0.8, ge=0.0, le=1.0)
    tire_wear: float = Field(default=0.0, ge=0.0, le=1.0)
    aero_mode: Literal[0, 1] = 0
    remaining_lap: float = Field(default=0.0, ge=0.0)
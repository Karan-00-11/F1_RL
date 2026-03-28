from dataclasses import dataclass
from typing import Literal
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


@dataclass 
class F1Action(Action):
    brake: float = Field(ge=0.0, le=1.0)
    Steering: int = Field([-1, 0, 1])
    throttle: float = Field(ge=0.0, le=1.0)
    aero_mode: Literal[0, 1]
    regen_intensity : float = Field(ge=0.0, le=1.0)
    energy_deploy_level : float = Field(ge=0.0, le=1.0)
    battery_status : Literal["REGEN", "NEUTRAL", "DEPLOY"] = Field(["REGEN", "NEUTRAL", "DEPLOY"])

@dataclass
class F1Observation(Observation):
    speed : float
    curvature_ahead : float
    battery_state_of_charge : float
    segment_progress : float
    tire_wear : float
    position_along_lap : tuple
    aero_mode : Literal[0, 1]

@dataclass
class F1State(State):
    speed : float
    curvature_ahead : float
    segment_progress : float
    position_along_lap : tuple
    battery_state_of_charge : float = Field(ge=0.0, le=8.5)
    tire_wear : float = Field(ge=0.0, le=1.0)
    aero_mode : Literal[0, 1]
    remaining_lap : float
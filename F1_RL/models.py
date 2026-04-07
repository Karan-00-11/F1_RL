# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the F1 Rl Environment.

The F1_RL environment is a simple test environment that echoes back messages.
"""
from typing import Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import AliasChoices, ConfigDict, Field


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


class F1Observation(Observation):
    speed: float = Field(default=0.0, ge=0.0)
    curvature_ahead: float = Field(default=0.0, ge=0.0)
    battery_state_of_charge: float = Field(default=0.8, ge=0.0, le=1.0)
    segment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    tire_wear: float = Field(default=0.0, ge=0.0, le=1.0)
    position_along_lap: tuple[float, float] = (0.0, 0.0)
    aero_mode: Literal[0, 1] = 0
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)

class F1State(State):
    episode_id: str
    step_count: int
    speed: float = Field(default=0.0, ge=0.0)
    curvature_ahead: float = Field(default=0.0, ge=0.0)
    segment_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    position_along_lap: tuple[float, float] = (0.0, 0.0)
    battery_state_of_charge: float = Field(default=0.8, ge=0.0, le=1.0)
    tire_wear: float = Field(default=0.0, ge=0.0, le=1.0)
    aero_mode: Literal[0, 1] = 0
    remaining_lap: float = Field(default=0.0, ge=0.0)
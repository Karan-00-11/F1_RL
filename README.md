# F1 RL Track Simulation Project

## Overview
This repository contains an OpenEnv-compatible F1-style reinforcement learning environment built around real track geometry and simplified physics.

The core environment package is in `F1_RL/`, with the server in `F1_RL/server/` and action/observation models in `F1_RL/models.py`.

## Environment Description
The environment models a single-car lap strategy problem with:

- Segment-based track representation from Melbourne track data
- Longitudinal speed update with drag, braking, deploy, and regen
- Lateral grip limits based on curvature, slip, tire state, and aero effects
- Tire wear and tire temperature evolution
- Battery state-of-charge management through deploy/regen modes
- Reward shaping for progress, corner control, energy strategy, and completion

The environment server exposes OpenEnv endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

## Action Space (`F1Actions`)

- `throttle`: float in `[0.0, 1.0]`
- `brake`: float in `[0.0, 1.0]`
- `steering`: float in `[-1.0, 1.0]`
- `regen_intensity`: float in `[0.0, 1.0]`
- `deploy_level`: float in `[0.0, 1.0]`
- `battery_status`: one of `REGEN`, `NEUTRAL`, `DEPLOY`

## Observation Space (`F1Observation`)

- `speed`: m/s, `>= 0`
- `speed_kmh`: km/h, `>= 0`
- `curvature_ahead`: track curvature signal
- `battery_state_of_charge`: float in `[0.0, 1.0]`
- `segment_progress`: float in `[0.0, 1.0]`
- `tire_wear`: float in `[0.0, 1.0]`
- `position_along_lap`: `(x, y)` tuple
- `aero_mode`: `0` or `1`
- `reward`: scalar reward
- `done`: terminal flag
- `metadata`: includes segment type, lateral demand/limit, turn direction, and reward breakdown

## Setup Instructions

### 1. Install dependencies

From the environment package directory:

```bash
cd F1_RL
uv sync
```

If you are using pip instead of uv:

```bash
cd F1_RL
pip install -e .
```

### 2. Run the environment server locally

```bash
cd F1_RL
uv run --project . server --port 8000
```

Alternative:

```bash
cd F1_RL
python -m F1_RL.server.app --port 8000
```

### 3. Run the LLM control loop

Configure `.env` in `F1_RL/` with at least:

- `HF_TOKEN` (or `API_KEY`)
- `MODEL_NAME` (optional, default provided)
- `API_BASE_URL` (optional, default provided)
- `ENV_BASE_URL` (for existing server/Space) or `IMAGE_NAME`/`LOCAL_IMAGE_NAME`

Then run:

```bash
cd F1_RL
python inference.py
```

### 4. Validate before submission

```bash
cd F1_RL
openenv validate
```

Optional pre-validation script:

```bash
cd F1_RL
bash pre_validation.sh https://<your-space>.hf.space
```

### 5. Deploy to Hugging Face Spaces

```bash
cd F1_RL
openenv push --repo-id <username>/F1_RL
```

## Project Structure

```text
RL_Hackathon/
├── README.md
└── F1_RL/
	├── README.md
	├── openenv.yaml
	├── pyproject.toml
	├── models.py
	├── client.py
	├── inference.py
	├── pre_validation.sh
	└── server/
		├── app.py
		├── F1_RL_environment.py
		├── physics.py
		├── rewards_updated.py
		├── track.py
		└── Melbourne.csv
```

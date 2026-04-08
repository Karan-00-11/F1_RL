"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import F1Actions
from openai import OpenAI

from client import F1EnvClient
from grader import AgentGrader
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK", "echo")
BENCHMARK = os.getenv("BENCHMARK", "my_env_v4")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
AUTO_REUSE_LOCAL_ENV = os.getenv("AUTO_REUSE_LOCAL_ENV", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_STEPS = 100
TEMPERATURE = 0.2
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
You are a deterministic F1 control policy.

Goal:
Your job is to maximize total episode reward and minimize lap time by driving as fast as possible while staying within vehicle physics, track limits, and energy constraints.

At each step, you receive observation JSON and optional metadata JSON.
Return exactly one valid JSON object with these keys only:
"throttle", "brake", "steering", "regen_intensity", "deploy_level", "battery_status"

Objective:
1. Use braking efficiently on corner entry.
2. Maintain lateral grip limits in corners.
3. Accelerate aggressively but only when the car is aligned and stable.
4. Use energy strategically: deploy it where it produces meaningful speed gain, and regenerate it where braking or lift-off naturally allows it.
5. Avoid wasting energy or hoarding it unnecessarily.
6. Complete the lap in the shortest possible time.
7. Maintain high progress per step.

Constraints:
1. throttle: 0.0 to 1.0
2. brake: 0.0 to 1.0
3. steering: -1.0 to 1.0
4. regen_intensity: 0.0 to 1.0
5. deploy_level: 0.0 to 1.0
6. battery_status: REGEN or NEUTRAL or DEPLOY

Control rules:
1. Never output text, markdown, explanations, or extra keys.
2. Avoid throttle and brake overlap.
3. If battery_status is DEPLOY, set regen_intensity to 0.0.
4. If battery_status is REGEN, set deploy_level to 0.0.
5. Brake and steer in corners before lateral limit is exceeded.
6. Prefer DEPLOY on straights and acceleration opportunities.
7. Prefer REGEN during braking or lift-off.
8. In high curvature or near lateral limit, reduce throttle, use smooth steering, avoid aggressive deploy.
9.  If done is true, output neutral safe action.

Decision logic:

1. Terminal safety:
    -If done is true OR lap_complete is true, return:
    {"throttle":0.0,"brake":0.0,"steering":0.0,"regen_intensity":0.0,"deploy_level":0.0,"battery_status":"NEUTRAL"}

2. Corner mode (if corner_risk is true):

   - battery_status = "REGEN"
   - deploy_level = 0.0
   - brake = clamp(0.18 + 0.55 * max(0, v_ratio-0.85) + 0.70 * max(0, lat_ratio-0.90), 0.0, 1.0)
   - throttle = clamp(0.22 - 0.35*max(0, lat_ratio-0.90), 0.0, 0.40)
   - steering_mag = clamp(0.12 + 0.55*max(0, lat_ratio-0.75), 0.10, 0.85)
   - steering sign:
        - if metadata has turn_direction == "left": steering = -steering_mag
        - if metadata has turn_direction == "right": steering = +steering_mag
        - else keep same sign as most recent non-zero steering from history, if none use 0.0
   - regen_intensity = clamp(0.25 + 0.75*brake, 0.0, 1.0)
   - lat_ratio = lateral_accel_demand / max(lateral_accel_limit, 1e-3)
   - v_ratio = speed / max(vmax_segment_mps, 1e-3)
   - corner_risk = segment_type != straight OR curvature_ahead >= 0.008 OR lat_ratio >= 0.82
   - if corner_risk: brake > 0, abs(steering) > 0, battery_status = REGEN

3. Straight mode (otherwise):
    - brake = 0.0
    - steering = 0.0 (or small recentering toward 0 if previous steering was non-zero)
    - if v_ratio < 0.85: throttle = 1.0
    - else if v_ratio < 0.95: throttle = 0.85
    - else: throttle = 0.60
    - if battery_state_of_charge > 0.12 AND v_ratio < 0.98:
        - battery_status = "DEPLOY"
        - deploy_level = clamp(0.55 + 0.35*(1.0 - v_ratio), 0.55, 1.0)
        - regen_intensity = 0.0
    - else:
        - battery_status = "NEUTRAL"
        - deploy_level = 0.0
        - regen_intensity = 0.0

4. Hard post-rules:
    - If battery_status == "DEPLOY", force regen_intensity = 0.0
    - If battery_status == "REGEN", force deploy_level = 0.0
    - If throttle > 0 and brake > 0, keep only the larger one and set the smaller one to 0.0
    - Clamp all numeric values to bounds

Return format:
    - Exactly one line JSON object
    - All six keys must be present every step
    - Values must be numeric (except battery_status string)
"""
).strip()

### LOGGING AND PROMPT HELPERS ###
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _safe_action_payload() -> Dict[str, Any]:
    return {
        "throttle": 0.0,
        "brake": 0.0,
        "steering": 0.0,
        "regen_intensity": 0.0,
        "deploy_level": 0.0,
        "battery_status": "NEUTRAL",
    }


def _bounded_float(value: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return float(max(lo, min(hi, parsed)))


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("empty model output")

    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")

    payload = json.loads(raw_text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("parsed JSON is not an object")
    return payload


def _sanitize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = _safe_action_payload()

    action["throttle"] = _bounded_float(payload.get("throttle", 0.0), 0.0, 1.0)
    action["brake"] = _bounded_float(payload.get("brake", 0.0), 0.0, 1.0)
    action["steering"] = _bounded_float(payload.get("steering", 0.0), -1.0, 1.0)
    action["regen_intensity"] = _bounded_float(payload.get("regen_intensity", 0.0), 0.0, 1.0)
    action["deploy_level"] = _bounded_float(payload.get("deploy_level", 0.0), 0.0, 1.0)

    status = str(payload.get("battery_status", "NEUTRAL")).upper()
    if status not in {"REGEN", "NEUTRAL", "DEPLOY"}:
        status = "NEUTRAL"
    action["battery_status"] = status

    # Enforce mutually exclusive power modes.
    if status == "DEPLOY":
        action["regen_intensity"] = 0.0
    elif status == "REGEN":
        action["deploy_level"] = 0.0

    # Avoid simultaneous throttle and brake usage.
    if action["throttle"] > 0.0 and action["brake"] > 0.0:
        if action["throttle"] >= action["brake"]:
            action["brake"] = 0.0
        else:
            action["throttle"] = 0.0

    return action


def build_user_prompt(step: int, observation: Any, history: List[str]) -> str:
    metadata = getattr(observation, "metadata", {}) or {}
    obs_payload = {
        "speed": float(getattr(observation, "speed", 0.0)),
        "speed_kmh": float(getattr(observation, "speed_kmh", getattr(observation, "speed", 0.0) * 3.6)),
        "curvature_ahead": float(getattr(observation, "curvature_ahead", 0.0)),
        "battery_state_of_charge": float(getattr(observation, "battery_state_of_charge", 0.0)),
        "segment_progress": float(getattr(observation, "segment_progress", 0.0)),
        "tire_wear": float(getattr(observation, "tire_wear", 0.0)),
        "aero_mode": int(getattr(observation, "aero_mode", 0)),
        "done": bool(getattr(observation, "done", False)),
    }

    metadata_payload = {
        "segment_type": metadata.get("segment_type"),
        "vmax_segment_mps": metadata.get("vmax_segment_mps"),
        "lateral_accel_demand": metadata.get("lateral_accel_demand"),
        "lateral_accel_limit": metadata.get("lateral_accel_limit"),
        "lap_complete": metadata.get("lap_complete"),
        "turn_direction": metadata.get("turn_direction"),
        "steering_input": metadata.get("steering_input"),
        "lateral_offset_m": metadata.get("lateral_offset_m"),
    }

    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation JSON:
        {json.dumps(obs_payload, separators=(",", ":"))}
        Metadata JSON:
        {json.dumps(metadata_payload, separators=(",", ":"))}
        Recent history:
        {history_block}
        Output exactly one valid action JSON object.
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    observation: Any,
    history: List[str],
) -> tuple[F1Actions, str, Optional[str]]:
    user_prompt = build_user_prompt(step, observation, history)
    parse_error: Optional[str] = None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        action_payload = _sanitize_action_payload(_extract_json_object(raw_text))
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
        action_payload = _safe_action_payload()

    action = F1Actions(**action_payload)
    action_json = json.dumps(action_payload, separators=(",", ":"))
    return action, action_json, parse_error


def _discover_running_container_urls(image_name: Optional[str]) -> List[str]:
    if not image_name:
        return []

    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"ancestor={image_name}",
                "--format",
                "{{.Ports}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    urls: List[str] = []
    for line in result.stdout.splitlines():
        for match in re.findall(r"(?:0\.0\.0\.0|\[::\]):(\d+)->8000/tcp", line):
            url = f"http://localhost:{match}"
            if url not in urls:
                urls.append(url)
    return urls


async def create_env_client() -> F1EnvClient:
    if ENV_BASE_URL:
        env = F1EnvClient(base_url=ENV_BASE_URL)
        await env.connect()
        return env

    if AUTO_REUSE_LOCAL_ENV:
        candidate_urls = ["http://localhost:8000", *_discover_running_container_urls(IMAGE_NAME)]
        for base_url in candidate_urls:
            try:
                env = F1EnvClient(base_url=base_url, connect_timeout_s=0.8)
                await env.connect()
                print(f"[DEBUG] Reusing running env server at {base_url}", flush=True)
                return env
            except Exception:
                continue

    if not IMAGE_NAME:
        raise ValueError(
            "IMAGE_NAME/LOCAL_IMAGE_NAME is not set. Provide one in .env or set ENV_BASE_URL to a running server."
        )

    return await F1EnvClient.from_docker_image(IMAGE_NAME)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await create_env_client()
    grader = AgentGrader()

    history: List[str] = []
    rewards: List[float] = []
    trajectory: List[Dict[str, Any]] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, action_str, action_error = get_model_action(
                client,
                step,
                result.observation,
                history,
            )

            result = await env.step(action)
            obs = result.observation

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            metadata = getattr(obs, "metadata", {}) or {}

            rewards.append(reward)
            trajectory.append(
                {
                    "reward": reward,
                    "obs": obs,
                    "metadata": metadata,
                }
            )
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=action_error)

            history.append(
                f"step={step} action={action_str} reward={reward:.2f} "
                f"segment={metadata.get('segment_type', 'unknown')}"
            )

            if done:
                break

        if trajectory:
            completion_score = grader.compeletion_based_grader(trajectory)
            energy_score = grader.energy_efficiency_grader(trajectory)
            consistency_score = grader.consistency_grader(trajectory)
            score = (completion_score + energy_score + consistency_score) / 3.0
        else:
            score = 0.0

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
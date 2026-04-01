from __future__ import annotations

import sys
from pathlib import Path

from openenv.core.env_server import create_fastapi_app

THIS_DIR = Path(__file__).resolve().parent
ENV_ROOT = THIS_DIR.parent
if str(ENV_ROOT) not in sys.path:
	sys.path.append(str(ENV_ROOT))

try:
	from .environment import F1Environment
except ImportError:
	from environment import F1Environment

try:
	from ..model import F1Action, F1Observation
except ImportError:
	from model import F1Action, F1Observation


app = create_fastapi_app(
	env=F1Environment,
	action_cls=F1Action,
	observation_cls=F1Observation,
)

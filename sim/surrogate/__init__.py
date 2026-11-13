from __future__ import annotations

from pathlib import Path

from surrogate.base import DEFAULT_HORIZON_S, CandidateScore, Predictor
from surrogate.features import FEATURE_NAMES, FEATURE_SCHEMA
from surrogate.rollout import RolloutPredictor


def make_predictor(name: str, model_path: Path | None = None) -> Predictor:
    if name == "rollout":
        return RolloutPredictor()
    if name == "learned":
        if model_path is None:
            raise ValueError("the learned predictor needs --model-path")
        from surrogate.learned import LearnedPredictor

        return LearnedPredictor(model_path)
    raise ValueError(f"unknown predictor {name!r}")


__all__ = [
    "DEFAULT_HORIZON_S",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "CandidateScore",
    "Predictor",
    "RolloutPredictor",
    "make_predictor",
]

"""Team-level geometric blend with the competition's make_model(track) API.

Edit FACTORIES to point at each participant's existing make_model function.
Do not include this module's make_model in FACTORIES (that would recurse).
Weights are selected by tune_blend.py and then frozen in WEIGHTS_FILE.
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

# EDIT THESE IMPORT PATHS to match your team folders.
# Format: "python.module.path:function_name"
FACTORIES = (
    "ensemble_method.models.models:make_model",
    "lightgbm_method.models:make_model",
    "catboost_method.models:make_model"
)
TRACKS = ("off_market", "list_aware")
WEIGHTS_FILE = Path(__file__).resolve().parent / "outputs" / "team_blend" / "frozen_weights.json"


def load_factories():
    """Return factories in the declared order, with unique identifiers."""
    if len(FACTORIES) < 2 or len(set(FACTORIES)) != len(FACTORIES):
        raise ValueError("FACTORIES must contain at least two distinct model factories")
    functions = []
    for path in FACTORIES:
        try:
            module_name, function_name = path.rsplit(":", 1)
            factory = getattr(import_module(module_name), function_name)
        except (ValueError, ImportError, AttributeError) as exc:
            raise ImportError(f"Could not import factory {path!r}: {exc}") from exc
        if not callable(factory):
            raise TypeError(f"Factory {path!r} is not callable")
        functions.append(factory)
    return functions


def validate_weights(weights, count):
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {count} finite blend weights")
    if (values < 0).any() or values.sum() <= 0:
        raise ValueError("Blend weights must be nonnegative with positive sum")
    return values / values.sum()


class BlendedModel:
    """Fresh constituent models; common fit/train and price prediction API."""

    def __init__(self, models, weights, names=None):
        self.models = list(models)
        if not self.models:
            raise ValueError("No models supplied")
        self.names = list(names) if names is not None else [f"model_{i}" for i in range(len(self.models))]
        if len(self.names) != len(self.models) or len(set(self.names)) != len(self.names):
            raise ValueError("Names must be unique and match the number of models")
        self.weights = validate_weights(weights, len(self.models))
        self._fitted = False

    def fit(self, train):
        self._fitted = False
        for name, model in zip(self.names, self.models):
            print(f"  Fitting {name}", flush=True)
            model.fit(train)
        self._fitted = True
        return self

    def predict_members(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        columns = {}
        for name, model in zip(self.names, self.models):
            prediction = np.asarray(model.predict(homes), dtype=float)
            if prediction.shape != (len(homes),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise ValueError(f"Invalid price predictions from {name}")
            columns[name] = prediction
        return pd.DataFrame(columns, index=homes.index)

    def predict(self, homes):
        values = self.predict_members(homes).to_numpy(dtype=float)
        result = np.exp(np.log(values) @ self.weights)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Blend produced invalid prices")
        return result


def make_model(track):
    """Return a fresh model for 'off_market' or 'list_aware' with FROZEN weights."""
    if track not in TRACKS:
        raise ValueError(f"Unknown track {track!r}")
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"No frozen weights at {WEIGHTS_FILE}; run tune_blend.py select first")
    settings = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    if settings.get("factories") != list(FACTORIES):
        raise ValueError("FACTORIES differs from frozen weights. Re-select weights or restore the original list/order.")
    weights = validate_weights(settings["weights"][track], len(FACTORIES))
    models = [factory(track) for factory in load_factories()]
    return BlendedModel(models, weights, names=FACTORIES)

"""Singular residual models and the competition construction entry point."""
from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .components import Component, ComponentSpec


def _log_values(frame, column):
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Expected positive finite {column}")
    return np.log(values)


def _training_rows(train, cutoff, start=None):
    if cutoff is not None or start is not None:
        dates = pd.to_datetime(train["date_sold"], errors="raise")
        if dates.isna().any():
            raise ValueError("Expected valid date_sold")
        eligible = pd.Series(True, index=train.index)
        if cutoff is not None:
            eligible &= dates < pd.Timestamp(cutoff)
        if start is not None:
            eligible &= dates >= pd.Timestamp(start)
        train = train.loc[eligible]
    if train.empty:
        raise ValueError("No eligible training rows")
    return train


class OffMarketModel:
    """One model whose components learn successive log-price residuals."""
    track = "off_market"

    def __init__(self, specs, *, center_target=False, training_cutoff=None):
        self.specs = deepcopy(list(specs))
        if not self.specs or any(not isinstance(s, ComponentSpec) for s in self.specs):
            raise ValueError("specs must contain ComponentSpec instances")
        self.center_target = center_target
        self.training_cutoff = training_cutoff
        self.components = [Component(s, track=self.track) for s in self.specs]
        self._fitted = False

    def _target(self, train):
        return _log_values(train, "sold_price")

    def fit(self, train):
        self._fitted = False
        train = _training_rows(train, self.training_cutoff)
        target = self._target(train)
        self.target_offset_ = float(np.median(target)) if self.center_target else 0.0
        residual = target - self.target_offset_
        self.components = [Component(s, track=self.track) for s in self.specs]
        for index, component in enumerate(self.components):
            component.fit(train, residual)
            if index + 1 < len(self.components):
                residual = residual - component.predict(train)
        self._fitted = True
        return self

    def _predict_target(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        prediction = np.full(len(homes), self.target_offset_, dtype=float)
        for component in self.components:
            prediction += component.predict(homes)
        return prediction

    def predict_log(self, homes):
        return self._predict_target(homes)

    def predict(self, homes):
        with np.errstate(over="ignore", under="ignore"):
            result = np.exp(self.predict_log(homes))
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Expected positive finite price predictions")
        return result


class ListAwareModel(OffMarketModel):
    """One residual model of log sale price or log sale/list ratio."""
    track = "list_aware"

    def __init__(self, specs, *, target_mode="ratio", center_target=False, training_cutoff=None):
        if target_mode not in ("direct", "ratio"):
            raise ValueError("target_mode must be direct or ratio")
        self.target_mode = target_mode
        super().__init__(specs, center_target=center_target, training_cutoff=training_cutoff)

    def _target(self, train):
        listed = _log_values(train, "list_price")
        sold = _log_values(train, "sold_price")
        return sold - listed if self.target_mode == "ratio" else sold

    def predict_log(self, homes):
        prediction = self._predict_target(homes)
        listed = _log_values(homes, "list_price")
        return prediction + listed if self.target_mode == "ratio" else prediction


def _accepted_specs(track):
    """Read exact accepted configurations; never substitute generic parameters."""
    filename = "release_checkpoint.zip" if track == "off_market" else "listaware_checkpoint.zip"
    path = Path(__file__).resolve().parents[1] / "colab_checkpoints" / filename
    with ZipFile(path) as archive:
        roster = json.loads(archive.read("roster.json"))
        settings = json.loads(archive.read("manifest.json"))["settings"]
        weights = json.loads(archive.read("summary.json"))["effective_member_weights"]
    members = {}
    for member in roster["members"]:
        candidate = member["candidate"]
        addons = candidate["addons"]
        component = ComponentSpec(
            estimator=candidate.get("family", "xgboost"),
            params=member["parameters"],
            features=dict(text_mode=candidate["text_mode"],
                          splines="spacetime_splines" in addons,
                          geo_context="geo_context" in addons,
                          comparable=candidate.get("comparable", "base"),
                          max_extra_terms=settings["max_extra_terms"], min_df=settings["min_df"]),
            row_mode=candidate["row_mode"], row_fraction=candidate["row_fraction"],
            seed=candidate["seed"],
        )
        members[candidate["name"]] = dict(specs=[component], center_target=True)
    # Original list-aware finalize() fixes this boundary; off-market finalize
    # uses all supplied historical sales from train_start, without an upper cap.
    return dict(members=members, weights=weights,
                training_start=settings["train_start"],
                training_cutoff="2026-06-01" if track == "list_aware" else None)


def make_model(track, specs=None):
    """Default: accepted ensemble. A list of ComponentSpecs selects one model.

    A dict with specs and optional target_mode/center_target/training_cutoff
    configures a singular model. A dict with members and weights configures an
    ensemble; member values are singular model configurations. training_cutoff
    is exclusive and optional: otherwise fit uses the caller's eligible history.
    """
    from .ensemble import OffMarketEnsemble, ListAwareEnsemble
    if track not in ("off_market", "list_aware"):
        raise ValueError(f"Unknown competition track: {track}")
    model_class = OffMarketModel if track == "off_market" else ListAwareModel
    ensemble_class = OffMarketEnsemble if track == "off_market" else ListAwareEnsemble
    config = _accepted_specs(track) if specs is None else deepcopy(specs)
    if not isinstance(config, dict):
        return model_class(config)
    if "members" not in config:
        return model_class(**config)
    members = config.pop("members")
    return ensemble_class(models={name: model_class(**settings) for name, settings in members.items()}, **config)

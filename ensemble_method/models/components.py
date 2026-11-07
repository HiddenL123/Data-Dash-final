"""A target-agnostic estimator and its independently fitted features."""
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter

import numpy as np
import pandas as pd

from ensemble_method.features.pipeline import FeaturePipeline


@dataclass
class ComponentSpec:
    estimator: str | type = "xgboost"
    params: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    row_mode: str = "all"
    row_fraction: float = 1.0
    seed: int = 42


def _sample_rows(n_rows, spec):
    """Original bagging sequence, applied after full-history feature fitting."""
    if n_rows < 1:
        raise ValueError("Need at least one training row")
    if spec.row_mode == "all":
        return np.arange(n_rows)
    return np.random.default_rng(spec.seed + 1_000_003).choice(
        n_rows, max(1, int(round(n_rows * spec.row_fraction))),
        replace=spec.row_mode == "bootstrap")


class Component:
    def __init__(self, spec: ComponentSpec, *, track="off_market"):
        self.spec = deepcopy(spec)
        if spec.row_mode not in ("all", "subbag", "bootstrap"):
            raise ValueError("Unknown row_mode")
        if not 0 < spec.row_fraction <= 1 or (spec.row_mode == "all" and spec.row_fraction != 1):
            raise ValueError("Invalid row_fraction")
        if not isinstance(spec.seed, int) or spec.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        family = spec.estimator if isinstance(spec.estimator, str) else spec.estimator.__module__.split(".")[0]
        names = {"catboost": "CatBoostRegressor", "xgboost": "XGBRegressor", "lightgbm": "LGBMRegressor"}
        if family not in names:
            raise ValueError("estimator must be a native CatBoost, XGBoost or LightGBM class or family")
        self.family = family
        self.estimator_class = (getattr(import_module(family), names[family])
                                if isinstance(spec.estimator, str) else spec.estimator)
        self.track = track
        # Validate feature settings before training; no pipeline state is shared.
        FeaturePipeline(model=family, track=track, **self.spec.features)
        self._fitted = False

    def fit(self, train, target):
        self._fitted = False
        if isinstance(target, pd.Series) and not target.index.equals(train.index):
            raise ValueError("Target rows do not match training rows")
        target = np.asarray(target, dtype=float)
        if target.shape != (len(train),) or not len(train) or not np.isfinite(target).all():
            raise ValueError("Expected one finite target per training row")
        X = self.prepare_features(train)
        return self.fit_estimator(X, target)

    def prepare_features(self, train):
        """Build features in the parent process, where blocks can be reused."""
        self._fitted = False
        self.pipeline = FeaturePipeline(model=self.family, track=self.track, **self.spec.features)
        started = perf_counter()
        print(f"[{self.track}/{self.family}] Building features for {len(train):,} training rows...", flush=True)
        X = self.pipeline.fit_transform(train)
        print(f"[{self.track}/{self.family}] Features ready: {X.shape[1]:,} columns "
              f"in {perf_counter() - started:.1f}s", flush=True)
        if not X.index.equals(train.index):
            raise ValueError("Feature rows were reordered")
        return X

    def fit_estimator(self, X, target, *, threads=None):
        self._fitted = False
        target = np.asarray(target, dtype=float)
        if target.shape != (len(X),) or not len(X) or not np.isfinite(target).all():
            raise ValueError("Expected one finite target per training row")
        rows = _sample_rows(len(X), self.spec)
        self.training_rows_ = rows
        params = deepcopy(self.spec.params)
        if threads is not None:
            # Override aliases too, so roster settings cannot defeat the budget.
            aliases = {"catboost": ("thread_count",),
                       "xgboost": ("n_jobs", "nthread"),
                       "lightgbm": ("n_jobs", "num_threads", "num_thread", "nthread", "nthreads")}
            for key in aliases[self.family]:
                params.pop(key, None)
            params["thread_count" if self.family == "catboost" else "n_jobs"] = threads
        # The original factories override roster parameter seeds per candidate.
        params["random_seed" if self.family == "catboost" else "random_state"] = self.spec.seed
        if self.family == "xgboost":
            params.setdefault("enable_categorical", True)
            params.setdefault("tree_method", "hist")
        self.estimator = self.estimator_class(**params)
        kwargs = {"cat_features": self.pipeline.categorical_features} if self.family == "catboost" else {}
        started = perf_counter()
        trees = params.get("iterations", params.get("n_estimators", "default"))
        print(f"[{self.track}/{self.family}] Fitting estimator: {len(rows):,} rows, "
              f"{trees} trees...", flush=True)
        self.estimator.fit(X.iloc[rows], target[rows], **kwargs)
        print(f"[{self.track}/{self.family}] Estimator fitted in {perf_counter() - started:.1f}s", flush=True)
        self._fitted = True
        return self

    def predict(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if homes.empty:
            return np.empty(0)
        X = self.pipeline.transform(homes)
        if not X.index.equals(homes.index):
            raise ValueError("Feature rows were reordered")
        result = np.asarray(self.estimator.predict(X), dtype=float)
        if result.shape != (len(homes),) or not np.isfinite(result).all():
            raise ValueError("Expected one finite component prediction per home")
        return result

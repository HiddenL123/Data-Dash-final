"""Shared geometric blending of independently trained singular models."""
import numpy as np
import pandas as pd
from time import perf_counter


class WeightedEnsemble:
    def __init__(self, models, weights, *, training_cutoff=None, training_start=None):
        from .models import OffMarketModel, ListAwareModel
        self.models = dict(models)
        self.weights = dict(weights)
        self.training_cutoff = training_cutoff
        self.training_start = training_start
        if not self.models or set(self.models) != set(self.weights):
            raise ValueError("Each model must have exactly one weight")
        if len({id(model) for model in self.models.values()}) != len(self.models):
            raise ValueError("Members must be independent model instances")
        for model in self.models.values():
            expected = OffMarketModel if self.track == "off_market" else ListAwareModel
            if not isinstance(model, expected) or model.track != self.track:
                raise ValueError("Member track does not match ensemble")
        self._weights()
        self._fitted = all(getattr(model, "_fitted", False) for model in self.models.values())

    def _weights(self):
        weights = np.asarray([self.weights[name] for name in self.models], dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
            raise ValueError("Invalid ensemble weights")
        weights = weights / weights.max()
        return weights / weights.sum()

    def fit(self, train):
        from .models import _training_rows
        self._fitted = False
        train = _training_rows(train, self.training_cutoff, self.training_start)
        for index, (name, model) in enumerate(self.models.items(), start=1):
            started = perf_counter()
            print(f"[{self.track}] Training member {index}/{len(self.models)}: {name}", flush=True)
            model.fit(train)
            print(f"[{self.track}] Member {name} ready in {perf_counter() - started:.1f}s", flush=True)
        self._fitted = True
        return self

    def predict_members(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if {"sold_price", "date_sold"} & set(homes.columns):
            raise ValueError("Remove sale outcomes before inference")
        predictions = {}
        for name, model in self.models.items():
            started = perf_counter()
            #print(f"[{self.track}] Predicting {len(homes):,} homes with {name}...", flush=True)
            values = np.asarray(model.predict(homes), dtype=float)
            if values.shape != (len(homes),) or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{name}: expected one positive finite price per home")
            predictions[name] = values
            #print(f"[{self.track}] Predictions ready in {perf_counter() - started:.1f}s", flush=True)
        return pd.DataFrame(predictions, index=homes.index)

    def predict(self, homes):
        prices = self.predict_members(homes).to_numpy(dtype=float)
        result = np.exp(np.log(prices) @ self._weights())
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Invalid blended prices")
        return result


class OffMarketEnsemble(WeightedEnsemble):
    track = "off_market"


class ListAwareEnsemble(WeightedEnsemble):
    track = "list_aware"

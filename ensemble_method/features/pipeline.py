
from __future__ import annotations

import pandas as pd

from .base import BoostingFeatures
from .text import (
    BinaryTextFeatures,
    build_text_bank,
    select_terms,
    condition_features,
    training_text_features,
)
from .comparables import ComparableSalesFeatures, add_list_comparable_features
from .spatial import SpaceTimeSplines, GeographicContextFeatures
from .cache import fit_block, cached_value


class FeaturePipeline:
    def __init__(
        self,
        *,
        model: str = "xgboost",
        track: str = "off_market",
        text_mode: str = "selected",
        conditions: bool = False,  # NEW
        splines: bool = True,
        geo_context: bool = True,
        comparable: str = "base",
        max_extra_terms: int = 1000,
        min_df: int = 10,
    ):
        if model not in ("xgboost", "lightgbm", "catboost"):
            raise ValueError(f"Unknown model: {model}")
        if track not in ("off_market", "list_aware"):
            raise ValueError(f"Unknown track: {track}")
        if text_mode not in ("selected", "full"):
            raise ValueError(f"Unknown text mode: {text_mode}")
        if comparable not in ("base", "comp_both"):
            raise ValueError(f"Unknown comparable mode: {comparable}")

        self.model = model
        self.track = track
        self.text_mode = text_mode
        self.max_extra_terms = max_extra_terms
        self.min_df = min_df

        self.base = BoostingFeatures(include_time=True)
        self.text = None

        kwargs = (
            dict(age_filter_first=True, min_same_type_neighbors=8)
            if comparable == "comp_both"
            else {}
        )
        self.comparables = ComparableSalesFeatures(**kwargs)
        self.splines = SpaceTimeSplines() if splines else None
        self.geo_context = GeographicContextFeatures() if geo_context else None

        self.columns_ = None

    def _combine(self, homes, fitting=False, text_frame=None):
        method = "fit_transform" if fitting else "transform"

        if fitting:
            self.base, base = fit_block(
                self.base, homes, model=self.model, track=self.track
            )
        else:
            base = self.base.transform(homes, model=self.model)

        frames = [base]

        for name in ("comparables", "splines", "geo_context"):
            provider = getattr(self, name)
            if provider is None:
                continue

            if fitting:
                provider, frame = fit_block(provider, homes)
                setattr(self, name, provider)
            else:
                frame = provider.transform(homes)

            if not frame.index.equals(homes.index):
                raise ValueError("Feature row indices do not match")

            frames.append(frame)

            if provider is self.comparables and self.track == "list_aware":
                frames[-1] = add_list_comparable_features(frame, homes)

        X = pd.concat(frames, axis=1)

        if text_frame is None:
            text_frame = getattr(self.text, method)(homes)
        X = pd.concat([X, text_frame], axis=1)

        if X.columns.duplicated().any():
            raise ValueError("Duplicate feature columns")

        return X

    def fit_transform(self, train):
        bank = cached_value(
            train, ("text_bank", self.max_extra_terms, self.min_df),
            lambda: build_text_bank(train["remarks"],
                                    max_features=self.max_extra_terms,
                                    min_df=self.min_df),
        )

        terms = select_terms(bank, self.text_mode)
        self.text = BinaryTextFeatures(terms=terms)

        text_frame = training_text_features(self.text, train, bank)
        X = self._combine(train, fitting=True, text_frame=text_frame)
        self.columns_ = X.columns.tolist()

        return X

    def transform(self, homes):
        if self.columns_ is None:
            raise RuntimeError("Fit the pipeline first")

        X = self._combine(homes)

        return X[self.columns_]

    @property
    def categorical_features(self):
        return self.base.categorical_features

    @property
    def text_feature_names(self):
        return self.text.text_feature_names if self.text is not None else {}

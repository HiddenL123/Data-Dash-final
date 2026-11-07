
    
  
"""Training-fitted property, categorical, and temporal housing features.

Text, comparables, and spatial features are fitted by separate providers.
Use one instance per training fold/member and do not fit on held-out homes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from .cache import cached_value

CATEGORICAL_COLUMNS = (
    "house_type_name",
    "house_style",
    "municipality",
    "community",
    "garage",
)


def square_feet(house_area: pd.Series) -> pd.Series:
    """Convert a single area or an area range to approximate square footage."""
    values = house_area.astype("string").str.replace(",", "", regex=False)
    bounds = values.str.extract(r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?")
    low = pd.to_numeric(bounds[0], errors="coerce")
    high = pd.to_numeric(bounds[1], errors="coerce").fillna(low)
    return ((low + high) / 2).astype("float32")


class BoostingFeatures:
    """Base features; only categorical dtypes differ among boosting libraries."""

    def __init__(self, include_time: bool = True):
        self.include_time = include_time
        self.category_levels_ = None
        self.track_ = None
        self.columns_ = None

    @staticmethod
    def _column(homes: pd.DataFrame, name: str) -> pd.Series:
        if name not in homes:
            return pd.Series(np.nan, index=homes.index)
        return homes[name]

    def _numeric(self, homes: pd.DataFrame) -> pd.DataFrame:
        # Formatting adds categorical columns, so never expose the cached frame.
        return cached_value(
            homes, ("base_numeric", self.include_time, self.track_),
            lambda: self._build_numeric(homes),
        ).copy()

    def _build_numeric(self, homes: pd.DataFrame) -> pd.DataFrame:
        def number(name: str) -> pd.Series:
            return pd.to_numeric(self._column(homes, name), errors="coerce").astype("float32")

        result = pd.DataFrame(index=homes.index)
        result["square_feet"] = square_feet(self._column(homes, "house_area"))
        result["bedrooms"] = number("bedrooms")
        result["washrooms"] = number("washrooms")
        result["parking"] = number("parking_total")
        result["garage_count"] = number("garage")
        result["lot_front"] = number("lot_front")
        result["lot_depth"] = number("lot_depth")
        result["lot_area"] = number("lot_area").fillna(result["lot_front"] * result["lot_depth"])
        result["longitude"] = number("longitude")
        result["latitude"] = number("latitude")

        if self.include_time:
            dates = pd.to_datetime(self._column(homes, "date_listed"), errors="coerce")
            month = dates.dt.month
            result["months_since_2021"] = ((dates.dt.year - 2021) * 12 + month - 1).astype("float32")
            result["season_sin"] = np.sin(2 * np.pi * month / 12).astype("float32")
            result["season_cos"] = np.cos(2 * np.pi * month / 12).astype("float32")

        if self.track_ == "list_aware":
            prices = number("list_price")
            result["log_list_price"] = np.log(prices.where(prices > 0))

        return result.replace([np.inf, -np.inf], np.nan)

    def _categories(self, homes: pd.DataFrame) -> pd.DataFrame:
        return cached_value(homes, ("base_categories",),
                            lambda: self._build_categories(homes)).copy()

    def _build_categories(self, homes: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=homes.index)
        for column in CATEGORICAL_COLUMNS:
            value = self._column(homes, column).fillna("Unknown").astype(str).str.strip()
            result[column] = value.mask(value.eq(""), "Unknown")
        return result

    def fit(self, homes: pd.DataFrame, *, track: str = "off_market") -> "BoostingFeatures":
        if track not in ("off_market", "list_aware"):
            raise ValueError("track must be 'off_market' or 'list_aware'")
        self.track_ = track
        categories = self._categories(homes)
        self.category_levels_ = {
            name: sorted(set(categories[name]).union({"Unknown"}))
            for name in CATEGORICAL_COLUMNS
        }
        self.columns_ = list(self._numeric(homes).columns) + list(CATEGORICAL_COLUMNS)
        return self

    def transform(self, homes: pd.DataFrame, *, model: str = "catboost") -> pd.DataFrame:
        if self.category_levels_ is None:
            raise RuntimeError("Call fit() before transform()")
        if model not in ("catboost", "lightgbm", "xgboost"):
            raise ValueError(f"Unknown boosting model: {model}")
        result = self._numeric(homes)
        categories = self._categories(homes)
        for column in CATEGORICAL_COLUMNS:
            known = self.category_levels_[column]
            values = categories[column].where(categories[column].isin(known), "Unknown")
            result[column] = (
                values.astype(str)
                if model == "catboost"
                else values.astype(CategoricalDtype(categories=known))
            )
        return result[self.columns_]

    def fit_transform(self, homes: pd.DataFrame, *, model: str = "catboost", track: str = "off_market") -> pd.DataFrame:
        return self.fit(homes, track=track).transform(homes, model=model)

    @property
    def categorical_features(self) -> list[str]:
        return list(CATEGORICAL_COLUMNS)


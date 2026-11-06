import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import SplineTransformer

EARTH_KM = 6371.0088
    
def _number(frame, name, default=np.nan):
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _string(frame, name):
    if name not in frame:
        return np.full(len(frame), "Unknown", dtype=object)
    return frame[name].fillna("Unknown").astype(str).to_numpy()


def _month(frame, column):
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Expected valid {column} dates")
    return values.dt.to_period("M").astype("int64").to_numpy(dtype=int)


def _haversine(lat, lon, center_lat, center_lon):
    a, b, c, d = map(np.deg2rad, (lat, lon, center_lat, center_lon))
    h = np.sin((a - c) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((b - d) / 2) ** 2
    return EARTH_KM * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _coords(homes):
    dates = pd.to_datetime(homes["date_listed"], errors="coerce")
    periods = dates.dt.to_period("M").astype("int64").to_numpy(dtype=float)
    return np.column_stack((
        pd.to_numeric(homes["latitude"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(homes["longitude"], errors="coerce").to_numpy(dtype=float),
        periods,
    ))


class SpaceTimeSplines:
    """Original 3-axis cubic splines and pairwise tensor interactions."""

    def fit(self, sales: pd.DataFrame):
        coords = _coords(sales)
        self.fill_coords = np.nanmedian(
            np.where(np.isfinite(coords), coords, np.nan), axis=0
        )
        self.fill_coords = np.where(
            np.isfinite(self.fill_coords), self.fill_coords, 0.0
        )
        coords = np.where(np.isfinite(coords), coords, self.fill_coords)
        self.splines = [
            SplineTransformer(n_knots=5, degree=3, include_bias=False)
            .fit(coords[:, [i]])
            for i in range(3)
        ]
        return self

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "splines"):
            raise RuntimeError("Call fit() before transform()")
        xyz = _coords(homes)
        xyz = np.where(np.isfinite(xyz), xyz, self.fill_coords)
        pieces = [spline.transform(xyz[:, [i]]) for i, spline in enumerate(self.splines)]
        data = {}
        for axis, part in enumerate(pieces):
            for j in range(part.shape[1]):
                data[f"extra_spline_{axis}_{j}"] = part[:, j]
        for a, b, title in ((0, 1, "space"), (0, 2, "lat_time"), (1, 2, "lon_time")):
            cross = np.einsum("ij,ik->ijk", pieces[a], pieces[b])
            for j in range(cross.shape[1]):
                for k in range(cross.shape[2]):
                    data[f"extra_tensor_{title}_{j}_{k}"] = cross[:, j, k]
        return pd.DataFrame(data, index=homes.index).astype("float32")

    def fit_transform(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self.fit(sales).transform(sales)


class GeographicContextFeatures:
    """Historical sale densities and non-price geographic centroid features."""

    def fit(self, sales: pd.DataFrame):
        lat, lon = _number(sales, "latitude"), _number(sales, "longitude")
        months = _month(sales, "date_sold")
        good = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
        if not good.any():
            raise ValueError("No geocoded training sales")
        municipalities = _string(sales, "municipality")
        self.city_ = (float(np.median(lat[good])), float(np.median(lon[good])))
        centers = pd.DataFrame({"municipality": municipalities[good],
                                "lat": lat[good], "lon": lon[good]})
        self.centers_ = centers.groupby("municipality")[["lat", "lon"]].median()
        self.history_ = pd.DataFrame({"month": months[good], "lat": lat[good], "lon": lon[good]})
        return self

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "history_"):
            raise RuntimeError("Call fit() before transform()")
        lat, lon = _number(homes, "latitude"), _number(homes, "longitude")
        months = _month(homes, "date_listed")
        names = _string(homes, "municipality")
        output = pd.DataFrame(index=homes.index)
        output["geo_city_centroid_km"] = _haversine(lat, lon, *self.city_)
        centers = self.centers_.reindex(names)
        output["geo_municipality_centroid_km"] = _haversine(
            lat, lon, centers.lat.to_numpy(), centers.lon.to_numpy()
        )
        other = np.full(len(homes), np.nan)
        for name in np.unique(names):
            ids = np.flatnonzero(names == name)
            possible = self.centers_.drop(index=name, errors="ignore")
            if len(possible):
                km = np.column_stack([
                    _haversine(lat[ids], lon[ids], c.lat, c.lon)
                    for c in possible.itertuples()
                ])
                other[ids] = np.nanmin(km, axis=1)
        output["geo_nearest_other_municipality_centroid_km"] = other
        dense_1 = np.zeros(len(homes), dtype=float)
        dense_5 = np.zeros(len(homes), dtype=float)
        for month in np.unique(months):
            loc = np.flatnonzero(months == month)
            hist = self.history_.loc[(self.history_.month < month) &
                                     (self.history_.month >= month - 12)]
            valid = np.isfinite(lat[loc]) & np.isfinite(lon[loc]) & (abs(lat[loc]) <= 90) & (abs(lon[loc]) <= 180)
            if hist.empty or not valid.any():
                continue
            tree = BallTree(np.deg2rad(hist[["lat", "lon"]].to_numpy()), metric="haversine")
            coords = np.deg2rad(np.column_stack([lat[loc[valid]], lon[loc[valid]]]))
            dense_1[loc[valid]] = tree.query_radius(coords, r=1.0 / EARTH_KM, count_only=True)
            dense_5[loc[valid]] = tree.query_radius(coords, r=5.0 / EARTH_KM, count_only=True)
        output["geo_sales_count_1km_12m"] = dense_1
        output["geo_sales_count_5km_12m"] = dense_5
        return output.add_prefix("extra_geo_context_").astype("float32")

    def fit_transform(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self.fit(sales).transform(sales)


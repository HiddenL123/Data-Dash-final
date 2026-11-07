# Frozen pre-optimization implementation: independent regression oracle.
"""Historical comparable-sale features for housing price prediction.

Supports both off-market and list-aware ensembles.
Comparable statistics use only sales preceding the listing month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088


def _area(value: pd.Series) -> np.ndarray:
    pieces = value.astype('string').str.replace(',', '', regex=False).str.extract(
        r'^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?'
    )
    lo = pd.to_numeric(pieces[0], errors='coerce')
    hi = pd.to_numeric(pieces[1], errors='coerce').fillna(lo)
    return ((lo + hi) / 2).to_numpy(dtype=float)


def _months(series: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(series, errors='coerce')
    if dates.isna().any():
        raise ValueError('Valid date_listed/date_sold required for comparable features')
    return (dates.dt.year.to_numpy() * 12 + dates.dt.month.to_numpy()).astype(int)


class ComparableSalesFeatures:
    """Local historical sales features, with property- and recency-weighting.

    Parameters
    ----------
    candidates : int
        Geographically closest sales to consider before similarity weighting.
    neighbors : int
        Maximum comparables after similarity ranking.
    radius_km : float
        Geographic bandwidth; nearer sales get more weight.
    half_life_days : float
        Temporal weight halves after this many days.
    area_bandwidth : float
        Log-area differences are penalized on this scale.
    max_age_days : float
        Old sales beyond this age are excluded, not only downweighted.
    age_filter_first : bool
        If True, filter to the eligible age window *before* spatial nearest-
        neighbor search. Default False exactly preserves the original method.
    min_same_type_neighbors : int
        If nonzero, require up to this many same-property-type sales among the
        final `neighbors` (when eligible candidates exist). A separate type-
        specific BallTree augments the candidate pool, but weights still
        penalize distant/old/size-mismatched sales. Default 0 preserves legacy.

    Input columns: date_sold, sold_price, latitude, longitude, house_area,
    house_type_name (fit); date_listed, latitude, longitude, house_area,
    house_type_name (transform).
    """

    columns_ = [
        'comp_log_price', 'comp_log_ppsf', 'comp_local_ppsf_median',
        'comp_price_std', 'comp_ppsf_std', 'comp_count',
        'comp_effective_n', 'comp_nearest_km', 'comp_mean_distance_km',
        'comp_mean_age_days', 'comp_same_type_share', 'comp_area_similarity',
        'comp_recent_180d_share', 'comp_momentum_recent_older',
    ]

    def __init__(self, candidates=128, neighbors=32, radius_km=4.0,
                 half_life_days=180.0, area_bandwidth=0.45,
                 max_age_days=730.0, min_comps=2,
                 age_filter_first=False, min_same_type_neighbors=0):
        self.candidates = int(candidates)
        self.neighbors = int(neighbors)
        self.radius_km = float(radius_km)
        self.half_life_days = float(half_life_days)
        self.area_bandwidth = float(area_bandwidth)
        self.max_age_days = float(max_age_days)
        self.min_comps = int(min_comps)
        self.age_filter_first = bool(age_filter_first)
        if not isinstance(min_same_type_neighbors, (int, np.integer)):
            raise ValueError('min_same_type_neighbors must be an integer')
        self.min_same_type_neighbors = int(min_same_type_neighbors)
        if self.neighbors < 1 or self.candidates < self.neighbors:
            raise ValueError('Require 1 <= neighbors <= candidates')
        if not 0 <= self.min_same_type_neighbors <= self.neighbors:
            raise ValueError('Require 0 <= min_same_type_neighbors <= neighbors')
        if min(self.radius_km, self.half_life_days, self.area_bandwidth, self.max_age_days) <= 0:
            raise ValueError('Bandwidths and max_age_days must be positive')
        self._history = None

    def fit(self, sales: pd.DataFrame):
        required = {'date_sold','sold_price','latitude','longitude', 'house_area','house_type_name'}
        missing = required - set(sales.columns)
        if missing:
            raise KeyError(f'Missing sale fields: {sorted(missing)}')
        lat = pd.to_numeric(sales['latitude'], errors='coerce').to_numpy(dtype=float)
        lon = pd.to_numeric(sales['longitude'], errors='coerce').to_numpy(dtype=float)
        area = _area(sales['house_area'])
        price = pd.to_numeric(sales['sold_price'], errors='coerce').to_numpy(dtype=float)
        sold = pd.to_datetime(sales['date_sold'], errors='coerce')
        valid = (np.isfinite(lat)&np.isfinite(lon)&(np.abs(lat)<=90)&(np.abs(lon)<=180)&
                 np.isfinite(area)&(area>0)&np.isfinite(price)&(price>0)&sold.notna().to_numpy())
        if not valid.any():
            raise ValueError('No valid comparable sales found')
        self._history = pd.DataFrame({
            'lat': np.deg2rad(lat[valid]), 'lon': np.deg2rad(lon[valid]),
            'log_area': np.log(area[valid]), 'log_price': np.log(price[valid]),
            'type': sales.loc[valid, 'house_type_name'].fillna('Unknown').astype(str).to_numpy(),
            'sold_day': sold.loc[valid].to_numpy(dtype='datetime64[D]').astype('int64'),
            'month': _months(sold.loc[valid]),
        }).sort_values('month', kind='stable').reset_index(drop=True)
        return self

    def _one(self, query_coord, query_area, query_type, cutoff_day, hist, tree,
             same_type_hist=None, same_type_tree=None,
             neighbors=None, same_neighbors=None):
        result = np.full(len(self.columns_), np.nan, dtype=float)
        if tree is None:
            result[5:7] = 0
            return result
        k = min(self.candidates, len(hist))
        d, ix = (tree.query(query_coord[None, :], k=k)
                 if neighbors is None else neighbors)
        d = d[0] * EARTH_KM
        selected = hist.iloc[ix[0]]
        age = cutoff_day - selected['sold_day'].to_numpy(dtype=float)
        valid = (age >= 0) & (age <= self.max_age_days)
        if not np.any(valid) and not (
            self.min_same_type_neighbors and same_type_tree is not None
        ):
            result[5:7] = 0
            return result
        d, age = d[valid], age[valid]
        selected = selected.iloc[np.flatnonzero(valid)]

        if self.min_same_type_neighbors and same_type_tree is not None:
            # Search the same-type pool separately, so minority types are not
            # lost to the geographically nearest 128 mixed-type transactions.
            # Index labels refer to the same historical rows in both frames.
            same_k = min(self.candidates, len(same_type_hist))
            same_d, same_ix = (
                same_type_tree.query(query_coord[None, :], k=same_k)
                if same_neighbors is None else same_neighbors
            )
            same_selected = same_type_hist.iloc[same_ix[0]]
            same_d = same_d[0] * EARTH_KM
            same_age = cutoff_day - same_selected['sold_day'].to_numpy(dtype=float)
            same_valid = (same_age >= 0) & (same_age <= self.max_age_days)
            same_selected = same_selected.iloc[np.flatnonzero(same_valid)]
            same_d = same_d[same_valid]

            # De-duplicate houses already found through the mixed-type search.
            combined_ids = np.concatenate([
                selected.index.to_numpy(), same_selected.index.to_numpy()
            ])
            combined_distances = np.concatenate([d, same_d])
            is_new = ~pd.Index(combined_ids).duplicated(keep='first')
            selected = hist.loc[combined_ids[is_new]]
            d = combined_distances[is_new]
            age = cutoff_day - selected['sold_day'].to_numpy(dtype=float)

        if len(selected) == 0:
            result[5:7] = 0
            return result
        is_type = (selected['type'].to_numpy() == query_type).astype(float)
        delta_area = np.abs(selected['log_area'].to_numpy() - query_area)
        # Rank by combined similarity instead of geographic proximity alone.
        w = (np.exp(-d/self.radius_km) *
             np.exp(-np.log(2)*age/self.half_life_days) *
             np.exp(-delta_area/self.area_bandwidth) *
             np.where(is_type > 0, 1.0, 0.25))
        if self.min_same_type_neighbors:
            same_positions = np.flatnonzero(is_type > 0)
            # Quota is applied to final neighbors, not just candidates.
            quota = min(self.min_same_type_neighbors, len(same_positions))
            required = same_positions[np.argsort(-w[same_positions])[:quota]]
            remaining_mask = np.ones(len(w), dtype=bool)
            remaining_mask[required] = False
            available = np.flatnonzero(remaining_mask)
            extras = available[
                np.argsort(-w[available])[:self.neighbors - len(required)]
            ]
            keep = np.concatenate([required, extras])
        else:
            # Bit-for-bit original selection when the quota is disabled.
            keep = np.argsort(-w)[:self.neighbors]
        d, age, w = d[keep], age[keep], w[keep]
        selected = selected.iloc[keep]
        is_type, delta_area = is_type[keep], delta_area[keep]
        if len(w) < self.min_comps or np.sum(w) < 1e-12:
            result[5] = len(w)
            result[6] = 0
            return result
        w = w / w.sum()
        lp = selected['log_price'].to_numpy(dtype=float)
        ppsf = lp - selected['log_area'].to_numpy(dtype=float)
        avg_ppsf = np.sum(w*ppsf)
        result[0] = avg_ppsf + query_area  # size-adjusted local value estimate
        result[1] = avg_ppsf
        result[2] = np.median(ppsf)
        result[3] = np.sqrt(np.sum(w*(lp-np.sum(w*lp))**2))
        result[4] = np.sqrt(np.sum(w*(ppsf-avg_ppsf)**2))
        result[5] = len(w)
        result[6] = 1 / np.sum(w*w)
        result[7] = float(d.min())
        result[8] = np.sum(w*d)
        result[9] = np.sum(w*age)
        result[10] = np.sum(w*is_type)
        result[11] = np.sum(w*np.exp(-delta_area))
        recent = age < 180
        result[12] = np.sum(w*recent)
        older = ~recent
        if recent.any() and older.any():
            result[13] = (np.sum(w[recent]*ppsf[recent])/w[recent].sum() -
                          np.sum(w[older]*ppsf[older])/w[older].sum())
        return result

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError('Call fit() before transform()')
        required = {'date_listed','latitude','longitude','house_area','house_type_name'}
        missing = required - set(homes.columns)
        if missing:
            raise KeyError(f'Missing home fields: {sorted(missing)}')
        if len(homes) == 0:
            return pd.DataFrame(index=homes.index, columns=self.columns_, dtype=float)
        months = _months(homes['date_listed'])
        lat = pd.to_numeric(homes['latitude'], errors='coerce').to_numpy(dtype=float)
        lon = pd.to_numeric(homes['longitude'], errors='coerce').to_numpy(dtype=float)
        area = _area(homes['house_area'])
        types = homes['house_type_name'].fillna('Unknown').astype(str).to_numpy()
        valid_queries = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(area) &
                         (area > 0) & (abs(lat) <= 90) & (abs(lon) <= 180))
        out = np.full((len(homes), len(self.columns_)), np.nan)
        # The global reference history is frozen at the fit cutoff, but queries
        # continue to age naturally into future months.
        for month in np.unique(months):
            positions = np.flatnonzero((months == month) & valid_queries)
            if not len(positions):
                continue
            historical = self._history.loc[self._history['month'] < month]
            cutoff = pd.Timestamp(year=(month-1)//12, month=(month-1)%12+1, day=1)
            cutoff_day = int(cutoff.to_datetime64().astype('datetime64[D]').astype('int64'))
            if self.age_filter_first:
                sold_days = historical['sold_day'].to_numpy(dtype=float)
                eligible = (
                    (sold_days >= cutoff_day - self.max_age_days) &
                    (sold_days <= cutoff_day)
                )
                historical = historical.iloc[np.flatnonzero(eligible)]

            tree = (BallTree(historical[['lat','lon']].to_numpy(), metric='haversine')
                    if len(historical) else None)
            type_pools = {}
            if self.min_same_type_neighbors and tree is not None:
                # Build at most one additional pool per queried property type
                # in this month; reuse it across all homes of that type.
                for house_type in np.unique(types[positions]):
                    matching = historical.loc[historical['type'] == house_type]
                    if len(matching):
                        type_pools[house_type] = (
                            matching,
                            BallTree(matching[['lat', 'lon']].to_numpy(), metric='haversine')
                        )
            if tree is None:
                out[positions, 5:7] = 0
                continue
            # Bounded batches amortize tree-query overhead without allocating
            # a full homes-by-candidates matrix for a large prediction month.
            for start in range(0, len(positions), 512):
                batch = positions[start:start + 512]
                coords = np.deg2rad(np.column_stack([lat[batch], lon[batch]]))
                distances, indices = tree.query(coords, k=min(self.candidates, len(historical)))
                same_queries = {}
                for house_type, (same_hist, same_tree) in type_pools.items():
                    local = np.flatnonzero(types[batch] == house_type)
                    if not len(local):
                        continue
                    d, ix = same_tree.query(coords[local], k=min(self.candidates, len(same_hist)))
                    for j, offset in enumerate(local):
                        same_queries[offset] = (d[j:j + 1], ix[j:j + 1])
                for offset, pos in enumerate(batch):
                    same_hist, same_tree = type_pools.get(types[pos], (None, None))
                    out[pos] = self._one(
                        coords[offset], np.log(area[pos]), types[pos], cutoff_day,
                        historical, tree, same_type_hist=same_hist, same_type_tree=same_tree,
                        neighbors=(distances[offset:offset + 1], indices[offset:offset + 1]),
                        same_neighbors=same_queries.get(offset),
                    )
        return pd.DataFrame(out, columns=self.columns_, index=homes.index).astype('float32')

    def fit_transform(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self.fit(sales).transform(sales)

def add_list_comparable_features(features, homes):
    listed = pd.to_numeric(
        homes["list_price"], errors="coerce"
    )

    if not np.isfinite(listed).all() or (listed <= 0).any():
        raise ValueError("Invalid listing prices")

    result = features.copy()

    result["comp_vs_list_log"] = (
        result["comp_log_price"] - np.log(listed)
    ).astype("float32")

    return result

"""Regression checks for bounded query batching and residual-fit work."""
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import BallTree

from features.comparables import ComparableSalesFeatures, _months, _area
from models import models
from models.components import ComponentSpec
from test_starter import make_listings


def scalar_comparables(provider, homes):
    """Pre-batching path: one tree query per valid home, original month pools."""
    months = _months(homes.date_listed)
    lat = pd.to_numeric(homes.latitude).to_numpy(float)
    lon = pd.to_numeric(homes.longitude).to_numpy(float)
    area = _area(homes.house_area)
    types = homes.house_type_name.fillna('Unknown').astype(str).to_numpy()
    result = np.full((len(homes), len(provider.columns_)), np.nan)
    for month in np.unique(months):
        history = provider._history.loc[provider._history.month < month]
        cutoff = pd.Timestamp(year=(month - 1)//12, month=(month - 1)%12 + 1, day=1)
        day = int(cutoff.to_datetime64().astype('datetime64[D]').astype('int64'))
        if provider.age_filter_first:
            history = history.loc[(history.sold_day >= day - provider.max_age_days) &
                                  (history.sold_day <= day)]
        tree = BallTree(history[['lat', 'lon']].to_numpy(), metric='haversine') if len(history) else None
        pools = {}
        for pos in np.flatnonzero(months == month):
            if not (np.isfinite(lat[pos]) and np.isfinite(lon[pos]) and np.isfinite(area[pos])
                    and area[pos] > 0 and abs(lat[pos]) <= 90 and abs(lon[pos]) <= 180):
                continue
            same_hist, same_tree = None, None
            if provider.min_same_type_neighbors and tree is not None:
                if types[pos] not in pools:
                    same = history.loc[history.type == types[pos]]
                    pools[types[pos]] = (same, BallTree(same[['lat', 'lon']].to_numpy(), metric='haversine')) if len(same) else (None, None)
                same_hist, same_tree = pools[types[pos]]
            result[pos] = provider._one(np.deg2rad([lat[pos], lon[pos]]), np.log(area[pos]),
                                        types[pos], day, history, tree, same_hist, same_tree)
    return pd.DataFrame(result, columns=provider.columns_, index=homes.index).astype('float32')


@pytest.mark.parametrize('age_first,quota', [(False, 0), (True, 0), (False, 8), (True, 8)])
def test_batched_comparables_match_scalar(age_first, quota):
    train = make_listings(800)
    # Tied locations, minority type, and sales exactly on a month boundary.
    train.loc[:150, ['latitude', 'longitude']] = [43.7, -79.4]
    train.loc[:3, 'house_type_name'] = 'Rare'
    train.loc[:3, 'date_sold'] = pd.Timestamp('2025-05-01')
    homes = make_listings(650, seed=4).assign(date_listed=pd.Timestamp('2026-06-15'))
    homes.loc[:9, ['latitude', 'longitude']] = [43.7, -79.4]
    homes.loc[0, 'latitude'] = np.nan
    homes.loc[1, 'longitude'] = 181
    homes.loc[2, 'house_area'] = '0'
    homes.loc[3, 'house_type_name'] = 'Rare'
    homes.loc[4, 'house_type_name'] = 'Unseen'
    homes.loc[5, 'date_listed'] = pd.Timestamp('1900-01-01')
    homes.loc[6, 'date_listed'] = pd.Timestamp('2030-01-01')
    homes.loc[7:9, 'date_listed'] = pd.Timestamp('2025-05-01')
    # Duplicate/nonmonotonic labels must not affect positional alignment.
    homes.index = np.arange(len(homes))[::-1] // 2
    provider = ComparableSalesFeatures(age_filter_first=age_first, min_same_type_neighbors=quota).fit(train)
    expected = scalar_comparables(provider, homes)
    pd.testing.assert_frame_equal(provider.transform(homes), expected, check_exact=True)
    pd.testing.assert_frame_equal(provider.transform(homes.iloc[::-1]), expected.iloc[::-1], check_exact=True)
    assert provider.transform(homes.iloc[:0]).shape == (0, len(provider.columns_))


@pytest.mark.parametrize('track,mode', [('off_market', None), ('list_aware', 'ratio'), ('list_aware', 'direct')])
@pytest.mark.parametrize('count', [1, 3])
def test_fit_only_predicts_when_another_residual_is_needed(monkeypatch, track, mode, count):
    targets, predicted = [], []
    class RecordingComponent:
        def __init__(self, spec, *, track):
            pass
        def fit(self, train, target):
            targets.append(target.copy())
            return self
        def predict(self, homes):
            predicted.append(len(homes))
            return np.full(len(homes), .25)
    monkeypatch.setattr(models, 'Component', RecordingComponent)
    train = make_listings(20)
    config = dict(specs=[ComponentSpec()] * count, center_target=True)
    if mode:
        config['target_mode'] = mode
    model = models.make_model(track, config).fit(train)
    assert len(predicted) == count - 1
    target = np.log(train.sold_price.to_numpy())
    if mode == 'ratio':
        target -= np.log(train.list_price.to_numpy())
    target -= np.median(target)
    for i, actual in enumerate(targets):
        np.testing.assert_allclose(actual, target - i * .25)
    expected = model.target_offset_ + count * .25
    if mode == 'ratio':
        expected += np.log(train.list_price.to_numpy())
    np.testing.assert_allclose(model.predict_log(train), expected)

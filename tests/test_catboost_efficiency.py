"""Equivalence checks for historical reuse and vectorized spatial features."""
import contextlib
import io

import joblib
import numpy as np
import pandas as pd
import pytest

import data
from catboost_method import features, models
from test_starter import make_listings


def scalar_spatial(provider, homes):
    columns = ['geo_median_price_5', 'geo_median_price_10', 'geo_median_ppsf_10',
               'geo_nearest_distance', 'geo_median_area_10', 'geo_comparable_count',
               'geo_median_price_20', 'geo_median_ppsf_20']
    result = np.full((len(homes), len(columns)), np.nan, dtype=np.float32)
    def median(values):
        values = values[np.isfinite(values)]
        return np.median(values) if len(values) else np.nan
    for pos, (_, home) in enumerate(homes.iterrows()):
        history = provider.data.get(home.house_type_name)
        if history is None or pd.isna(home.latitude) or pd.isna(home.longitude):
            continue
        distances, indices = history['tree'].query(
            [float(home.latitude), float(home.longitude) * .72],
            k=min(100, len(history['dates'])), workers=1)
        distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
        legal = history['dates'][indices] < np.datetime64(pd.Timestamp(home.date_listed))
        selected = indices[legal]
        if not len(selected):
            continue
        prices, areas = history['prices'][selected], history['areas'][selected]
        pps = prices / np.where(areas > 0, areas, np.nan)
        result[pos] = [median(prices[:5]), median(prices[:10]), median(pps[:10]),
                       distances[legal][0], median(areas[:10]), len(selected),
                       median(prices[:20]), median(pps[:20])]
    return pd.DataFrame(result, index=homes.index, columns=columns)


@pytest.mark.parametrize('n', [1, 40, 650])
def test_spatial_matches_scalar_with_edge_cases(n):
    train = make_listings(n).assign(house_type_name='Detached')
    train.loc[:min(15, n-1), ['latitude', 'longitude']] = [43.7, -79.4]
    if n > 1:
        train.loc[0, 'sold_price'] = np.inf
        train.loc[1, 'house_area'] = '0'
        train.loc[2, 'house_area'] = None
    homes = make_listings(620, seed=8).assign(house_type_name='Detached', date_listed=pd.Timestamp('2027-01-01'))
    homes.loc[:20, ['latitude', 'longitude']] = [43.7, -79.4]
    homes.loc[0, 'latitude'] = np.nan
    homes.loc[1, 'longitude'] = np.nan
    homes.loc[2, 'house_type_name'] = 'Unknown type'
    homes.loc[4, 'date_listed'] = pd.Timestamp('1900-01-01')
    homes.loc[5, 'date_listed'] = train.date_sold.iloc[0]
    homes.loc[6, 'date_listed'] = train.date_sold.iloc[0] + pd.Timedelta(nanoseconds=1)
    homes.index = np.arange(len(homes))[::-1] // 2
    provider = features.SpatialComparableFeatures().fit(train)
    pd.testing.assert_frame_equal(provider.transform(homes), scalar_spatial(provider, homes), check_exact=True)
    pd.testing.assert_frame_equal(provider.transform(homes.iloc[::-1]), scalar_spatial(provider, homes.iloc[::-1]), check_exact=True)
    pd.testing.assert_frame_equal(provider.transform(homes.iloc[:0]), scalar_spatial(provider, homes.iloc[:0]), check_exact=True)
    missing_date = homes.iloc[[10]].assign(date_listed=pd.NaT)
    with pytest.raises(TypeError):
        scalar_spatial(provider, missing_date)
    with pytest.raises(TypeError):
        provider.transform(missing_date)


def test_historical_reuses_requests_and_invalidates_on_refit(monkeypatch):
    train = make_listings(80)
    homes = pd.concat([train.iloc[[0]]] * 20).assign(date_listed=pd.Timestamp('2026-01-01'))
    provider = features.HistoricalFeatures().fit(train)
    calls = []
    original = provider._stats
    def record(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(provider, '_stats', record)
    actual = provider.transform(homes)
    assert len(calls) == 3
    # One-row calls reproduce the old uncached request pattern.
    expected = pd.concat([provider.transform(homes.iloc[[i]]) for i in range(len(homes))])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    provider.fit(train.assign(sold_price=train.sold_price * 2))
    changed = provider.transform(homes)
    np.testing.assert_allclose(changed.type_median_price, actual.type_median_price * 2)
    missing = homes.assign(date_listed=pd.NaT)
    assert provider.transform(missing).type_sales_count.eq(0).all()
    assert provider.transform(homes.iloc[:0]).shape == (0, 16)


def test_history_cutoffs_and_windows():
    cutoff = pd.Timestamp('2026-06-01')
    train = make_listings(5).assign(community='One', house_type_name='Detached',
        sold_price=[100., 200., 300., 1e9, 1e9], house_area='1000',
        date_sold=[cutoff-pd.Timedelta(days=366), cutoff-pd.Timedelta(days=365),
                   cutoff-pd.Timedelta(days=90), cutoff, cutoff+pd.Timedelta(days=1)])
    homes = train.iloc[[0, 0]].assign(date_listed=cutoff)
    X = features.HistoricalFeatures().fit(train).transform(homes)
    assert X.type_median_price.eq(250).all()
    assert X.type_sales_count.eq(2).all()
    assert X.type_recent_trend.isna().all()  # Only one eligible sale in 90 days.


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
def test_models_fit_predict_save_and_do_not_mutate_input(track, tmp_path):
    train = make_listings(100)
    original = train.copy(deep=True)
    model = models.make_model(track)
    model.model.set_params(iterations=3, depth=3, thread_count=1, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(train)
        homes = data.hide_answers(train.iloc[[8, 4, 2]], track)
        predicted = model.predict(homes)
        reverse = model.predict(homes.iloc[::-1])
    pd.testing.assert_frame_equal(train, original, check_exact=True)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    np.testing.assert_array_equal(reverse, predicted[::-1])
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(homes), predicted)

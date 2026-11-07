"""Equivalence, isolation, refit, and serialization checks for feature reuse."""
import joblib
import numpy as np
import pandas as pd
import pytest

import data
from ensemble_method.features.cache import feature_cache, _active
from ensemble_method.features.comparables import ComparableSalesFeatures
from ensemble_method.features.pipeline import FeaturePipeline
from ensemble_method.models.components import ComponentSpec
from ensemble_method.models.models import make_model
from reference_ensemble_comparables import ComparableSalesFeatures as Reference
from test_starter import make_listings
from ensemble_method.features import base, text, pipeline


@pytest.mark.parametrize('age_first,quota', [(False, 0), (True, 0), (False, 8), (True, 8)])
def test_comparables_exact(age_first, quota):
    train = make_listings(450)
    train.loc[:40, ['latitude', 'longitude']] = [43.7, -79.4]
    train.loc[0, 'house_area'] = '0'
    train.loc[1, 'sold_price'] = np.nan
    homes = make_listings(620, seed=12)
    homes.loc[:40, ['latitude', 'longitude']] = [43.7, -79.4]
    homes.loc[2, 'latitude'] = np.nan
    homes.loc[3, 'house_type_name'] = 'Unseen'
    homes.loc[4, 'date_listed'] = pd.Timestamp('1900-01-01')
    homes.loc[5, 'date_listed'] = pd.Timestamp('2030-01-01')
    homes.index = np.arange(len(homes))[::-1] // 2
    params = dict(age_filter_first=age_first, min_same_type_neighbors=quota)
    actual = ComparableSalesFeatures(**params).fit(train)
    expected = Reference(**params).fit(train)
    for queries in (homes, homes.iloc[::-1], homes.iloc[:0]):
        pd.testing.assert_frame_equal(actual.transform(queries), expected.transform(queries), check_exact=True)


def test_cache_reuse_isolation_and_cleanup(monkeypatch):
    train = make_listings(100)
    calls = []
    original = ComparableSalesFeatures.fit_transform
    def record(self, frame):
        calls.append(len(frame))
        return original(self, frame)
    monkeypatch.setattr(ComparableSalesFeatures, 'fit_transform', record)
    with feature_cache():
        a, b = FeaturePipeline(), FeaturePipeline()
        pd.testing.assert_frame_equal(a.fit_transform(train), b.fit_transform(train), check_exact=True)
        assert calls == [100]
        assert a.comparables is not b.comparables
        a.comparables._history.loc[:, 'log_price'] = 0
        assert b.comparables._history.log_price.ne(0).all()
        FeaturePipeline(comparable='comp_both').fit_transform(train)
        FeaturePipeline().fit_transform(train.iloc[:50])
        assert calls == [100, 100, 50]
    assert _active.get() is None
    with pytest.raises(RuntimeError), feature_cache():
        raise RuntimeError('simulated fit failure')
    assert _active.get() is None


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
@pytest.mark.parametrize('family', ['xgboost', 'lightgbm', 'catboost'])
def test_cached_model_predictions_refit_and_pickle(track, family, tmp_path):
    train = make_listings(100)
    original = train.copy(deep=True)
    homes = data.hide_answers(make_listings(12, seed=2), track)
    params = (dict(iterations=3, depth=2, thread_count=1, verbose=False, allow_writing_files=False)
              if family == 'catboost' else dict(n_estimators=3, max_depth=2, n_jobs=1))
    if family == 'lightgbm':
        params['verbosity'] = -1
    train['remarks'] = ['Renovated family home with a garden', 'Cozy condo with balcony'] * 50
    original = train.copy(deep=True)
    specs = [ComponentSpec(estimator=family, params=params,
                           features=dict(min_df=1, text_mode='full' if mode == 'subbag' else 'selected'),
                           row_mode=mode, row_fraction=fraction)
             for mode, fraction in [('all', 1.), ('subbag', .8), ('bootstrap', .8)]]
    ensemble = make_model(track, dict(members={str(i): dict(specs=[spec]) for i, spec in enumerate(specs)},
                                      weights={str(i): 1 for i in range(3)}))
    ensemble.fit(train)
    expected = []
    for i, spec in enumerate(specs):
        independent = make_model(track, [spec]).fit(train)
        np.testing.assert_array_equal(ensemble.models[str(i)].components[0].training_rows_,
                                      independent.components[0].training_rows_)
        expected.append(independent.predict(homes))
    np.testing.assert_array_equal(ensemble.predict_members(homes).to_numpy(), np.column_stack(expected))
    path = tmp_path / 'model.joblib'
    joblib.dump(ensemble, path)
    np.testing.assert_array_equal(joblib.load(path).predict(homes), ensemble.predict(homes))
    changed = train.assign(sold_price=train.sold_price * 1.2)
    ensemble.fit(changed)
    for i, spec in enumerate(specs):
        np.testing.assert_array_equal(ensemble.models[str(i)].predict(homes),
                                      make_model(track, [spec]).fit(changed).predict(homes))
    pd.testing.assert_frame_equal(train, original, check_exact=True)


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
def test_text_and_base_reuse_across_families_and_modes(track, monkeypatch):
    train = make_listings(80)
    train['remarks'] = ['Fully renovated family home. Garden garden!',
                        'Cozy condo, CENTRAL AIR and a balcony', None, ''] * 20
    train.loc[0, 'community'] = '  '
    train.index = np.arange(len(train))[::-1] // 2
    configs = [dict(model=family, text_mode=mode, track=track, min_df=1)
               for family in ('xgboost', 'catboost', 'lightgbm')
               for mode in ('selected', 'full')]
    expected = [FeaturePipeline(**config).fit_transform(train) for config in configs]
    calls = dict(numeric=0, categories=0, bank=0, tokens=0)
    def count(owner, name, key):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, wrapped)
    count(base.BoostingFeatures, '_build_numeric', 'numeric')
    count(base.BoostingFeatures, '_build_categories', 'categories')
    count(pipeline, 'build_text_bank', 'bank')
    count(text.CountVectorizer, 'transform', 'tokens')
    with feature_cache():
        fitted = []
        for config, reference in zip(configs, expected):
            provider = FeaturePipeline(**config)
            actual = provider.fit_transform(train)
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
            fitted.append(provider)
        assert calls == dict(numeric=1, categories=1, bank=1, tokens=1)
        assert fitted[0].base is not fitted[1].base
        assert fitted[0].text.vectorizer_ is not fitted[1].text.vectorizer_
        # Outputs are independently assembled; mutations cannot poison reuse.
        actual.iloc[0, 0] = -999
        again = FeaturePipeline(**configs[-1]).fit_transform(train)
        pd.testing.assert_frame_equal(again, expected[-1], check_exact=True)
        FeaturePipeline(**(configs[0] | dict(min_df=2))).fit_transform(train)
        assert calls['bank'] == 2
        changed = train.assign(remarks='Entirely different vocabulary')
        FeaturePipeline(**configs[0]).fit_transform(changed)
        assert calls['bank'] == 3
    assert _active.get() is None


def test_shared_text_matches_original_vectorization():
    train = make_listings(60)
    train['remarks'] = ['Fully renovated cozy garden garden!', 'New family home', None] * 20
    bank = text.build_text_bank(train.remarks, min_df=1)
    with feature_cache():
        for mode in ('selected', 'full'):
            terms = text.select_terms(bank, mode)
            original = text.CountVectorizer(vocabulary=list(terms), ngram_range=(1, 2),
                                           binary=True, dtype=np.float32).fit(text._remarks(train))
            provider = text.BinaryTextFeatures(terms=terms)
            actual = text.training_text_features(provider, train, bank)
            np.testing.assert_array_equal(actual.to_numpy(), original.transform(text._remarks(train)).toarray())
            assert provider.text_feature_names == dict(zip(actual.columns, original.get_feature_names_out()))
            # Prediction still uses only this member's vocabulary, including unseen text.
            homes = train.iloc[:3].assign(remarks=['Unseen token', 'fully renovated', ''])
            np.testing.assert_array_equal(provider.transform(homes).to_numpy(),
                                          original.transform(text._remarks(homes)).toarray())

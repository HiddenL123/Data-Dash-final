"""Real process-worker equivalence, scheduling, and failure regression tests."""
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest

import data
from ensemble_method.models import workers
from ensemble_method.models.components import ComponentSpec
from ensemble_method.models.models import make_model
from ensemble_method.features.comparables import ComparableSalesFeatures
from ensemble_method.features.cache import _active
from test_starter import make_listings


def config():
    members = {}
    for i, family in enumerate(('xgboost', 'lightgbm', 'catboost', 'xgboost')):
        params = (dict(iterations=3, depth=2, thread_count=1, verbose=False, allow_writing_files=False)
                  if family == 'catboost' else dict(n_estimators=3, max_depth=2, n_jobs=1))
        if family == 'lightgbm':
            params['verbosity'] = -1
        spec = ComponentSpec(estimator=family, params=params, features=dict(min_df=1,
                             text_mode='full' if i % 2 else 'selected'),
                             row_mode='subbag' if i % 2 else 'all',
                             row_fraction=.8 if i % 2 else 1.)
        members[str(i)] = dict(specs=[spec, deepcopy(spec)] if i == 3 else [spec],
                              center_target=True,
                              training_cutoff='2026-01-01' if i == 3 else None)
    return dict(members=members, weights={str(i): i + 1 for i in range(4)})


@pytest.mark.parametrize('track,count', [('off_market', 2), ('list_aware', 3), ('off_market', 4)])
def test_processes_match_sequential(track, count, tmp_path, monkeypatch):
    train = make_listings(100)
    train['remarks'] = ['Renovated family home', 'Cozy balcony garden'] * 50
    original = train.copy(deep=True)
    homes = data.hide_answers(make_listings(10, seed=8), track)
    sequential = make_model(track, config() | dict(workers=1)).fit(train)
    calls = []
    fit = ComparableSalesFeatures.fit_transform
    def record(self, frame):
        calls.append(len(frame))
        return fit(self, frame)
    monkeypatch.setattr(ComparableSalesFeatures, 'fit_transform', record)
    parallel = make_model(track, config() | dict(workers=count)).fit(train)
    # One shared block for the first three members; another for the cutoff member.
    assert len(calls) == 2
    assert _active.get() is None
    assert list(parallel.models) == list(sequential.models)
    np.testing.assert_allclose(parallel.predict(homes), sequential.predict(homes), rtol=1e-12, atol=1e-8)
    for name in parallel.models:
        a, b = parallel.models[name], sequential.models[name]
        for c, d in zip(a.components, b.components):
            np.testing.assert_array_equal(c.training_rows_, d.training_rows_)
            assert c.spec.params == d.spec.params
            threads = workers.worker_budget(count, 4)[1]
            key = 'thread_count' if c.family == 'catboost' else 'n_jobs'
            assert c.estimator.get_params()[key] == threads
    path = tmp_path / 'parallel.joblib'
    joblib.dump(parallel, path)
    np.testing.assert_array_equal(joblib.load(path).predict(homes), parallel.predict(homes))
    changed = train.assign(sold_price=train.sold_price * 1.1)
    parallel.fit(changed)
    sequential.fit(changed)
    np.testing.assert_allclose(parallel.predict(homes), sequential.predict(homes), rtol=1e-12, atol=1e-8)
    pd.testing.assert_frame_equal(train, original, check_exact=True)


def test_budget_and_environment(monkeypatch):
    monkeypatch.setattr(workers, 'cpu_count', lambda: 8)
    monkeypatch.setenv('ENSEMBLE_WORKERS', 'auto')
    assert workers.worker_budget(None, 4) == (4, 2)
    assert workers.worker_budget(None, 3) == (3, 2)
    assert workers.worker_budget(2, 4) == (2, 4)
    assert workers.worker_budget(20, 12) == (8, 1)
    assert workers.worker_budget(1, 4) == (1, 8)
    for invalid in (0, -1, 1.5, True, 'bad'):
        with pytest.raises(ValueError, match='workers'):
            workers.worker_budget(invalid, 4)


def test_worker_failure_leaves_ensemble_unfitted():
    settings = config()
    settings['members']['2']['specs'][0].params['loss_function'] = 'invalid_loss'
    model = make_model('off_market', settings | dict(workers=2))
    with pytest.raises(Exception, match='invalid_loss'):
        model.fit(make_listings(60))
    assert not model._fitted
    assert _active.get() is None
    with pytest.raises(RuntimeError, match='fit'):
        model.predict(data.hide_answers(make_listings(2), 'off_market'))

"""Bounded-lifetime reuse of target-independent feature blocks during one fit."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from joblib import hash as fingerprint


_active = ContextVar("ensemble_feature_cache", default=None)


@contextmanager
def feature_cache():
    # Never survive refits, folds, or exceptions. Each fit gets its own scope.
    token = _active.set({})
    try:
        yield
    finally:
        _active.reset(token)


def cached_value(homes, key, build):
    """Reuse a read-only value for the same frame and configuration in this fit.

    Callers must not mutate the frame during the scope or the returned value.
    Retaining the frame prevents Python from recycling its identity.
    """
    cache = _active.get()
    if cache is None:
        return build()
    key = ("value", id(homes), key)
    if key not in cache:
        cache[key] = (homes, build())
    return cache[key][1]


def fit_block(provider, homes, **kwargs):
    cache = _active.get()
    if cache is None:
        return provider, provider.fit_transform(homes, **kwargs)
    key = (id(homes), type(provider), fingerprint((provider.__dict__, kwargs)))
    if key not in cache:
        frame = provider.fit_transform(homes, **kwargs)
        # Retain homes to prevent object-id reuse; keep fitted state independent.
        cache[key] = (homes, deepcopy(provider), frame)
        return provider, frame
    _, fitted, frame = cache[key]
    return deepcopy(fitted), frame

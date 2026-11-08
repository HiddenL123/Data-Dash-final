"""Process workers for estimator fitting; feature preparation stays in the parent.

Set ENSEMBLE_WORKERS to 2, 3, 4, or auto before starting Python, or pass
workers=... in an ensemble configuration. The default is 1 (sequential).
Auto caps workers at the member count and available CPUs. Each worker gets
floor(CPUs/workers) estimator threads. Months are never parallelized here.
"""
import os
from time import perf_counter

from joblib import Parallel, delayed, parallel_config, cpu_count

from ensemble_method.features.cache import feature_cache

# Temporary sequential benchmark override. Set to None to restore configuration.
WORKER_OVERRIDE = 1


def worker_budget(requested, members):
    if WORKER_OVERRIDE is not None:
        requested = WORKER_OVERRIDE
    if requested is None:
        requested = os.environ.get("ENSEMBLE_WORKERS", "1")
    cores = max(1, cpu_count())
    if requested == "auto":
        count = min(cores, members)
    else:
        try:
            count = int(requested)
        except (ValueError, TypeError):
            raise ValueError("workers must be a positive integer or 'auto'") from None
        if isinstance(requested, bool) or str(count) != str(requested) or count < 1:
            raise ValueError("workers must be a positive integer or 'auto'")
        count = min(count, members, cores)
    return count, max(1, cores // count)


def _fit_member(name, model, prepared, threads):
    start = perf_counter()
    model._finish_parallel_fit(prepared, threads)
    print(f"[{model.track}] Worker finished {name} in {perf_counter() - start:.1f}s", flush=True)
    return name, model


def fit_members(models, train, workers, threads):
    # All feature construction runs in one cache scope before workers start.
    # This retains one final matrix per component until dispatch completes.
    prepared = []
    with feature_cache():
        for name, model in models.items():
            print(f"[{model.track}] Preparing member {name}", flush=True)
            prepared.append((name, model, model._prepare_parallel_fit(train)))
    with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=threads):
        results = Parallel(pre_dispatch=workers, batch_size=1)(
            delayed(_fit_member)(name, model, features, threads)
            for name, model, features in prepared
        )
    # Joblib preserves submission order, and thus the existing weight order.
    return dict(results)

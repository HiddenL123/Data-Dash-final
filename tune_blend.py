"""Walk-forward select and independently verify team blending weights.

Usage (from repository root):
    uv run python tune_blend.py select   # Sep-Nov 2025: fit constituent models, choose weights
    uv run python tune_blend.py verify   # Dec 2025-May 2026: frozen weights, NO retuning

Edit FACTORIES in blended_model.py before selecting. Prediction caches are
separate for select and verify; a crash can be resumed without refitting
previously completed factory/month/track combinations.

IMPORTANT: verify is a held-out EVALUATION; do not optimize weights based on it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import data
from blended_model import FACTORIES, TRACKS, WEIGHTS_FILE, load_factories, validate_weights

# These date ranges intentionally do not overlap.
SELECTION_MONTHS = pd.period_range("2025-09", "2025-11", freq="M")
VERIFICATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
OUT = WEIGHTS_FILE.parent


def metrics(true, predicted):
    ape = 100 * np.abs(np.asarray(predicted) - np.asarray(true)) / np.asarray(true)
    return {"mape": float(np.mean(ape)), "mdape": float(np.median(ape)),
            "p90_ape": float(np.percentile(ape, 90))}


def blend(prices, weights):
    return np.exp(np.log(prices) @ validate_weights(weights, prices.shape[1]))


def _cache_name(phase, track, month, factory):
    # Hash the exact factory path, so renaming/reordering won't silently read the wrong model.
    digest = hashlib.sha256(factory.encode()).hexdigest()[:12]
    return OUT / "predictions" / phase / f"{track}_{month}_{digest}.csv"


def get_predictions(phase, months):
    """Generate aligned per-home predictions for ALL factories and tracks."""
    listings = data.load("listings")
    if not {"date_sold", "sold_price", "ml_num"}.issubset(listings.columns):
        raise ValueError("Historical listings lack required columns")
    factories = load_factories()
    output = []
    failures = []
    for month in months:
        train = listings.loc[listings.date_sold < month.start_time]
        homes = listings.loc[listings.date_sold.dt.to_period("M") == month].copy()
        if train.empty or homes.empty:
            raise ValueError(f"Empty training or validation split for {month}")
        if homes["ml_num"].duplicated().any():
            raise ValueError(f"Duplicate ml_num in {month}; cannot align predictions")
        print(f"\n{phase.upper()} {month}: {len(train):,} train / {len(homes):,} holdout", flush=True)
        for track in TRACKS:
            hidden = data.hide_answers(homes, track)
            aligned = pd.DataFrame({"ml_num": homes.ml_num.to_numpy(),
                                    "sold_price": homes.sold_price.to_numpy(dtype=float),
                                    "month": str(month)})
            for j, (path, factory) in enumerate(zip(FACTORIES, factories)):
                cache = _cache_name(phase, track, month, path)
                try:
                    if cache.exists():
                        saved = pd.read_csv(cache)
                        # Strict identity and order check; do not merge on row numbers.
                        if not np.array_equal(saved.ml_num.to_numpy(), homes.ml_num.to_numpy()):
                            raise ValueError(f"Cached ml_num or ordering changed: {cache}")
                        if not np.array_equal(saved.sold_price.to_numpy(dtype=float), homes.sold_price.to_numpy(dtype=float)):
                            raise ValueError(f"Cached sold_price changed: {cache}")
                        prediction = saved.predicted_price.to_numpy(dtype=float)
                        print(f"  [{track}] reused {path}", flush=True)
                    else:
                        model = factory(track)
                        print(f"  [{track}] fitting {path}", flush=True)
                        model.fit(train)
                        prediction = np.asarray(model.predict(hidden), dtype=float)
                        if prediction.shape != (len(homes),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
                            raise ValueError("Invalid predictions")
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame({"ml_num": homes.ml_num.to_numpy(),
                                      "sold_price": homes.sold_price.to_numpy(dtype=float),
                                      "predicted_price": prediction}).to_csv(cache, index=False)
                    if prediction.shape != (len(homes),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
                        raise ValueError("Invalid cached predictions")
                    aligned[f"model_{j}"] = prediction
                except Exception as exc:
                    failures.append({"month": str(month), "track": track,
                                     "factory": path, "error": str(exc),
                                     "traceback": traceback.format_exc()})
                    print(f"  FAILED [{track}] {path}: {exc}", flush=True)
            if all(f"model_{j}" in aligned for j in range(len(FACTORIES))):
                aligned["track"] = track
                output.append(aligned)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{phase}_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} fits failed; see {OUT / (phase + '_failures.json')}. "
                           "Fix failures, then rerun to reuse successful caches.")
    return pd.concat(output, ignore_index=True)


def candidate_weights(count):
    """Deterministic, practical search; 2 factories get an exact 0.01 grid."""
    if count == 2:
        for alpha in np.linspace(0, 1, 101):
            yield np.array([1 - alpha, alpha])
        return
    # For 3+ models, include corners and equal blend plus a seeded simplex search.
    yield np.ones(count) / count
    for index in range(count):
        corner = np.zeros(count)
        corner[index] = 1
        yield corner
    rng = np.random.default_rng(42)
    for _ in range(10000):
        yield rng.dirichlet(np.ones(count))


def select():
    if WEIGHTS_FILE.exists():
        raise FileExistsError(f"Weights already frozen at {WEIGHTS_FILE}. "
                              "Remove or explicitly archive this file before a new selection.")
    frame = get_predictions("select", SELECTION_MONTHS)
    score_rows = []
    chosen = {}
    cols = [f"model_{i}" for i in range(len(FACTORIES))]
    for track in TRACKS:
        sub = frame.loc[frame.track == track]
        actual = sub.sold_price.to_numpy(dtype=float)
        preds = sub[cols].to_numpy(dtype=float)
        best_key, best_w, best_metrics = None, None, None
        for weights in candidate_weights(len(FACTORIES)):
            scores = metrics(actual, blend(preds, weights))
            # Select MdAPE first; break exact ties by MAPE, then preference for simpler corners.
            key = (scores["mdape"], scores["mape"])
            if best_key is None or key < best_key:
                best_key, best_w, best_metrics = key, weights.copy(), scores
        chosen[track] = best_w.tolist()
        for index, path in enumerate(FACTORIES):
            score_rows.append({"track": track, "factory": path, **metrics(actual, preds[:, index])})
        print(f"\n{track}: selected weights {dict(zip(FACTORIES, np.round(best_w, 4)))}")
        print(f"Selection performance: {best_metrics}")
    frame.to_csv(OUT / "selection_member_predictions.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUT / "selection_individual_scores.csv", index=False)
    payload = {"factories": list(FACTORIES), "selection_months": [str(p) for p in SELECTION_MONTHS],
               "selection_metric": "pooled_mdape", "weights": chosen}
    tmp = WEIGHTS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(WEIGHTS_FILE)
    print(f"\nFROZEN: {WEIGHTS_FILE}")


def verify():
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"Run selection first: {WEIGHTS_FILE}")
    saved = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    if saved.get("factories") != list(FACTORIES):
        raise ValueError("Current factory order differs from frozen configuration")
    if saved.get("selection_months") != [str(p) for p in SELECTION_MONTHS]:
        raise ValueError("Frozen weights were not selected on the configured three months")
    frame = get_predictions("verify", VERIFICATION_MONTHS)
    cols = [f"model_{i}" for i in range(len(FACTORIES))]
    records = []
    frame["blended_price"] = np.nan
    for track in TRACKS:
        sub = frame.loc[frame.track == track]
        weights = validate_weights(saved["weights"][track], len(FACTORIES))
        prediction = blend(sub[cols].to_numpy(dtype=float), weights)
        frame.loc[sub.index, "blended_price"] = prediction
        records.append({"month": "pooled", "track": track, "model": "frozen_blend", "homes": len(sub),
                        **metrics(sub.sold_price, prediction)})
        for i, name in enumerate(FACTORIES):
            records.append({"month": "pooled", "track": track, "model": name, "homes": len(sub),
                            **metrics(sub.sold_price, sub[cols[i]])})
        for month, group in sub.groupby("month", sort=True):
            records.append({"month": month, "track": track, "model": "frozen_blend", "homes": len(group),
                            **metrics(group.sold_price, blend(group[cols].to_numpy(dtype=float), weights))})
    results = pd.DataFrame(records)
    frame.to_csv(OUT / "verification_predictions.csv", index=False)
    results.to_csv(OUT / "verification_scores.csv", index=False)
    print("\nFROZEN-WEIGHT VERIFICATION (no weight tuning):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"select", "verify"}:
        raise SystemExit("Usage: uv run python tune_blend.py [select|verify]")
    {"select": select, "verify": verify}[sys.argv[1]]()

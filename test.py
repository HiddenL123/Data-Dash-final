
from copy import deepcopy
import traceback

import numpy as np
import data

from models.models import make_model, OffMarketModel, ListAwareModel


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

train = data.load("listings").iloc[:2000].copy()

failures = []
successes = []


def check(name, function):
    try:
        function()
        successes.append(name)
        print(f"PASS: {name}")

    except Exception as exc:
        failures.append({
            "test": name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(f"FAIL: {name} — {type(exc).__name__}: {exc}")


def small_spec(spec):
    """Copy an accepted component specification with fewer trees."""
    spec = deepcopy(spec)
    if spec.estimator == "catboost":
        spec.params["iterations"] = 30
    else:
        spec.params["n_estimators"] = 30
    return spec


def get_homes(track):
    return data.hide_answers(train.tail(20).copy(), track)


def assert_prices(predictions, n):
    values = np.asarray(predictions, dtype=float)
    assert values.shape == (n,), f"Unexpected shape: {values.shape}"
    assert np.isfinite(values).all(), "Nonfinite predictions"
    assert (values > 0).all(), "Nonpositive predictions"


# ------------------------------------------------------------
# Test 1: Weighted ensemble blending
# ------------------------------------------------------------

def test_ensemble(track):
    original = make_model(track)

    # Select two accepted members.
    selected = list(original.models.items())[:2]

    members = {}
    for name, member in selected:
        specs = [small_spec(spec) for spec in member.specs]

        if track == "off_market":
            members[name] = OffMarketModel(
                specs,
                center_target=member.center_target,
            )
        else:
            members[name] = ListAwareModel(
                specs,
                center_target=member.center_target,
                target_mode=member.target_mode,
            )

    weights = {
        name: original.weights[name]
        for name in members
    }

    config = {
        "members": {
            name: {
                "specs": member.specs,
                "center_target": member.center_target,
                **(
                    {"target_mode": member.target_mode}
                    if track == "list_aware"
                    else {}
                ),
            }
            for name, member in members.items()
        },
        "weights": weights,
    }

    ensemble = make_model(track, specs=config)

    ensemble.fit(train)

    homes = get_homes(track)

    # Predictions from individual members.
    member_predictions = ensemble.predict_members(homes)

    assert list(member_predictions.columns) == list(members)

    for name in members:
        assert_prices(member_predictions[name], len(homes))

    # Predictions from the ensemble.
    blended = ensemble.predict(homes)
    assert_prices(blended, len(homes))

    # Independently calculate the geometric weighted mean.
    w = np.array(
        [weights[name] for name in member_predictions.columns],
        dtype=float,
    )
    w /= w.sum()

    expected = np.exp(
        np.log(member_predictions.to_numpy(dtype=float)) @ w
    )

    np.testing.assert_allclose(
        blended,
        expected,
        rtol=1e-10,
        atol=1e-8,
    )

    print(f"  {track}: {len(members)} members, {len(homes)} predictions")
    print(f"  Mean blended price: ${blended.mean():,.0f}")


check(
    "Off-market weighted ensemble",
    lambda: test_ensemble("off_market"),
)

check(
    "List-aware weighted ensemble",
    lambda: test_ensemble("list_aware"),
)


# ------------------------------------------------------------
# Test 2: Sequential residual components
# ------------------------------------------------------------

def test_residual(track):
    original = make_model(track)

    # Use two different accepted component configurations.
    member_specs = [
        small_spec(member.specs[0])
        for member in list(original.models.values())[:2]
    ]

    if track == "off_market":
        model = OffMarketModel(
            member_specs,
            center_target=True,
        )
    else:
        model = ListAwareModel(
            member_specs,
            center_target=True,
            target_mode="ratio",
        )

    model.fit(train)

    homes = get_homes(track)

    # Sum the residual components manually.
    component_outputs = [
        component.predict(homes)
        for component in model.components
    ]

    expected_target = (
        model.target_offset_
        + sum(component_outputs)
    )

    # Check on the model's working target scale.
    np.testing.assert_allclose(
        model._predict_target(homes),
        expected_target,
        rtol=1e-10,
        atol=1e-10,
    )

    # Check final price reconstruction.
    if track == "list_aware":
        expected_prices = (
            homes["list_price"].to_numpy(dtype=float)
            * np.exp(expected_target)
        )
    else:
        expected_prices = np.exp(expected_target)

    actual_prices = model.predict(homes)

    np.testing.assert_allclose(
        actual_prices,
        expected_prices,
        rtol=1e-10,
        atol=1e-8,
    )

    assert_prices(actual_prices, len(homes))

    print(f"  {track}: {len(model.components)} residual components")


check(
    "Off-market sequential residual model",
    lambda: test_residual("off_market"),
)

check(
    "List-aware sequential residual model",
    lambda: test_residual("list_aware"),
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 65)
print(f"RESULTS: {len(successes)} passed, {len(failures)} failed")

if failures:
    print("\nFAILED TESTS:")
    for failure in failures:
        print(f"  {failure['test']}: {failure['error']}")

    print("\nFULL TRACEBACKS:")
    for failure in failures:
        print(f"\n--- {failure['test']} ---")
        print(failure["traceback"])

else:
    print("All integration tests passed.")

"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from . import features
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return FeatureModel(features.list_aware_features)
        # return FeatureModel(features.list_aware_features)   # try this instead
    return LightGBMModel()
    # return FeatureModel(features.off_market_features)       # try this instead


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            random_state=0,
        )

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))
class CatBoostModel:
    def __init__(self):
        self.cat_columns = [
            "community",
            "municipality",
            "house_style",
            "house_type_name",
        ]
        self.model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.08,
            loss_function="MAE",
            thread_count=4,
            random_seed=0,
            verbose=200,
            allow_writing_files=False,

        )

    def make_features(self, homes):
        X = features.off_market_features(homes).copy()


        for column in self.cat_columns:
            X[column] = (
                homes[column]
                .fillna("Unknown")
                .astype(str)
            )
        remarks = (
            homes["remarks"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        condition_patterns = {
            "renovated": (
                r"renovated|fully redone|completely redone|"
                r"redone from top to bottom"
            ),
            "updated": (
                r"updated kitchen|updated bathroom|"
                r"new countertops|newer flooring|newer windows"
            ),
            "luxury": (
                r"luxury|high-end|designer|"
                r"heated floors|heated bathroom floors"
            ),
            "needs_work": (
                r"needs work|needs updating|requires updating|"
                r"requires renovation|fixer.upper|handyman|"
                r"full cosmetic overhaul"
            ),
            "original_dated": (
                r"original finishes|\bdated\b|tired finishes|"
                r"older appliances|older windows"
            ),
            "as_is": r"as.is|where.is",
            "well_maintained": (
                r"well.maintained|well.cared.for|"
                r"clean and cared for|good condition"
            ),
            "move_in_ready": r"move.in.ready|turnkey",
            "finished_basement": r"\bfinished basement\b",
            "unfinished_basement": r"\bunfinished basement\b",
        }

        for name, pattern in condition_patterns.items():
            X["remarks_" + name] = remarks.str.contains(
                pattern, regex=True
            ).astype(int)
        return X

    def fit(self, train):
        X = self.make_features(train)
        y = np.log(train["sold_price"])

        self.model.fit(
            X,
            y,
            cat_features=self.cat_columns,
        )
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        return np.exp(self.model.predict(X))

class LightGBMModel(CatBoostModel):
    def __init__(self):
        self.cat_columns = [
            "community",
            "municipality",
            "house_style",
            "house_type_name",
        ]
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.08182323783418449,
            num_leaves=128,
            max_depth=16,
            min_child_samples=40,
            reg_lambda=2.4835467240461706,
            colsample_bytree=0.7174250836504598,
            subsample=0.9598528437324806,
            subsample_freq=1,
            random_state=0,
            n_jobs=4,
            verbosity=-1,
        )


    def make_features(self, homes):
        X = super().make_features(homes)

        for column in self.cat_columns:
            X[column] = X[column].astype("category")

        return X

    def fit(self, train):
        X = self.make_features(train)
        y = np.log(train["sold_price"])

        self.model.fit(
            X,
            y,
            categorical_feature=self.cat_columns,
        )
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        return np.exp(self.model.predict(X))
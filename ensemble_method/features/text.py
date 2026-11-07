"""Training-fitted binary bag-of-words features for winning ensemble members.

'full' means the full selected-terms-plus-training-vocabulary bank, NOT the
n_text setting found in the old candidate roster. The text column numbering
follows the explicit selected-terms-plus-training-vocabulary bank order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from .cache import cached_value

# Terms and ordering verified against accepted checkpoint feature manifests.
_PRESELECTED = (
    "tfidf_as is", "tfidf_fully", "tfidf_first", "tfidf_downsizers",
    "tfidf_young professionals", "tfidf_investors", "tfidf_starting",
    "tfidf_discerning", "tfidf_special", "tfidf_family", "tfidf_growing",
    "tfidf_offered", "tfidf_renovated", "tfidf_move up", "tfidf_potential",
    "tfidf_first time", "tfidf_vision", "tfidf_entertaining", "tfidf_redone",
    "tfidf_entertain", "tfidf_move", "tfidf_time", "tfidf_new",
    "tfidf_original", "tfidf_older", "tfidf_custom", "tfidf_growing family",
    "tfidf_couple", "tfidf_opportunity", "tfidf_central air", "tfidf_grow",
    "tfidf_front", "tfidf_discerning buyer", "tfidf_turnkey",
    "tfidf_investor", "tfidf_project", "tfidf_starting out",
    "tfidf_unit", "tfidf_cozy", "tfidf_spa like", "tfidf_wide plank",
    "tfidf_top", "tfidf_room to", "tfidf_beautifully", "tfidf_needs",
    "tfidf_bones", "tfidf_overhaul",
)
SELECTED_TERMS = tuple(term.removeprefix("tfidf_") for term in _PRESELECTED)
NGRAM_RANGE = (1, 2)


def _remarks(homes: pd.DataFrame) -> pd.Series:
    if "remarks" not in homes.columns:
        return pd.Series("", index=homes.index, dtype=str)
    return homes["remarks"].fillna("").astype(str)


def training_vocabulary(remarks: pd.Series, *, max_features: int = 1000, min_df: int = 10) -> tuple[str, ...]:
    """Learn additional unigram/bigram terms using training remarks only."""
    if not isinstance(max_features, int) or max_features < 1:
        raise ValueError("max_features must be a positive integer")
    if not isinstance(min_df, int) or min_df < 1:
        raise ValueError("min_df must be a positive integer")
    documents = pd.Series(remarks).fillna("").astype(str)
    selector = CountVectorizer(max_features=max_features, min_df=min_df,
                               ngram_range=NGRAM_RANGE, dtype=np.float32)
    return tuple(selector.fit(documents).get_feature_names_out())


def build_text_bank(remarks: pd.Series, *, max_features: int = 1000, min_df: int = 10) -> tuple[str, ...]:
    """Reproduce original bank: all 47 anchors, then fitted extra terms."""
    extra = training_vocabulary(remarks, max_features=max_features, min_df=min_df)
    return tuple(dict.fromkeys((*SELECTED_TERMS, *extra)))


def select_terms(bank: tuple[str, ...], mode: str) -> tuple[str, ...]:
    """Select only the two modes used by the seven accepted members."""
    if mode == "full":
        return tuple(bank)
    if mode == "selected":
        selected = set(SELECTED_TERMS)
        result = tuple(term for term in bank if term in selected)
        if len(result) != len(selected):
            raise ValueError("Text bank is missing a predefined reference term")
        return result
    raise ValueError("mode must be 'selected' or 'full'")


class BinaryTextFeatures:
    """Standalone BoW provider; composition with base features is external."""

    def __init__(self, *, terms: tuple[str, ...] | None = None):
        self.terms = tuple(terms) if terms is not None else None
        self.vectorizer_ = None
        self.columns_ = None

    def fit(self, homes: pd.DataFrame) -> "BinaryTextFeatures":
        if not self.terms:
            raise ValueError("Pass training-derived terms from select_terms()")
        self.vectorizer_ = CountVectorizer(
            vocabulary=list(self.terms), ngram_range=NGRAM_RANGE,
            binary=True, dtype=np.float32,
        ).fit([""])
        # Vocabulary is fixed: fitting only initializes sklearn's vocabulary
        # metadata. Scanning training documents here would discard their counts.
        self.columns_ = [f"text_{i}" for i in range(len(self.vectorizer_.get_feature_names_out()))]
        return self

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit() before transform()")
        matrix = self.vectorizer_.transform(_remarks(homes))
        return pd.DataFrame(matrix.toarray(), index=homes.index, columns=self.columns_)

    def fit_transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        return self.fit(homes).transform(homes)

    @property
    def text_feature_names(self) -> dict[str, str]:
        if self.vectorizer_ is None:
            return {}
        return dict(zip(self.columns_, self.vectorizer_.get_feature_names_out()))


def training_text_features(provider, homes, bank):
    """Tokenize the full bank once; densify only this member's selected columns."""
    def build():
        full = BinaryTextFeatures(terms=bank).fit(homes)
        return full.vectorizer_.transform(_remarks(homes))

    matrix = cached_value(homes, ("binary_text", bank), build)
    provider.fit(homes)
    positions = {term: index for index, term in enumerate(bank)}
    selected = [positions[term] for term in provider.terms]
    return pd.DataFrame(matrix[:, selected].toarray(), index=homes.index,
                        columns=provider.columns_)



def condition_features(homes: pd.DataFrame) -> pd.DataFrame:
    """Ten handcrafted property-condition indicators from listing remarks."""
    remarks = _remarks(homes).str.lower()

    patterns = {
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

    return pd.DataFrame(
        {
            f"remarks_{name}": remarks.str.contains(
                pattern, regex=True
            ).astype(int)
            for name, pattern in patterns.items()
        },
        index=homes.index,
    )

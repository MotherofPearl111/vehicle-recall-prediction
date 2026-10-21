"""Train and compare classifiers that predict whether a complaint is followed
by a recall (same vehicle + component group) within 365 days.

Evaluation design
-----------------
* Out-of-time split: train on complaints received 2015-2021, test on 2023-2024.
  2022 is a gap year: a label needs 365 days of follow-up, so in January 2023
  the outcome of a 2022 complaint is not yet known. Skipping it keeps any
  "future" recall information out of training, as in real deployment.
* Rolling (forward-chaining) validation inside the training years: for each
  validation year V in 2018-2021, fit on complaints from 2015 to V-2 and score
  year V. An earlier version used random folds grouped by vehicle; those gave
  AUCs ~0.1 higher than the true future test set, because mixing years lets
  the model learn era-specific patterns that do not carry forward.
* Metrics: ROC AUC (ranking quality), PR AUC / average precision (better for
  rare outcomes; only ~7% of complaints are positive), accuracy, and
  precision among the 10% of complaints the model ranks highest.
"""

import json
import sqlite3
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nhtsa.db"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
SEED = 42
VALIDATION_YEARS = [2018, 2019, 2020, 2021]

TARGET = "recall_within_1y"
BINARY = ["crash", "fire", "medical_attn", "towed", "police_report", "prior_recall_2y"]
COUNTS = ["injured", "deaths", "miles", "speed",
          "prior_cmpl_90d", "prior_cmpl_365d", "prior_vehicle_cmpl_90d"]
NUMERIC = ["vehicle_age"]
CATEGORICAL = ["comp_group", "make", "cmpl_type", "fuel_type"]
TEXT = "text"


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM model_data", con)
    df[COUNTS] = df[COUNTS].fillna(0).clip(lower=0)
    df["vehicle_age"] = df["vehicle_age"].clip(-1, 30)
    df[CATEGORICAL] = df[CATEGORICAL].fillna("NA")
    df[TEXT] = df[TEXT].fillna("")
    return df


def structured_preprocessor(scale: bool) -> ColumnTransformer:
    """Binary flags pass through, counts are log-transformed, categories one-hot."""
    log_counts = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
    if scale:
        log_counts.append(("scale", StandardScaler()))
    numeric = [("scale", StandardScaler())] if scale else [("pass", "passthrough")]
    return ColumnTransformer([
        ("bin", "passthrough", BINARY),
        ("counts", Pipeline(log_counts), COUNTS),
        ("num", Pipeline(numeric), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200,
                              sparse_output=scale), CATEGORICAL),
    ])


def build_models() -> dict[str, Pipeline]:
    text_and_structured = ColumnTransformer([
        ("structured", structured_preprocessor(scale=True), BINARY + COUNTS + NUMERIC + CATEGORICAL),
        # Words only: numbers are dropped so the model cannot memorize recall
        # campaign IDs (e.g. "15V313000") or model years quoted in the text
        ("text", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=50,
                                 max_features=30_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
                                 stop_words="english"), TEXT),
    ])
    return {
        "Logistic regression (structured)": Pipeline([
            ("prep", structured_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "Logistic regression (structured + text)": Pipeline([
            ("prep", text_and_structured),
            ("clf", LogisticRegression(C=0.5, max_iter=2000)),
        ]),
        "Random forest (structured)": Pipeline([
            ("prep", structured_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=50,
                                           max_features="sqrt", n_jobs=-1, random_state=SEED)),
        ]),
        "Gradient boosting (structured)": Pipeline([
            ("prep", structured_preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                   max_leaf_nodes=31, min_samples_leaf=200,
                                                   l2_regularization=1.0, random_state=SEED)),
        ]),
    }


def precision_at_top(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.10) -> float:
    k = int(len(scores) * frac)
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, scores >= 0.5),
        "precision_top10pct": precision_at_top(y_true, scores),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = load_data()
    train = df[df["recv_year"] <= 2021].reset_index(drop=True)
    test = df[df["recv_year"] >= 2023].reset_index(drop=True)
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    print(f"Train: {len(train):,} complaints ({y_train.mean():.1%} positive), years 2015-2021")
    print(f"Test:  {len(test):,} complaints ({y_test.mean():.1%} positive), years 2023-2024")

    years = train["recv_year"].to_numpy()
    folds = [(np.where(years <= v - 2)[0], np.where(years == v)[0]) for v in VALIDATION_YEARS]

    cv_rows, test_rows, test_scores = [], [], {}
    for name, model in build_models().items():
        print(f"\n== {name}")
        t0 = time.time()
        for year, (tr, va) in zip(VALIDATION_YEARS, folds):
            model.fit(train.iloc[tr], y_train[tr])
            scores = model.predict_proba(train.iloc[va])[:, 1]
            cv_rows.append({"model": name, "val_year": year, **evaluate(y_train[va], scores)})
            print(f"  validate {year}: AUC {cv_rows[-1]['roc_auc']:.3f}")

        # Refit on all training years, then score the held-out future years once
        model.fit(train, y_train)
        scores = model.predict_proba(test)[:, 1]
        test_scores[name] = scores
        test_rows.append({"model": name, **evaluate(y_test, scores)})
        print(f"  test AUC {test_rows[-1]['roc_auc']:.3f}  ({time.time() - t0:.0f}s)")

        if name == "Logistic regression (structured)":
            save_odds_ratios(model)
        if name == "Logistic regression (structured + text)":
            save_top_terms(model)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    cv_summary = cv_df.groupby("model").agg(["mean", "std"]).drop(columns="val_year")
    cv_summary.columns = [f"{m}_{s}" for m, s in cv_summary.columns]
    cv_summary.round(4).to_csv(RESULTS_DIR / "cv_summary.csv")
    test_df = pd.DataFrame(test_rows).set_index("model").round(4)
    test_df.to_csv(RESULTS_DIR / "test_results.csv")

    meta = {
        "n_train": len(train), "n_test": len(test),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
    }
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(meta, indent=2))

    print("\nRolling validation (mean over 2018-2021):")
    print(cv_summary.filter(like="_mean").round(3).to_string())
    print("\nHeld-out test set (2023-2024):")
    print(test_df.to_string())

    plot_curves(y_test, test_scores)


def save_odds_ratios(model: Pipeline) -> None:
    """Odds ratios from the structured logistic regression. Binary flags are
    0/1, so their OR compares yes vs no; log counts and vehicle age are
    standardized, so their OR is per 1 SD increase."""
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    odds = pd.DataFrame({"feature": names, "odds_ratio": np.exp(coefs)})
    odds.round(3).sort_values("odds_ratio").to_csv(RESULTS_DIR / "odds_ratios.csv", index=False)

    labels = {
        "bin__prior_recall_2y": "Recall for same vehicle + component in past 2 yrs",
        "bin__crash": "Crash reported", "bin__fire": "Fire reported",
        "bin__medical_attn": "Medical attention needed", "bin__towed": "Vehicle towed",
        "bin__police_report": "Police report",
        "counts__prior_cmpl_90d": "Similar complaints, prior 90 days (per SD)",
        "counts__prior_cmpl_365d": "Similar complaints, prior 365 days (per SD)",
        "counts__prior_vehicle_cmpl_90d": "Any complaints on vehicle, prior 90 days (per SD)",
        "counts__miles": "Mileage (per SD)", "counts__speed": "Speed (per SD)",
        "counts__injured": "Injuries (per SD)", "counts__deaths": "Deaths (per SD)",
        "num__vehicle_age": "Vehicle age (per SD)",
    }
    plot = odds[odds["feature"].isin(labels)].copy()
    plot["label"] = plot["feature"].map(labels)
    plot = plot.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#C44E52" if v > 1 else "#4C72B0" for v in plot["odds_ratio"]]
    ax.barh(plot["label"], plot["odds_ratio"], color=colors)
    ax.axvline(1, color="k", lw=1)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set(title="Logistic regression odds ratios\n(adjusted for component group and make)",
           xlabel="Odds ratio of a recall within 1 year (log scale)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "odds_ratios.png", dpi=150)


def save_top_terms(model: Pipeline, n: int = 25) -> None:
    """Text terms with the largest positive / negative logistic coefficients."""
    prep = model.named_steps["prep"]
    names = prep.get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    terms = pd.DataFrame({"feature": names, "coef": coefs})
    terms = terms[terms["feature"].str.startswith("text__")]
    terms["feature"] = terms["feature"].str.removeprefix("text__")
    top = pd.concat([terms.nlargest(n, "coef"), terms.nsmallest(n, "coef")])
    top.round(3).to_csv(RESULTS_DIR / "top_text_terms.csv", index=False)


def plot_curves(y_test: np.ndarray, test_scores: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, scores in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, scores):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, scores)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, scores):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    axes[0].set(title="ROC curve, held-out 2023-2024 complaints",
                xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="Random guess")
    axes[1].set(title="Precision-recall curve, held-out 2023-2024 complaints",
                xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png", dpi=150)
    print(f"\nSaved {FIG_DIR / 'roc_pr_curves.png'}")


if __name__ == "__main__":
    main()

"""Evaluate leakage-free uncertainty-aware stacking of clinical and activity models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_clinical_ceiling import (
    bootstrap_intervals,
    build_preprocessor,
    load_table,
    metric_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=("core", "extended"), default="core")
    parser.add_argument(
        "--clinical-model",
        choices=("lasso", "random_forest"),
        default="lasso",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def activity_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.001,
                    class_weight="balanced",
                    max_iter=5_000,
                    penalty="l2",
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def clinical_pipeline(
    dictionary: pd.DataFrame,
    feature_names: list[str],
    seed: int,
    model_name: str,
) -> Pipeline:
    if model_name == "lasso":
        classifier = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=5_000,
            penalty="l1",
            random_state=seed,
            solver="liblinear",
        )
    else:
        classifier = RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=20,
            n_estimators=600,
            n_jobs=1,
            random_state=seed,
        )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(dictionary, feature_names, dense=False)),
            ("classifier", classifier),
        ]
    )


def meta_features(
    clinical_probability: np.ndarray,
    activity_probability: np.ndarray,
    reliability_mean: np.ndarray,
    reliability_minimum: np.ndarray,
) -> np.ndarray:
    clinical_score = logit(clinical_probability)
    activity_score = logit(activity_probability)
    uncertainty = 4 * clinical_probability * (1 - clinical_probability)
    return np.column_stack(
        [
            clinical_score,
            activity_score,
            reliability_mean,
            reliability_minimum,
            activity_score * uncertainty,
            activity_score * reliability_mean,
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(args.covariates)
    labels = pd.read_csv(args.labels)
    dictionary = pd.read_csv(args.dictionary)
    cohort = labels.loc[labels["phq9_complete"].eq(1), ["SEQN", "depression"]].merge(
        table,
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )
    targets = cohort["depression"].to_numpy(dtype=np.int8)
    activity_sources = {"ACTIVITY_CONCEPTS", "MANTIS_V2"}
    activity_names = dictionary.loc[
        dictionary["source"].isin(activity_sources), "feature"
    ].tolist()
    clinical_dictionary = dictionary.loc[
        ~dictionary["source"].isin(activity_sources)
    ].copy()
    roles = {"core"} if args.scenario == "core" else {"core", "extended"}
    clinical_names = clinical_dictionary.loc[
        clinical_dictionary["role"].isin(roles), "feature"
    ].tolist()
    reliability_mean = cohort["activity_reliability_mean"].to_numpy(float)
    reliability_minimum = cohort["activity_reliability_minimum"].to_numpy(float)
    predictions = np.full(len(cohort), np.nan, dtype=float)
    fold_results: list[dict[str, object]] = []
    outer_cv = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )

    for fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(cohort, targets), start=1
    ):
        inner_cv = StratifiedKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + fold,
        )
        clinical_model = clinical_pipeline(
            clinical_dictionary,
            clinical_names,
            args.seed + fold,
            args.clinical_model,
        )
        activity_model = activity_pipeline(args.seed + fold)
        clinical_oof = cross_val_predict(
            clinical_model,
            cohort.iloc[train_indices][clinical_names],
            targets[train_indices],
            cv=inner_cv,
            method="predict_proba",
            n_jobs=args.jobs,
        )[:, 1]
        activity_oof = cross_val_predict(
            activity_model,
            cohort.iloc[train_indices][activity_names],
            targets[train_indices],
            cv=inner_cv,
            method="predict_proba",
            n_jobs=args.jobs,
        )[:, 1]
        meta_train = meta_features(
            clinical_oof,
            activity_oof,
            reliability_mean[train_indices],
            reliability_minimum[train_indices],
        )
        meta_search = GridSearchCV(
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                penalty="l2",
                solver="liblinear",
            ),
            {"C": [0.01, 0.1, 1.0, 10.0]},
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=args.jobs,
        )
        meta_search.fit(meta_train, targets[train_indices])
        clinical_fitted = clone(clinical_model).fit(
            cohort.iloc[train_indices][clinical_names],
            targets[train_indices],
        )
        activity_fitted = clone(activity_model).fit(
            cohort.iloc[train_indices][activity_names],
            targets[train_indices],
        )
        clinical_test = clinical_fitted.predict_proba(
            cohort.iloc[test_indices][clinical_names]
        )[:, 1]
        activity_test = activity_fitted.predict_proba(
            cohort.iloc[test_indices][activity_names]
        )[:, 1]
        meta_test = meta_features(
            clinical_test,
            activity_test,
            reliability_mean[test_indices],
            reliability_minimum[test_indices],
        )
        fold_probabilities = meta_search.predict_proba(meta_test)[:, 1]
        predictions[test_indices] = fold_probabilities
        fold_metric = metric_values(targets[test_indices], fold_probabilities)
        fold_results.append(
            {
                "fold": fold,
                "test_auroc": fold_metric["auroc"],
                "meta_best_c": meta_search.best_params_["C"],
                "meta_coefficients": meta_search.best_estimator_.coef_[0].tolist(),
            }
        )
        print(
            f"fold {fold}/{args.outer_folds}: AUROC={fold_metric['auroc']:.4f}",
            flush=True,
        )

    metrics = metric_values(targets, predictions)
    intervals = bootstrap_intervals(
        targets,
        predictions,
        args.bootstrap_iterations,
        args.seed,
    )
    pd.DataFrame(
        {
            "SEQN": cohort["SEQN"].astype(np.int64),
            "label": targets,
            "oof_probability": predictions,
        }
    ).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    result = {
        "scenario": args.scenario,
        "clinical_model": args.clinical_model,
        "participants": int(len(cohort)),
        "clinical_features": int(len(clinical_names)),
        "activity_features": int(len(activity_names)),
        "metrics": metrics,
        "bootstrap_95_ci": intervals,
        "folds": fold_results,
        "protocol": "Outer-fold evaluation with inner-fold out-of-fold base predictions.",
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

"""Estimate the leakage-controlled clinical ceiling with nested cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_NAMES = ("lasso_logistic", "hist_gradient_boosting", "random_forest")
SCENARIOS = ("core", "extended", "sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def feature_names_for_scenario(dictionary: pd.DataFrame, scenario: str) -> list[str]:
    allowed_roles = {
        "core": {"core"},
        "extended": {"core", "extended"},
        "sensitivity": {"core", "extended", "sensitivity"},
    }[scenario]
    return dictionary.loc[dictionary["role"].isin(allowed_roles), "feature"].tolist()


def build_preprocessor(
    dictionary: pd.DataFrame,
    feature_names: list[str],
    dense: bool,
) -> ColumnTransformer:
    kinds = dictionary.set_index("feature")["kind"]
    categorical = [name for name in feature_names if kinds[name] == "categorical"]
    numeric = [name for name in feature_names if name not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def build_search(
    model_name: str,
    dictionary: pd.DataFrame,
    feature_names: list[str],
    inner_cv: StratifiedKFold,
    seed: int,
    jobs: int,
) -> GridSearchCV:
    if model_name == "lasso_logistic":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
            solver="liblinear",
            tol=1e-4,
        )
        grid = {
            "classifier__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "classifier__penalty": ["l1", "l2"],
        }
        dense = False
    elif model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        )
        grid = {
            "classifier__learning_rate": [0.03, 0.07, 0.1],
            "classifier__max_iter": [150, 300],
            "classifier__max_leaf_nodes": [7, 15, 31],
            "classifier__l2_regularization": [0.0, 1.0],
        }
        dense = True
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
        grid = {
            "classifier__n_estimators": [300, 600],
            "classifier__max_depth": [4, 8, None],
            "classifier__min_samples_leaf": [5, 20],
            "classifier__max_features": ["sqrt", 0.5],
        }
        dense = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(dictionary, feature_names, dense=dense)),
            ("classifier", classifier),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=jobs,
        refit=True,
        return_train_score=False,
    )


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "prevalence": float(labels.mean()),
    }


def bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_probabilities = probabilities[indices]
        values = metric_values(sampled_labels, sampled_probabilities)
        for name in estimates:
            estimates[name].append(values[name])
    return {
        name: [float(value) for value in np.quantile(samples, [0.025, 0.975])]
        for name, samples in estimates.items()
    }


def fit_oof(
    features: pd.DataFrame,
    labels: np.ndarray,
    dictionary: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    jobs: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    fold_results: list[dict[str, object]] = []
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(features, labels), start=1
    ):
        inner_cv = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + fold,
        )
        search = build_search(
            model_name,
            dictionary,
            feature_names,
            inner_cv,
            seed + fold,
            jobs,
        )
        search.fit(features.iloc[train_indices], labels[train_indices])
        fold_probabilities = search.predict_proba(features.iloc[test_indices])[:, 1]
        predictions[test_indices] = fold_probabilities
        fold_results.append(
            {
                "fold": fold,
                "train_n": int(len(train_indices)),
                "test_n": int(len(test_indices)),
                "inner_best_auroc": float(search.best_score_),
                "outer_auroc": float(roc_auc_score(labels[test_indices], fold_probabilities)),
                "best_params": search.best_params_,
            }
        )
        print(
            f"{model_name} fold {fold}/{outer_folds}: "
            f"AUROC={fold_results[-1]['outer_auroc']:.4f}",
            flush=True,
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Some participants did not receive an out-of-fold prediction.")
    return predictions, fold_results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    covariates = load_table(args.covariates)
    labels = pd.read_csv(args.labels)
    dictionary = pd.read_csv(args.dictionary)
    cohort = labels.loc[labels["phq9_complete"].eq(1)].merge(
        covariates,
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )
    targets = cohort["depression"].to_numpy(dtype=np.int8)
    all_results: dict[str, object] = {}

    for scenario in args.scenarios:
        feature_names = feature_names_for_scenario(dictionary, scenario)
        features = cohort[feature_names]
        for model_index, model_name in enumerate(args.models):
            run_name = f"{scenario}__{model_name}"
            run_seed = args.seed + 100 * SCENARIOS.index(scenario) + model_index
            print(f"Starting {run_name} with {len(feature_names)} source features", flush=True)
            probabilities, fold_results = fit_oof(
                features=features,
                labels=targets,
                dictionary=dictionary,
                feature_names=feature_names,
                model_name=model_name,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                seed=run_seed,
                jobs=args.jobs,
            )
            metrics = metric_values(targets, probabilities)
            intervals = bootstrap_intervals(
                targets,
                probabilities,
                args.bootstrap_iterations,
                run_seed,
            )
            run_dir = args.output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "SEQN": cohort["SEQN"].astype(np.int64),
                    "label": targets,
                    "oof_probability": probabilities,
                }
            ).to_csv(run_dir / "oof_predictions.csv", index=False)
            result = {
                "scenario": scenario,
                "model": model_name,
                "participants": int(len(cohort)),
                "source_feature_count": int(len(feature_names)),
                "features": feature_names,
                "metrics": metrics,
                "bootstrap_95_ci": intervals,
                "folds": fold_results,
                "selection_protocol": "Nested CV; every reported probability is out of fold.",
            }
            with (run_dir / "results.json").open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
            all_results[run_name] = result
            print(f"Finished {run_name}: {json.dumps(metrics)}", flush=True)

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)


if __name__ == "__main__":
    main()

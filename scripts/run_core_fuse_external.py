"""Evaluate reliability-conditioned four-branch fusion across NHANES cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_uncertainty_stack import clinical_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-table", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--internal-baseline", type=Path, required=True)
    parser.add_argument("--external-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("full", "no_rar", "no_reliability", "no_mantis"),
        default="full",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def load_cohort(table_path: Path, labels_path: Path) -> pd.DataFrame:
    table = pd.read_pickle(table_path)
    labels = pd.read_csv(labels_path)
    complete = labels.loc[labels["phq9_complete"].eq(1), ["SEQN", "depression"]]
    return complete.merge(table, on="SEQN", how="inner", validate="one_to_one")


def dense_logistic(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5_000,
                    penalty="l2",
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def clipped_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def reliability_meta_features(
    branch_probabilities: np.ndarray,
    branch_names: list[str],
    use_reliability: bool,
    activity_mean: np.ndarray,
    activity_minimum: np.ndarray,
    rar_mean: np.ndarray,
    rar_minimum: np.ndarray,
) -> np.ndarray:
    logits = clipped_logit(branch_probabilities)
    if not use_reliability:
        return logits
    reliability_columns = []
    interaction_columns = []
    if any(name in branch_names for name in ("activity_concepts", "mantis_v2")):
        reliability_columns.extend([activity_mean, activity_minimum])
    if "classic_rar" in branch_names:
        reliability_columns.extend([rar_mean, rar_minimum])
    for index, name in enumerate(branch_names):
        if name in {"activity_concepts", "mantis_v2"}:
            interaction_columns.append(logits[:, index] * activity_mean)
        elif name == "classic_rar":
            interaction_columns.append(logits[:, index] * rar_mean)
    return np.column_stack([logits, *reliability_columns, *interaction_columns])


def make_meta_search(cv: StratifiedKFold, jobs: int) -> GridSearchCV:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3_000,
        penalty="l2",
        solver="liblinear",
    )
    return GridSearchCV(
        model,
        {"C": [0.01, 0.1, 1.0]},
        cv=cv,
        scoring="average_precision",
        n_jobs=jobs,
    )


def branch_predictions(
    models: list[Pipeline],
    feature_sets: list[list[str]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    cv: StratifiedKFold,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_predictions = []
    test_predictions = []
    for model, names in zip(models, feature_sets):
        inner = cross_val_predict(
            clone(model),
            train[names],
            labels,
            cv=cv,
            method="predict_proba",
            n_jobs=jobs,
        )[:, 1]
        fitted = clone(model).fit(train[names], labels)
        train_predictions.append(inner)
        test_predictions.append(fitted.predict_proba(test[names])[:, 1])
    return np.column_stack(train_predictions), np.column_stack(test_predictions)


def reliability_arrays(cohort: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        cohort["activity_reliability_mean"].to_numpy(float),
        cohort["activity_reliability_minimum"].to_numpy(float),
        cohort["rar_reliability_mean"].to_numpy(float),
        cohort["rar_reliability_minimum"].to_numpy(float),
    )


def nested_internal_predictions(
    models: list[Pipeline],
    feature_sets: list[list[str]],
    branch_names: list[str],
    use_reliability: bool,
    cohort: pd.DataFrame,
    labels: np.ndarray,
    folds: int,
    seed: int,
    jobs: int,
) -> np.ndarray:
    predictions = np.zeros(len(cohort), dtype=float)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(outer.split(cohort, labels), 1):
        train = cohort.iloc[train_index]
        test = cohort.iloc[test_index]
        inner = StratifiedKFold(
            n_splits=folds - 1,
            shuffle=True,
            random_state=seed + fold,
        )
        branch_train, branch_test = branch_predictions(
            models,
            feature_sets,
            train,
            test,
            labels[train_index],
            inner,
            jobs,
        )
        meta_train = reliability_meta_features(
            branch_train,
            branch_names,
            use_reliability,
            *reliability_arrays(train),
        )
        meta_test = reliability_meta_features(
            branch_test,
            branch_names,
            use_reliability,
            *reliability_arrays(test),
        )
        search = make_meta_search(inner, jobs).fit(meta_train, labels[train_index])
        predictions[test_index] = search.predict_proba(meta_test)[:, 1]
        print(f"Completed outer fold {fold}/{folds}; meta C={search.best_params_['C']}")
    return predictions


def cross_fitted_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=2_000, solver="lbfgs")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    values = clipped_logit(probabilities).reshape(-1, 1)
    return cross_val_predict(model, values, labels, cv=cv, method="predict_proba")[:, 1]


def fit_calibrator(labels: np.ndarray, probabilities: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, max_iter=2_000, solver="lbfgs")
    return model.fit(clipped_logit(probabilities).reshape(-1, 1), labels)


def metric_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    bootstrap = []
    for _ in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        bootstrap.append(
            [
                roc_auc_score(labels[indices], probabilities[indices]),
                average_precision_score(labels[indices], probabilities[indices]),
                brier_score_loss(labels[indices], probabilities[indices]),
            ]
        )
    values = np.asarray(bootstrap)
    prevalence = float(labels.mean())
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auroc_95_ci": np.quantile(values[:, 0], [0.025, 0.975]).tolist(),
        "auprc": float(average_precision_score(labels, probabilities)),
        "auprc_95_ci": np.quantile(values[:, 1], [0.025, 0.975]).tolist(),
        "auprc_prevalence_ratio": float(
            average_precision_score(labels, probabilities) / prevalence
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
        "brier_95_ci": np.quantile(values[:, 2], [0.025, 0.975]).tolist(),
    }


def align_baseline(path: Path, cohort: pd.DataFrame, column: str) -> np.ndarray:
    baseline = pd.read_csv(path)
    aligned = cohort[["SEQN"]].merge(
        baseline[["SEQN", column]],
        on="SEQN",
        how="left",
        validate="one_to_one",
    )
    if aligned[column].isna().any():
        raise ValueError(f"Missing {column} baseline predictions after SEQN alignment.")
    return aligned[column].to_numpy(float)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = load_cohort(args.train_table, args.train_labels)
    test = load_cohort(args.test_table, args.test_labels)
    dictionary = pd.read_csv(args.dictionary)
    clinical_names = dictionary.loc[
        (~dictionary["source"].isin({"ACTIVITY_CONCEPTS", "MANTIS_V2"}))
        & dictionary["role"].isin({"core", "extended"}),
        "feature",
    ].tolist()
    concept_names = [
        name for name in train if name.startswith("activity_")
    ]
    rar_names = [
        name
        for name in train
        if name.startswith("rar_") and not name.startswith("rar_reliability_")
    ]
    mantis_names = [name for name in train if name.startswith("mantis_")]
    branch_definitions = {
        "clinical": (
            clinical_names,
            clinical_pipeline(dictionary, clinical_names, args.seed, "lasso"),
        ),
        "activity_concepts": (concept_names, dense_logistic(0.01, args.seed)),
        "classic_rar": (rar_names, dense_logistic(0.1, args.seed)),
        "mantis_v2": (mantis_names, dense_logistic(0.001, args.seed)),
    }
    branch_names = list(branch_definitions)
    if args.variant == "no_rar":
        branch_names.remove("classic_rar")
    elif args.variant == "no_mantis":
        branch_names.remove("mantis_v2")
    use_reliability = args.variant != "no_reliability"
    feature_sets = [branch_definitions[name][0] for name in branch_names]
    models = [branch_definitions[name][1] for name in branch_names]
    y_train = train["depression"].to_numpy(np.int8)
    y_test = test["depression"].to_numpy(np.int8)

    internal_raw = nested_internal_predictions(
        models,
        feature_sets,
        branch_names,
        use_reliability,
        train,
        y_train,
        args.folds,
        args.seed,
        args.jobs,
    )
    internal_probability = cross_fitted_calibration(
        y_train,
        internal_raw,
        args.folds,
        args.seed + 100,
    )
    full_cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    branch_oof, branch_external = branch_predictions(
        models,
        feature_sets,
        train,
        test,
        y_train,
        full_cv,
        args.jobs,
    )
    meta_train = reliability_meta_features(
        branch_oof,
        branch_names,
        use_reliability,
        *reliability_arrays(train),
    )
    meta_test = reliability_meta_features(
        branch_external,
        branch_names,
        use_reliability,
        *reliability_arrays(test),
    )
    meta_search = make_meta_search(full_cv, args.jobs).fit(meta_train, y_train)
    external_raw = meta_search.predict_proba(meta_test)[:, 1]
    calibrator = fit_calibrator(y_train, internal_raw)
    external_probability = calibrator.predict_proba(
        clipped_logit(external_raw).reshape(-1, 1)
    )[:, 1]

    internal_rf = align_baseline(args.internal_baseline, train, "rf_calibrated")
    external_rf = align_baseline(args.external_baseline, test, "rf")
    internal_ensemble = 0.5 * internal_rf + 0.5 * internal_probability
    external_ensemble = 0.5 * external_rf + 0.5 * external_probability
    internal_predictions = pd.DataFrame(
        {
            "SEQN": train["SEQN"].astype(int),
            "label": y_train,
            "core_fuse": internal_probability,
            "rf": internal_rf,
            "rf_core_fuse_ensemble": internal_ensemble,
        }
    )
    external_predictions = pd.DataFrame(
        {
            "SEQN": test["SEQN"].astype(int),
            "label": y_test,
            "core_fuse": external_probability,
            "rf": external_rf,
            "rf_core_fuse_ensemble": external_ensemble,
        }
    )
    internal_predictions.to_csv(
        args.output_dir / "internal_oof_predictions.csv",
        index=False,
    )
    external_predictions.to_csv(
        args.output_dir / "external_predictions.csv",
        index=False,
    )
    internal_metrics = {
        name: metric_interval(
            y_train,
            internal_predictions[name].to_numpy(float),
            args.bootstrap_iterations,
            args.seed + index,
        )
        for index, name in enumerate(["core_fuse", "rf", "rf_core_fuse_ensemble"])
    }
    external_metrics = {
        name: metric_interval(
            y_test,
            external_predictions[name].to_numpy(float),
            args.bootstrap_iterations,
            args.seed + 10 + index,
        )
        for index, name in enumerate(["core_fuse", "rf", "rf_core_fuse_ensemble"])
    }
    report = {
        "model": "CoRe-Fuse",
        "variant": args.variant,
        "branches": branch_names,
        "reliability_conditioning": use_reliability,
        "meta_features": int(meta_train.shape[1]),
        "meta_best_c": float(meta_search.best_params_["C"]),
        "internal_n": len(train),
        "internal_events": int(y_train.sum()),
        "external_n": len(test),
        "external_events": int(y_test.sum()),
        "internal": internal_metrics,
        "external": external_metrics,
        "selection_gate": {
            "required_internal_auprc_absolute_gain": 0.015,
            "required_internal_auprc_relative_gain": 0.05,
            "internal_ensemble_vs_rf_auprc_gain": (
                internal_metrics["rf_core_fuse_ensemble"]["auprc"]
                - internal_metrics["rf"]["auprc"]
            ),
            "external_ensemble_vs_rf_auprc_gain": (
                external_metrics["rf_core_fuse_ensemble"]["auprc"]
                - external_metrics["rf"]["auprc"]
            ),
        },
    }
    (args.output_dir / "core_fuse_results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

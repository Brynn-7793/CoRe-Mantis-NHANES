"""Run leakage-audited bidirectional NHANES cross-cycle model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from run_core_fuse_external import (
    branch_predictions,
    clipped_logit,
    cross_fitted_calibration,
    dense_logistic,
    fit_calibrator,
    make_meta_search,
    metric_interval,
    nested_internal_predictions,
    reliability_arrays,
    reliability_meta_features,
)
from run_uncertainty_stack import clinical_pipeline


STRICT_ACTIVITY_FEATURES = (
    "activity_mean_valid_activity",
    "activity_activity_coefficient_of_variation",
    "activity_between_day_profile_consistency",
    "activity_lag_24h_autocorrelation",
    "activity_valid_wear_fraction",
    "activity_reliability_mean",
    "activity_reliability_minimum",
    "activity_reliability_standard_deviation",
)

MODEL_COLUMNS = (
    "clinical",
    "activity_concepts",
    "classic_rar",
    "mantis_v2",
    "transparent_wearable",
    "random_forest",
    "core_fuse",
    "fixed_ensemble",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--activity-mode",
        choices=("strict", "legacy"),
        default="strict",
        help="Strict excludes activity features whose cut points were fitted in 2013-2014.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--paired-iterations", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def load_cohort(table_path: Path, labels_path: Path) -> pd.DataFrame:
    table = pd.read_pickle(table_path)
    labels = pd.read_csv(labels_path)
    complete = labels.loc[
        labels["phq9_complete"].eq(1), ["SEQN", "depression"]
    ].copy()
    complete["depression"] = complete["depression"].astype(np.int8)
    cohort = complete.merge(table, on="SEQN", how="inner", validate="one_to_one")
    if cohort["SEQN"].duplicated().any():
        raise ValueError("Duplicate participant identifiers after cohort merge.")
    return cohort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dictionary: pd.DataFrame,
    activity_mode: str,
) -> dict[str, list[str]]:
    clinical = dictionary.loc[
        dictionary["role"].isin({"core", "extended"}), "feature"
    ].tolist()
    if activity_mode == "strict":
        activity = list(STRICT_ACTIVITY_FEATURES)
    else:
        activity = [name for name in train if name.startswith("activity_")]
    rar = [
        name
        for name in train
        if name.startswith("rar_") and not name.startswith("rar_reliability_")
    ]
    mantis = [name for name in train if name.startswith("mantis_")]
    feature_sets = {
        "clinical": clinical,
        "activity_concepts": activity,
        "classic_rar": rar,
        "mantis_v2": mantis,
        "transparent_wearable": [*activity, *rar],
    }
    for branch, names in feature_sets.items():
        missing_train = sorted(set(names) - set(train.columns))
        missing_test = sorted(set(names) - set(test.columns))
        if missing_train or missing_test:
            raise ValueError(
                f"Missing {branch} columns: train={missing_train}, test={missing_test}"
            )
    return feature_sets


def branch_models(
    dictionary: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    seed: int,
) -> dict[str, object]:
    return {
        "clinical": clinical_pipeline(
            dictionary,
            feature_sets["clinical"],
            seed,
            "lasso",
        ),
        "activity_concepts": dense_logistic(0.01, seed),
        "classic_rar": dense_logistic(0.1, seed),
        "mantis_v2": dense_logistic(0.001, seed),
        "transparent_wearable": dense_logistic(0.01, seed),
    }


def calibrated_branch_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, list[str]],
    dictionary: pd.DataFrame,
    folds: int,
    seed: int,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = branch_models(dictionary, feature_sets, seed)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    internal = pd.DataFrame({"SEQN": train["SEQN"].astype(int), "label": labels})
    external = pd.DataFrame(
        {
            "SEQN": test["SEQN"].astype(int),
            "label": test["depression"].astype(int),
        }
    )
    for index, name in enumerate(
        ("clinical", "activity_concepts", "classic_rar", "mantis_v2", "transparent_wearable")
    ):
        model = models[name]
        features = feature_sets[name]
        raw_oof = cross_val_predict(
            clone(model),
            train[features],
            labels,
            cv=splitter,
            method="predict_proba",
            n_jobs=jobs,
        )[:, 1]
        internal[name] = cross_fitted_calibration(
            labels,
            raw_oof,
            folds,
            seed + 100 + index,
        )
        fitted = clone(model).fit(train[features], labels)
        external_raw = fitted.predict_proba(test[features])[:, 1]
        calibrator = fit_calibrator(labels, raw_oof)
        external[name] = calibrator.predict_proba(
            clipped_logit(external_raw).reshape(-1, 1)
        )[:, 1]
    return internal, external


def calibrated_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    clinical_features: list[str],
    dictionary: pd.DataFrame,
    folds: int,
    seed: int,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray, object]:
    model = clinical_pipeline(dictionary, clinical_features, seed, "random_forest")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    raw_oof = cross_val_predict(
        clone(model),
        train[clinical_features],
        labels,
        cv=splitter,
        method="predict_proba",
        n_jobs=jobs,
    )[:, 1]
    internal = cross_fitted_calibration(labels, raw_oof, folds, seed + 500)
    fitted = clone(model).fit(train[clinical_features], labels)
    external_raw = fitted.predict_proba(test[clinical_features])[:, 1]
    calibrator = fit_calibrator(labels, raw_oof)
    external = calibrator.predict_proba(
        clipped_logit(external_raw).reshape(-1, 1)
    )[:, 1]
    return internal, external, fitted


def core_fuse_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, list[str]],
    dictionary: pd.DataFrame,
    folds: int,
    seed: int,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    names = ["clinical", "activity_concepts", "classic_rar", "mantis_v2"]
    all_models = branch_models(dictionary, feature_sets, seed)
    models = [all_models[name] for name in names]
    features = [feature_sets[name] for name in names]
    internal_raw = nested_internal_predictions(
        models,
        features,
        names,
        True,
        train,
        labels,
        folds,
        seed,
        jobs,
    )
    internal = cross_fitted_calibration(labels, internal_raw, folds, seed + 100)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    branch_oof, branch_external = branch_predictions(
        models,
        features,
        train,
        test,
        labels,
        splitter,
        jobs,
    )
    meta_train = reliability_meta_features(
        branch_oof,
        names,
        True,
        *reliability_arrays(train),
    )
    meta_test = reliability_meta_features(
        branch_external,
        names,
        True,
        *reliability_arrays(test),
    )
    search = make_meta_search(splitter, jobs).fit(meta_train, labels)
    external_raw = search.predict_proba(meta_test)[:, 1]
    calibrator = fit_calibrator(labels, internal_raw)
    external = calibrator.predict_proba(
        clipped_logit(external_raw).reshape(-1, 1)
    )[:, 1]
    metadata = {
        "branches": names,
        "meta_feature_count": int(meta_train.shape[1]),
        "meta_best_c": float(search.best_params_["C"]),
    }
    return internal, external, metadata


def calibration_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    model = LogisticRegression(C=1e6, max_iter=2_000, solver="lbfgs")
    model.fit(clipped_logit(probabilities).reshape(-1, 1), labels)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def paired_interval(
    labels: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    draws = []
    for _ in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        y = labels[indices]
        first = candidate[indices]
        second = comparator[indices]
        draws.append(
            [
                roc_auc_score(y, first) - roc_auc_score(y, second),
                average_precision_score(y, first) - average_precision_score(y, second),
                brier_score_loss(y, first) - brier_score_loss(y, second),
            ]
        )
    values = np.asarray(draws)
    names = ("auroc", "auprc", "brier")
    estimates = (
        roc_auc_score(labels, candidate) - roc_auc_score(labels, comparator),
        average_precision_score(labels, candidate)
        - average_precision_score(labels, comparator),
        brier_score_loss(labels, candidate) - brier_score_loss(labels, comparator),
    )
    output = {}
    for index, name in enumerate(names):
        finite = values[:, index][np.isfinite(values[:, index])]
        probability = 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0))
        output[name] = {
            "difference": float(estimates[index]),
            "95_ci": np.quantile(finite, [0.025, 0.975]).tolist(),
            "two_sided_probability": float(min(probability, 1.0)),
        }
    return output


def threshold_for_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target: float,
) -> float:
    _, sensitivity, thresholds = roc_curve(labels, probabilities)
    eligible = np.flatnonzero(sensitivity >= target)
    if not eligible.size:
        return 0.0
    return float(np.max(thresholds[eligible]))


def operating_characteristics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "alerts_per_1000": float(1_000.0 * (tp + fp) / len(labels)),
    }


def decision_curve(
    labels: np.ndarray,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    prevalence = float(labels.mean())
    for threshold in np.linspace(0.01, 0.30, 30):
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "model": "treat_all",
                "net_benefit": prevalence - (1.0 - prevalence) * odds,
            }
        )
        rows.append({"threshold": threshold, "model": "treat_none", "net_benefit": 0.0})
        for name in ("clinical", "random_forest", "core_fuse", "fixed_ensemble"):
            predicted = predictions[name].to_numpy(float) >= threshold
            tp = np.sum(predicted & (labels == 1))
            fp = np.sum(predicted & (labels == 0))
            rows.append(
                {
                    "threshold": threshold,
                    "model": name,
                    "net_benefit": float(tp / len(labels) - fp / len(labels) * odds),
                }
            )
    return pd.DataFrame(rows)


def evaluate_direction(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dictionary: pd.DataFrame,
    output_dir: Path,
    activity_mode: str,
    folds: int,
    bootstrap_iterations: int,
    paired_iterations: int,
    jobs: int,
) -> dict[str, object]:
    branch_seed = 20261023
    fusion_seed = 20261021
    feature_sets = select_features(train, test, dictionary, activity_mode)
    labels = train["depression"].to_numpy(np.int8)
    test_labels = test["depression"].to_numpy(np.int8)
    internal, external = calibrated_branch_predictions(
        train,
        test,
        labels,
        feature_sets,
        dictionary,
        folds,
        branch_seed,
        jobs,
    )
    internal_rf, external_rf, fitted_rf = calibrated_random_forest(
        train,
        test,
        labels,
        feature_sets["clinical"],
        dictionary,
        folds,
        fusion_seed,
        jobs,
    )
    internal_core, external_core, core_metadata = core_fuse_predictions(
        train,
        test,
        labels,
        feature_sets,
        dictionary,
        folds,
        fusion_seed,
        jobs,
    )
    internal["random_forest"] = internal_rf
    external["random_forest"] = external_rf
    internal["core_fuse"] = internal_core
    external["core_fuse"] = external_core
    internal["fixed_ensemble"] = 0.5 * (internal_rf + internal_core)
    external["fixed_ensemble"] = 0.5 * (external_rf + external_core)

    direction_dir = output_dir / name
    direction_dir.mkdir(parents=True, exist_ok=True)
    internal.to_csv(direction_dir / "development_oof_predictions.csv", index=False)
    external.to_csv(direction_dir / "external_predictions.csv", index=False)
    joblib.dump(fitted_rf, direction_dir / "random_forest_full_training.joblib")

    report: dict[str, object] = {
        "direction": name,
        "status": "post-freeze exploratory bidirectional sensitivity analysis",
        "activity_mode": activity_mode,
        "development_n": int(len(train)),
        "development_events": int(labels.sum()),
        "evaluation_n": int(len(test)),
        "evaluation_events": int(test_labels.sum()),
        "feature_counts": {key: len(value) for key, value in feature_sets.items()},
        "feature_sets": feature_sets,
        "core_fuse": core_metadata,
        "development": {},
        "evaluation": {},
        "paired_comparisons": {},
        "operating_points": {},
    }
    for index, model in enumerate(MODEL_COLUMNS):
        report["development"][model] = {
            **metric_interval(
                labels,
                internal[model].to_numpy(float),
                bootstrap_iterations,
                20261026 + index,
            ),
            "calibration": calibration_summary(labels, internal[model].to_numpy(float)),
        }
        report["evaluation"][model] = {
            **metric_interval(
                test_labels,
                external[model].to_numpy(float),
                bootstrap_iterations,
                20261126 + index,
            ),
            "calibration": calibration_summary(
                test_labels, external[model].to_numpy(float)
            ),
        }

    for candidate in ("core_fuse", "fixed_ensemble"):
        for comparator in ("clinical", "random_forest"):
            key = f"{candidate}_minus_{comparator}"
            report["paired_comparisons"][key] = {
                "development": paired_interval(
                    labels,
                    internal[candidate].to_numpy(float),
                    internal[comparator].to_numpy(float),
                    paired_iterations,
                    20261226,
                ),
                "evaluation": paired_interval(
                    test_labels,
                    external[candidate].to_numpy(float),
                    external[comparator].to_numpy(float),
                    paired_iterations,
                    20261326,
                ),
            }

    for target in (0.80, 0.90):
        threshold = threshold_for_sensitivity(
            labels,
            internal["fixed_ensemble"].to_numpy(float),
            target,
        )
        report["operating_points"][str(target)] = {
            "development": operating_characteristics(
                labels,
                internal["fixed_ensemble"].to_numpy(float),
                threshold,
            ),
            "evaluation": operating_characteristics(
                test_labels,
                external["fixed_ensemble"].to_numpy(float),
                threshold,
            ),
        }

    decision_curve(labels, internal).assign(cohort="development").to_csv(
        direction_dir / "decision_curve_development.csv", index=False
    )
    decision_curve(test_labels, external).assign(cohort="evaluation").to_csv(
        direction_dir / "decision_curve_evaluation.csv", index=False
    )
    with (direction_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "2011_2012": project
        / "data/processed/bidirectional_preflight/multimodal_2011_2012_augmented.pkl",
        "2013_2014": project
        / "data/processed/bidirectional_preflight/multimodal_2013_2014_augmented.pkl",
    }
    label_paths = {
        "2011_2012": project
        / "data/processed/nhanes_2011_2012_covariates/labels_from_dpq.csv",
        "2013_2014": project
        / "data/processed/nhanes_2013_2014_covariates/labels_from_dpq.csv",
    }
    dictionary_path = project / (
        "data/processed/nhanes_2013_2014_covariates/feature_dictionary.csv"
    )
    cohorts = {
        cycle: load_cohort(table_paths[cycle], label_paths[cycle])
        for cycle in table_paths
    }
    if set(cohorts["2011_2012"]["SEQN"]) & set(cohorts["2013_2014"]["SEQN"]):
        raise ValueError("Participant identifiers overlap across survey cycles.")
    dictionary = pd.read_csv(dictionary_path)
    reports = {
        "reverse_2013_2014_to_2011_2012": evaluate_direction(
            "reverse_2013_2014_to_2011_2012",
            cohorts["2013_2014"],
            cohorts["2011_2012"],
            dictionary,
            output,
            args.activity_mode,
            args.folds,
            args.bootstrap_iterations,
            args.paired_iterations,
            args.jobs,
        ),
        "forward_2011_2012_to_2013_2014": evaluate_direction(
            "forward_2011_2012_to_2013_2014",
            cohorts["2011_2012"],
            cohorts["2013_2014"],
            dictionary,
            output,
            args.activity_mode,
            args.folds,
            args.bootstrap_iterations,
            args.paired_iterations,
            args.jobs,
        ),
    }
    script_path = Path(__file__).resolve()
    manifest = {
        "analysis_status": "post-freeze exploratory bidirectional sensitivity analysis",
        "activity_mode": args.activity_mode,
        "label_rule": "all nine PHQ-9 items valid; total score >=10",
        "cycles": {
            cycle: {
                "participants": int(len(frame)),
                "events": int(frame["depression"].sum()),
            }
            for cycle, frame in cohorts.items()
        },
        "seeds": {
            "branch": 20261023,
            "fusion": 20261021,
            "bootstrap_base": 20261026,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            str(path.relative_to(project)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [*table_paths.values(), *label_paths.values(), dictionary_path]
        },
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
    }
    with (output / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (output / "bidirectional_results.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    print(json.dumps({key: value["evaluation"] for key, value in reports.items()}, indent=2))


if __name__ == "__main__":
    main()

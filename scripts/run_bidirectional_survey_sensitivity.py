"""Survey-weighted sensitivity analysis for bidirectional NHANES predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_submission_gap_outputs import (
    percentile_interval,
    rao_wu_weights,
    two_sided_probability,
    weighted_metrics,
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

CONTRASTS = (
    ("core_fuse", "clinical"),
    ("core_fuse", "random_forest"),
    ("fixed_ensemble", "clinical"),
    ("fixed_ensemble", "random_forest"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261027)
    return parser.parse_args()


def load_evaluation_frame(
    predictions_path: Path,
    covariates_path: Path,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path).rename(columns={"label": "depression"})
    covariates = pd.read_csv(
        covariates_path,
        usecols=["SEQN", "survey_weight", "survey_stratum", "survey_psu"],
    )
    frame = predictions.merge(
        covariates,
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(predictions):
        raise ValueError("Survey-design variables were unavailable for some predictions.")
    required = [
        "depression",
        "survey_weight",
        "survey_stratum",
        "survey_psu",
        *MODEL_COLUMNS,
    ]
    if frame[required].isna().any().any():
        raise ValueError("Missing values detected in survey analysis inputs.")
    return frame


def analyse(
    frames: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    point: dict[tuple[str, str, str], float] = {}
    draws: dict[tuple[str, str, str], list[float]] = {}
    prevalence_point: dict[str, float] = {}
    prevalence_draws: dict[str, list[float]] = {}

    for direction, frame in frames.items():
        labels = frame["depression"].to_numpy(dtype=int)
        weights = frame["survey_weight"].to_numpy(dtype=float)
        prevalence_point[direction] = float(np.average(labels, weights=weights))
        prevalence_draws[direction] = []
        for model in MODEL_COLUMNS:
            values = weighted_metrics(
                labels,
                frame[model].to_numpy(dtype=float),
                weights,
            )
            for metric, estimate in values.items():
                point[(direction, model, metric)] = estimate
                draws[(direction, model, metric)] = []

    for _ in range(replicates):
        for direction, frame in frames.items():
            labels = frame["depression"].to_numpy(dtype=int)
            replicate_weights = rao_wu_weights(frame, rng)
            prevalence_draws[direction].append(
                float(np.average(labels, weights=replicate_weights))
            )
            for model in MODEL_COLUMNS:
                values = weighted_metrics(
                    labels,
                    frame[model].to_numpy(dtype=float),
                    replicate_weights,
                )
                for metric, estimate in values.items():
                    draws[(direction, model, metric)].append(estimate)

    performance_rows: list[dict[str, object]] = []
    for key, estimate in point.items():
        direction, model, metric = key
        values = np.asarray(draws[key], dtype=float)
        lower, upper = percentile_interval(values)
        performance_rows.append(
            {
                "direction": direction,
                "model": model,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_replicates": int(np.isfinite(values).sum()),
            }
        )

    contrast_rows: list[dict[str, object]] = []
    for direction in frames:
        for candidate, comparator in CONTRASTS:
            for metric in ("auroc", "auprc", "brier"):
                candidate_draws = np.asarray(
                    draws[(direction, candidate, metric)],
                    dtype=float,
                )
                comparator_draws = np.asarray(
                    draws[(direction, comparator, metric)],
                    dtype=float,
                )
                differences = candidate_draws - comparator_draws
                lower, upper = percentile_interval(differences)
                contrast_rows.append(
                    {
                        "direction": direction,
                        "contrast": f"{candidate}_minus_{comparator}",
                        "metric": metric,
                        "difference": point[(direction, candidate, metric)]
                        - point[(direction, comparator, metric)],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "two_sided_probability": two_sided_probability(differences),
                        "valid_replicates": int(np.isfinite(differences).sum()),
                    }
                )

    summary = {
        "analysis_status": "post-freeze exploratory sensitivity analysis",
        "method": (
            "Rao-Wu rescaled PSU bootstrap within NHANES masked variance "
            "strata; cycles analysed separately"
        ),
        "replicates": replicates,
        "seed": seed,
        "weight": "WTMEC2YR retained separately for each two-year cycle",
        "directions": {
            direction: {
                "n": int(len(frame)),
                "events": int(frame["depression"].sum()),
                "weighted_prevalence": prevalence_point[direction],
                "weighted_prevalence_95_ci": list(
                    percentile_interval(
                        np.asarray(prevalence_draws[direction], dtype=float)
                    )
                ),
                "strata": int(frame["survey_stratum"].nunique()),
                "psus": int(
                    frame[["survey_stratum", "survey_psu"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
            for direction, frame in frames.items()
        },
        "note": (
            "This is a design-respecting sensitivity analysis, not an "
            "NCHS-supplied replicate-weight analysis."
        ),
    }
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(contrast_rows),
        summary,
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    prediction_root = args.prediction_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frames = {
        "reverse_2013_2014_to_2011_2012": load_evaluation_frame(
            prediction_root
            / "reverse_2013_2014_to_2011_2012/external_predictions.csv",
            project / "data/processed/nhanes_2011_2012_covariates/covariates.csv",
        ),
        "forward_2011_2012_to_2013_2014": load_evaluation_frame(
            prediction_root
            / "forward_2011_2012_to_2013_2014/external_predictions.csv",
            project
            / "data/processed/multimodal_2013_2014/covariates_with_concepts.csv",
        ),
    }
    performance, contrasts, summary = analyse(
        frames,
        replicates=args.replicates,
        seed=args.seed,
    )
    performance.to_csv(
        output / "survey_weighted_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrasts.to_csv(
        output / "survey_weighted_contrasts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output / "survey_weighted_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

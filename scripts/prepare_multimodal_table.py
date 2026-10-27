"""Align NHANES covariates with Mantis embeddings and activity concepts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONCEPT_NAMES = (
    "mean_valid_activity",
    "activity_coefficient_of_variation",
    "low_activity_fraction",
    "high_activity_fraction",
    "activity_state_transition_rate",
    "prolonged_low_activity_fraction",
    "between_day_profile_consistency",
    "lag_24h_autocorrelation",
    "valid_wear_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sequence-ids", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dictionary_rows(names: list[str], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": names,
            "source": source,
            "role": "core",
            "kind": "numeric",
            "description": names,
        }
    )


def save_dictionary(
    base: pd.DataFrame,
    activity: pd.DataFrame,
    mantis: pd.DataFrame,
    output_dir: Path,
    name: str,
    include_clinical: bool,
    include_self_rated_health: bool,
    include_mantis: bool,
) -> None:
    clinical = base.copy()
    if not include_clinical:
        clinical["role"] = "excluded"
    elif not include_self_rated_health:
        clinical.loc[clinical["feature"].eq("self_rated_health"), "role"] = "excluded"
    additions = [activity]
    if include_mantis:
        additions.append(mantis)
    dictionary = pd.concat([clinical, *additions], ignore_index=True)
    dictionary.to_csv(output_dir / f"dictionary_{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_ids = np.load(args.sequence_ids).astype(np.int64)
    embeddings = np.load(args.feature_dir / "mantis_embeddings.npy")
    concepts = np.load(args.feature_dir / "activity_concepts.npy")
    reliabilities = np.load(args.feature_dir / "concept_reliabilities.npy")
    expected_shapes = {
        "embeddings": (len(sequence_ids), 1_024),
        "concepts": (len(sequence_ids), len(CONCEPT_NAMES)),
        "reliabilities": (len(sequence_ids), len(CONCEPT_NAMES)),
    }
    observed_shapes = {
        "embeddings": embeddings.shape,
        "concepts": concepts.shape,
        "reliabilities": reliabilities.shape,
    }
    if observed_shapes != expected_shapes:
        raise ValueError(f"Feature shape mismatch: {observed_shapes} != {expected_shapes}")

    covariates = pd.read_csv(args.covariates)
    cohort = pd.DataFrame({"SEQN": sequence_ids}).merge(
        covariates,
        on="SEQN",
        how="left",
        validate="one_to_one",
    )
    reliable_concepts = concepts * reliabilities
    concept_names = [f"activity_{name}" for name in CONCEPT_NAMES]
    reliability_names = [
        "activity_reliability_mean",
        "activity_reliability_minimum",
        "activity_reliability_standard_deviation",
    ]
    concept_frame = pd.DataFrame(reliable_concepts, columns=concept_names)
    concept_frame[reliability_names[0]] = reliabilities.mean(axis=1)
    concept_frame[reliability_names[1]] = reliabilities.min(axis=1)
    concept_frame[reliability_names[2]] = reliabilities.std(axis=1)
    concept_frame.insert(0, "SEQN", sequence_ids)

    embedding_names = [f"mantis_{index:04d}" for index in range(embeddings.shape[1])]
    embedding_frame = pd.DataFrame(embeddings, columns=embedding_names)
    embedding_frame.insert(0, "SEQN", sequence_ids)
    concept_table = cohort.merge(concept_frame, on="SEQN", validate="one_to_one")
    multimodal_table = concept_table.merge(
        embedding_frame,
        on="SEQN",
        validate="one_to_one",
    )
    concept_table.to_csv(args.output_dir / "covariates_with_concepts.csv", index=False)
    multimodal_table.to_pickle(args.output_dir / "covariates_with_mantis.pkl")

    base_dictionary = pd.read_csv(args.dictionary)
    activity_dictionary = dictionary_rows(
        [*concept_names, *reliability_names],
        "ACTIVITY_CONCEPTS",
    )
    mantis_dictionary = dictionary_rows(embedding_names, "MANTIS_V2")
    configurations = (
        ("activity_concepts", False, False, False),
        ("objective_concepts", True, False, False),
        ("core_concepts", True, True, False),
        ("activity_mantis", False, False, True),
        ("objective_mantis", True, False, True),
        ("core_mantis", True, True, True),
    )
    for name, include_clinical, include_health, include_mantis in configurations:
        save_dictionary(
            base_dictionary,
            activity_dictionary,
            mantis_dictionary,
            args.output_dir,
            name,
            include_clinical,
            include_health,
            include_mantis,
        )
    summary = {
        "participants": int(len(sequence_ids)),
        "clinical_features": int(len(base_dictionary)),
        "activity_features": int(len(activity_dictionary)),
        "mantis_features": int(len(mantis_dictionary)),
        "concept_table": "covariates_with_concepts.csv",
        "multimodal_table": "covariates_with_mantis.pkl",
    }
    with (args.output_dir / "multimodal_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

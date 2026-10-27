"""Extract MantisV2 embeddings and transparent concepts from NHANES arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acticoncept.concepts import (  # noqa: E402
    CONCEPT_NAMES,
    RELIABILITY_NAMES,
    ConceptThresholds,
    extract_concept_reliabilities,
    extract_concepts,
    fit_concept_thresholds,
)
from acticoncept.preprocessing import prepare_mantis_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument(
        "--threshold-metadata",
        type=Path,
        help="Training-cohort feature_metadata.json for leakage-free external use.",
    )
    parser.add_argument("--mantis-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[512, 1_024])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def load_mantis(args: argparse.Namespace) -> torch.nn.Module:
    sys.path.insert(0, str(args.mantis_source / "src"))
    from mantis.architecture import MantisV2

    network = MantisV2(
        return_transf_layer=2,
        output_token="combined",
        device=args.device,
    )
    network = network.from_pretrained("paris-noah/MantisV2")
    network.eval()
    return network


def extract_embeddings(
    network: torch.nn.Module,
    activity: np.ndarray,
    validity: np.ndarray,
    resolutions: list[int],
    batch_size: int,
    device: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, activity.shape[0], batch_size):
        stop = min(start + batch_size, activity.shape[0])
        views = prepare_mantis_views(activity[start:stop], validity[start:stop], resolutions)
        resolution_embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for resolution in resolutions:
                batch = torch.from_numpy(views[resolution]).to(device)
                resolution_embeddings.append(network(batch).cpu().numpy())
        chunks.append(np.concatenate(resolution_embeddings, axis=1).astype(np.float32))
        print(f"Embedded {stop}/{activity.shape[0]} participants", flush=True)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    activity = np.load(args.dataset_dir / "activity_mims.npy", mmap_mode="r")
    validity = np.load(args.dataset_dir / "valid_mask.npy", mmap_mode="r")
    if args.threshold_metadata is not None:
        with args.threshold_metadata.open("r", encoding="utf-8") as handle:
            training_metadata = json.load(handle)
        thresholds = ConceptThresholds(
            low_activity=float(training_metadata["low_activity_threshold"]),
            high_activity=float(training_metadata["high_activity_threshold"]),
        )
        threshold_source = str(args.threshold_metadata)
    elif args.split_file is not None:
        split = np.load(args.split_file)
        thresholds = fit_concept_thresholds(
            activity,
            validity,
            split["train_indices"],
            seed=args.seed,
        )
        threshold_source = str(args.split_file)
    else:
        raise ValueError("Provide --split-file or --threshold-metadata.")
    concepts = extract_concepts(activity, validity, thresholds)
    reliabilities = extract_concept_reliabilities(validity)
    network = load_mantis(args)
    embeddings = extract_embeddings(
        network,
        activity,
        validity,
        args.resolutions,
        args.batch_size,
        args.device,
    )

    np.save(args.output_dir / "mantis_embeddings.npy", embeddings)
    np.save(args.output_dir / "activity_concepts.npy", concepts)
    np.save(args.output_dir / "concept_reliabilities.npy", reliabilities)
    metadata = {
        "n_participants": int(activity.shape[0]),
        "embedding_shape": list(embeddings.shape),
        "concept_names": list(CONCEPT_NAMES),
        "reliability_names": list(RELIABILITY_NAMES),
        "low_activity_threshold": thresholds.low_activity,
        "high_activity_threshold": thresholds.high_activity,
        "concept_threshold_source": threshold_source,
        "resolutions": args.resolutions,
        "mantis_repository": "paris-noah/MantisV2",
        "mantis_return_layer": 2,
        "mantis_output_token": "combined",
    }
    with (args.output_dir / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

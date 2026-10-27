"""Extract classical RAR features and label-free reliability values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acticoncept.rar import (
    RAR_FEATURE_NAMES,
    RAR_RELIABILITY_NAMES,
    extract_rar_features,
    extract_rar_reliabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activity = np.load(args.dataset_dir / "activity_mims.npy", mmap_mode="r")
    validity = np.load(args.dataset_dir / "valid_mask.npy", mmap_mode="r")
    features = extract_rar_features(activity, validity)
    reliabilities = extract_rar_reliabilities(validity)

    np.save(args.output_dir / "rar_features.npy", features)
    np.save(args.output_dir / "rar_reliabilities.npy", reliabilities)
    metadata = {
        "feature_names": list(RAR_FEATURE_NAMES),
        "reliability_names": list(RAR_RELIABILITY_NAMES),
        "feature_shape": list(features.shape),
        "reliability_shape": list(reliabilities.shape),
        "label_free": True,
    }
    with (args.output_dir / "rar_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(
        f"participants={features.shape[0]} features={features.shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()

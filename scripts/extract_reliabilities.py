"""Compute concept-aligned reliability without re-extracting embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acticoncept.concepts import extract_concept_reliabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validity = np.load(args.dataset_dir / "valid_mask.npy", mmap_mode="r")
    reliabilities = extract_concept_reliabilities(validity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "concept_reliabilities.npy"
    np.save(output, reliabilities)
    print(
        f"saved={output} shape={reliabilities.shape} "
        f"mean={reliabilities.mean():.6f} min={reliabilities.min():.6f}"
    )


if __name__ == "__main__":
    main()

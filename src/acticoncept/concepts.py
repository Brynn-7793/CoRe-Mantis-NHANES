"""Transparent activity concepts computed directly from minute-level data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import MINUTES_PER_DAY, MINUTES_PER_WEEK


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

RELIABILITY_NAMES = tuple(f"reliability_{name}" for name in CONCEPT_NAMES)


@dataclass(frozen=True)
class ConceptThresholds:
    """Activity cut points fitted on training participants only."""

    low_activity: float
    high_activity: float


def fit_concept_thresholds(
    activity: np.ndarray,
    validity: np.ndarray,
    train_indices: np.ndarray,
    sample_per_participant: int = 512,
    seed: int = 19,
) -> ConceptThresholds:
    """Estimate robust cohort cut points without validation or test leakage."""
    rng = np.random.default_rng(seed)
    sampled_values: list[np.ndarray] = []
    for index in np.asarray(train_indices, dtype=np.int64):
        values = np.asarray(activity[index], dtype=np.float32)
        support = np.asarray(validity[index], dtype=np.bool_) & np.isfinite(values)
        observed = values[support]
        if observed.size > sample_per_participant:
            observed = rng.choice(observed, size=sample_per_participant, replace=False)
        sampled_values.append(observed)
    pooled = np.concatenate(sampled_values)
    return ConceptThresholds(
        low_activity=float(np.quantile(pooled, 0.25)),
        high_activity=float(np.quantile(pooled, 0.75)),
    )


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return 0.0
    first_valid = first[valid]
    second_valid = second[valid]
    if np.std(first_valid) < 1e-6 or np.std(second_valid) < 1e-6:
        return 0.0
    return float(np.corrcoef(first_valid, second_valid)[0, 1])


def _long_low_fraction(
    low_state: np.ndarray,
    validity: np.ndarray,
    minimum_minutes: int = 30,
) -> float:
    eligible = np.asarray(low_state, dtype=np.bool_) & np.asarray(validity, dtype=np.bool_)
    padded = np.pad(eligible.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    lengths = stops - starts
    return float(lengths[lengths >= minimum_minutes].sum() / max(int(validity.sum()), 1))


def extract_participant_concepts(
    activity: np.ndarray,
    validity: np.ndarray,
    thresholds: ConceptThresholds,
) -> np.ndarray:
    """Compute nine auditable concepts for one seven-day sequence."""
    values = np.asarray(activity, dtype=np.float32)
    support = np.asarray(validity, dtype=np.bool_) & np.isfinite(values)
    if values.size != MINUTES_PER_WEEK or support.shape != values.shape:
        raise ValueError("each participant must have exactly seven days of minute data")
    observed = values[support]
    if observed.size == 0:
        return np.zeros(len(CONCEPT_NAMES), dtype=np.float32)

    mean_activity = float(np.mean(observed))
    coefficient_variation = float(np.std(observed) / max(abs(mean_activity), 1e-6))
    low_observed = observed <= thresholds.low_activity
    high_observed = observed >= thresholds.high_activity
    low_full = values <= thresholds.low_activity
    adjacent_valid = support[1:] & support[:-1]
    state_changes = low_full[1:] != low_full[:-1]
    transition_rate = float(np.mean(state_changes[adjacent_valid])) if adjacent_valid.any() else 0.0

    daily_values = values.reshape(7, MINUTES_PER_DAY)
    daily_support = support.reshape(7, MINUTES_PER_DAY)
    daily_profiles = np.where(daily_support, daily_values, np.nan)
    profile_correlations = [
        _safe_correlation(daily_profiles[first], daily_profiles[second])
        for first in range(7)
        for second in range(first + 1, 7)
    ]
    lag_correlation = _safe_correlation(values[:-MINUTES_PER_DAY], values[MINUTES_PER_DAY:])
    return np.asarray(
        [
            mean_activity,
            coefficient_variation,
            float(np.mean(low_observed)),
            float(np.mean(high_observed)),
            transition_rate,
            _long_low_fraction(low_full, support),
            float(np.mean(profile_correlations)),
            lag_correlation,
            float(np.mean(support)),
        ],
        dtype=np.float32,
    )


def extract_concepts(
    activity: np.ndarray,
    validity: np.ndarray,
    thresholds: ConceptThresholds,
) -> np.ndarray:
    """Compute concept matrix for a participant batch."""
    return np.stack(
        [
            extract_participant_concepts(series, support, thresholds)
            for series, support in zip(activity, validity, strict=True)
        ]
    )


def extract_participant_reliability(validity: np.ndarray) -> np.ndarray:
    """Estimate how much observed support is available for each concept.

    Reliability is label-free and lies in ``[0, 1]``. Point summaries use
    minute coverage, transition summaries use adjacent-minute coverage, and
    rhythm summaries use the temporal pairs required by their definitions.
    The wear-fraction concept is computed from the validity mask itself and is
    therefore assigned unit reliability.
    """
    support = np.asarray(validity, dtype=np.bool_)
    if support.ndim != 1 or support.size != MINUTES_PER_WEEK:
        raise ValueError("validity must contain exactly seven days of minute data")

    minute_coverage = float(np.mean(support))
    adjacent_coverage = float(np.mean(support[:-1] & support[1:]))
    lag_coverage = float(
        np.mean(support[:-MINUTES_PER_DAY] & support[MINUTES_PER_DAY:])
    )
    daily_support = support.reshape(7, MINUTES_PER_DAY)
    pair_coverages = [
        np.mean(daily_support[first] & daily_support[second])
        for first in range(7)
        for second in range(first + 1, 7)
    ]
    between_day_coverage = float(np.mean(pair_coverages))
    return np.asarray(
        [
            minute_coverage,
            minute_coverage,
            minute_coverage,
            minute_coverage,
            adjacent_coverage,
            adjacent_coverage,
            between_day_coverage,
            lag_coverage,
            1.0,
        ],
        dtype=np.float32,
    )


def extract_concept_reliabilities(validity: np.ndarray) -> np.ndarray:
    """Compute a concept-aligned reliability matrix for a participant batch."""
    return np.stack(
        [extract_participant_reliability(support) for support in validity]
    )

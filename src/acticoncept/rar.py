"""Classical, auditable rest-activity rhythm features for weekly actigraphy."""

from __future__ import annotations

import numpy as np

from .preprocessing import MINUTES_PER_DAY, MINUTES_PER_WEEK


RAR_FEATURE_NAMES = (
    "interdaily_stability",
    "intradaily_variability",
    "relative_amplitude",
    "m10_mean_activity",
    "l5_mean_activity",
    "m10_start_sin",
    "m10_start_cos",
    "l5_start_sin",
    "l5_start_cos",
    "cosinor_mesor",
    "cosinor_amplitude",
    "cosinor_acrophase_sin",
    "cosinor_acrophase_cos",
    "valid_wear_fraction",
)

RAR_RELIABILITY_NAMES = tuple(
    f"reliability_{feature_name}" for feature_name in RAR_FEATURE_NAMES
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _clock_profile(values: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    daily_values = values.reshape(7, MINUTES_PER_DAY)
    daily_support = support.reshape(7, MINUTES_PER_DAY)
    counts = daily_support.sum(axis=0)
    sums = np.where(daily_support, daily_values, 0.0).sum(axis=0)
    profile = np.divide(
        sums,
        counts,
        out=np.full(MINUTES_PER_DAY, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return profile, counts


def _fill_circular_profile(profile: np.ndarray) -> np.ndarray:
    observed = np.flatnonzero(np.isfinite(profile))
    if observed.size == 0:
        return np.zeros_like(profile, dtype=np.float64)
    if observed.size == 1:
        return np.full_like(profile, profile[observed[0]], dtype=np.float64)
    positions = np.arange(profile.size)
    extended_positions = np.concatenate(
        [observed - profile.size, observed, observed + profile.size]
    )
    extended_values = np.tile(profile[observed], 3)
    return np.interp(positions, extended_positions, extended_values)


def _extreme_window(profile: np.ndarray, width: int, maximize: bool) -> tuple[float, int]:
    extended = np.concatenate([profile, profile[: width - 1]])
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    means = np.convolve(extended, kernel, mode="valid")
    start = int(np.argmax(means) if maximize else np.argmin(means))
    return float(means[start]), start


def _encode_clock_minute(minute: float) -> tuple[float, float]:
    angle = 2.0 * np.pi * minute / MINUTES_PER_DAY
    return float(np.sin(angle)), float(np.cos(angle))


def _cosinor(values: np.ndarray, support: np.ndarray) -> tuple[float, float, float]:
    positions = np.flatnonzero(support).astype(np.float64)
    observed = values[support].astype(np.float64)
    if observed.size < 3:
        return 0.0, 0.0, 0.0
    angle = 2.0 * np.pi * positions / MINUTES_PER_DAY
    design = np.column_stack(
        [np.ones(observed.size), np.cos(angle), np.sin(angle)]
    )
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    mesor = float(coefficients[0])
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    acrophase = float(np.arctan2(coefficients[2], coefficients[1]))
    return mesor, amplitude, acrophase


def extract_participant_rar(
    activity: np.ndarray,
    validity: np.ndarray,
) -> np.ndarray:
    """Compute classical RAR and cosinor features from one seven-day record."""
    values = np.asarray(activity, dtype=np.float64)
    support = np.asarray(validity, dtype=np.bool_) & np.isfinite(values)
    if values.ndim != 1 or values.size != MINUTES_PER_WEEK:
        raise ValueError("activity must contain exactly seven days of minute data")
    if support.shape != values.shape:
        raise ValueError("validity must align with activity")

    wear_fraction = float(np.mean(support))
    if support.sum() < 3:
        output = np.zeros(len(RAR_FEATURE_NAMES), dtype=np.float32)
        output[-1] = wear_fraction
        return output

    observed = values[support]
    variance = float(np.var(observed))
    profile, _ = _clock_profile(values, support)
    filled_profile = _fill_circular_profile(profile)

    profile_variance = float(np.var(filled_profile))
    interdaily_stability = float(
        np.clip(_safe_ratio(profile_variance, variance), 0.0, 1.0)
    )
    adjacent = support[:-1] & support[1:]
    squared_changes = np.square(values[1:] - values[:-1])
    mean_squared_change = float(np.mean(squared_changes[adjacent])) if adjacent.any() else 0.0
    intradaily_variability = max(_safe_ratio(mean_squared_change, variance), 0.0)

    m10, m10_start = _extreme_window(filled_profile, 10 * 60, maximize=True)
    l5, l5_start = _extreme_window(filled_profile, 5 * 60, maximize=False)
    relative_amplitude = float(np.clip(_safe_ratio(m10 - l5, m10 + l5), -1.0, 1.0))
    m10_sin, m10_cos = _encode_clock_minute(m10_start)
    l5_sin, l5_cos = _encode_clock_minute(l5_start)

    mesor, amplitude, acrophase = _cosinor(values, support)
    acrophase_sin = float(np.sin(acrophase))
    acrophase_cos = float(np.cos(acrophase))
    output = np.asarray(
        [
            interdaily_stability,
            intradaily_variability,
            relative_amplitude,
            m10,
            l5,
            m10_sin,
            m10_cos,
            l5_sin,
            l5_cos,
            mesor,
            amplitude,
            acrophase_sin,
            acrophase_cos,
            wear_fraction,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(output, copy=False)


def extract_rar_features(activity: np.ndarray, validity: np.ndarray) -> np.ndarray:
    """Compute RAR features for an aligned participant matrix."""
    activity_array = np.asarray(activity)
    validity_array = np.asarray(validity)
    if activity_array.ndim != 2 or activity_array.shape != validity_array.shape:
        raise ValueError("activity and validity must be aligned two-dimensional arrays")
    return np.stack(
        [
            extract_participant_rar(series, support)
            for series, support in zip(activity_array, validity_array, strict=True)
        ]
    )


def extract_rar_reliabilities(validity: np.ndarray) -> np.ndarray:
    """Return label-free support measures aligned with each RAR feature."""
    validity_array = np.asarray(validity, dtype=np.bool_)
    if validity_array.ndim != 2 or validity_array.shape[1] != MINUTES_PER_WEEK:
        raise ValueError("validity must have shape (participants, 10080)")

    outputs: list[np.ndarray] = []
    for support in validity_array:
        minute_coverage = float(np.mean(support))
        adjacent_coverage = float(np.mean(support[:-1] & support[1:]))
        daily_support = support.reshape(7, MINUTES_PER_DAY)
        clock_counts = daily_support.sum(axis=0)
        profile_support = float(np.mean(np.clip(clock_counts / 3.0, 0.0, 1.0)))
        pair_coverages = [
            np.mean(daily_support[first] & daily_support[second])
            for first in range(7)
            for second in range(first + 1, 7)
        ]
        between_day_support = float(np.mean(pair_coverages))
        outputs.append(
            np.asarray(
                [
                    between_day_support,
                    adjacent_coverage,
                    profile_support,
                    profile_support,
                    profile_support,
                    profile_support,
                    profile_support,
                    profile_support,
                    profile_support,
                    minute_coverage,
                    minute_coverage,
                    minute_coverage,
                    minute_coverage,
                    1.0,
                ],
                dtype=np.float32,
            )
        )
    return np.stack(outputs)

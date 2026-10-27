"""Leakage-safe preparation of weekly activity for MantisV2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


MINUTES_PER_DAY = 1_440
MINUTES_PER_WEEK = 7 * MINUTES_PER_DAY


def interpolate_valid_minutes(activity: np.ndarray, validity: np.ndarray) -> np.ndarray:
    """Linearly fill invalid minutes without using another participant's data."""
    values = np.asarray(activity, dtype=np.float32)
    support = np.asarray(validity, dtype=np.bool_) & np.isfinite(values)
    if values.ndim != 1 or support.shape != values.shape:
        raise ValueError("activity and validity must be aligned one-dimensional arrays")

    observed = np.flatnonzero(support)
    if observed.size == 0:
        return np.zeros_like(values)
    if observed.size == 1:
        return np.full_like(values, values[observed[0]])

    positions = np.arange(values.size)
    return np.interp(positions, observed, values[observed]).astype(np.float32)


def resample_series(series: np.ndarray, output_length: int) -> np.ndarray:
    """Resample a one-dimensional series to a fixed Mantis-compatible length."""
    if output_length <= 0 or output_length % 32:
        raise ValueError("output_length must be a positive multiple of 32")
    source = np.asarray(series, dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, source.size, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, output_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, source).astype(np.float32)


def prepare_mantis_views(
    activity: np.ndarray,
    validity: np.ndarray,
    resolutions: Sequence[int] = (512, 1_024),
) -> dict[int, np.ndarray]:
    """Create full-week interpolation views; no label information is used."""
    activity_array = np.asarray(activity, dtype=np.float32)
    validity_array = np.asarray(validity, dtype=np.bool_)
    if activity_array.ndim != 2 or activity_array.shape != validity_array.shape:
        raise ValueError("activity and validity must be aligned two-dimensional arrays")

    outputs = {
        resolution: np.empty(
            (activity_array.shape[0], 1, resolution),
            dtype=np.float32,
        )
        for resolution in resolutions
    }
    for row_index, (series, support) in enumerate(zip(activity_array, validity_array, strict=True)):
        filled = interpolate_valid_minutes(series, support)
        for resolution in resolutions:
            outputs[resolution][row_index, 0] = resample_series(filled, resolution)
    return outputs


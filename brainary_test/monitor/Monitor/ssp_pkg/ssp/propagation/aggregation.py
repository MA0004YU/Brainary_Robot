"""Module: Aggregation functions for risk propagation | Paper section: §3.2 | Status: wip"""

import numpy as np
from numpy.typing import NDArray


def noisy_or(values: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Noisy-OR aggregation: p = 1 - prod(1 - p_i).

    Treats each input as an independent probability of activation.
    """
    if not values:
        return np.zeros(values[0].shape if values else 0, dtype=np.float64)
    result = np.ones_like(values[0])
    for v in values:
        result *= 1.0 - v
    return 1.0 - result


def max_aggregation(values: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Element-wise max aggregation for severity."""
    if not values:
        return np.zeros(values[0].shape if values else 0, dtype=np.float64)
    stacked = np.stack(values)
    return np.max(stacked, axis=0)  # type: ignore[no-any-return]


def grouped_noisy_or(
    values: list[NDArray[np.floating]],
    group_keys: list[tuple[str, str, str]],
) -> NDArray[np.floating]:
    """Noisy-OR with deduplication: max within group, noisy-OR across groups.

    Groups are defined by (re_type, hazard_id, target_id) triples.
    Within each group, take element-wise max (avoid double-counting correlated paths).
    Across groups, apply noisy-OR (independent risk sources).
    """
    if not values:
        return np.zeros(values[0].shape if values else 0, dtype=np.float64)

    groups: dict[tuple[str, str, str], NDArray[np.floating]] = {}
    for val, key in zip(values, group_keys, strict=True):
        if key in groups:
            groups[key] = np.maximum(groups[key], val)
        else:
            groups[key] = val.copy()

    return noisy_or(list(groups.values()))

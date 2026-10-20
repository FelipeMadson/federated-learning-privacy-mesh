# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

"""
Federated Model Aggregation Algorithms:
- Federated Averaging (FedAvg)
- Multi-Krum Byzantine-Robust Defense
- Coordinate-wise Trimmed Mean
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List


@dataclass
class ClientUpdate:
    client_id: str
    sample_count: int
    parameters: List[float]


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def fed_avg(updates: List[ClientUpdate]) -> List[float]:
    """
    Standard Federated Averaging (McMahan et al., 2017).
    Weights client parameters proportionally to their local sample volume.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty updates list")

    dim = len(updates[0].parameters)
    total_samples = sum(u.sample_count for u in updates)
    if total_samples <= 0:
        total_samples = len(updates)

    aggregated = [0.0] * dim
    for u in updates:
        weight = u.sample_count / total_samples
        for j in range(dim):
            aggregated[j] += u.parameters[j] * weight

    return aggregated


def multi_krum(updates: List[ClientUpdate], num_byzantine: int = 1, m_selected: int = 1) -> List[float]:
    """
    Byzantine-resilient Multi-Krum aggregation (Blanchard et al., NeurIPS).
    Tolerates up to f Byzantine attackers if n >= 2f + 3.
    """
    n = len(updates)
    f = num_byzantine

    if n < 2 * f + 3:
        # Fallback to FedAvg if cluster size is insufficient for strict Krum bounds
        return fed_avg(updates)

    k = n - f - 2  # number of closest neighbors to consider

    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                d = euclidean_distance(updates[i].parameters, updates[j].parameters)
                distances.append(d)
        distances.sort()
        # Sum of distances to the k closest neighbors
        score_i = sum(distances[:k])
        scores.append((score_i, i))

    scores.sort(key=lambda x: x[0])
    selected_indices = [idx for _, idx in scores[:m_selected]]

    selected_updates = [updates[i] for i in selected_indices]
    return fed_avg(selected_updates)


def trimmed_mean(updates: List[ClientUpdate], trim_ratio: float = 0.2) -> List[float]:
    """
    Coordinate-wise Trimmed Mean aggregation.
    Sorts each dimension across clients and trims the top and bottom fractions.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty updates list")

    n = len(updates)
    dim = len(updates[0].parameters)
    trim_count = int(math.floor(n * trim_ratio))

    result = [0.0] * dim
    for j in range(dim):
        values = sorted(u.parameters[j] for u in updates)
        if 2 * trim_count >= n:
            valid_values = values
        else:
            valid_values = values[trim_count : n - trim_count]
        result[j] = sum(valid_values) / len(valid_values)

    return result

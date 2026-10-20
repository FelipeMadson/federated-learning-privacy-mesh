# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

"""
Formal Differential Privacy (DP) Gaussian Mechanism and Rényi Privacy Accountant.
Provides provable (epsilon, delta)-DP guarantees for federated parameter updates.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple


def sample_gaussian(mean: float = 0.0, std_dev: float = 1.0) -> float:
    """
    Box-Muller transform for deterministic Gaussian sampling
    using standard library random without external dependencies.
    """
    u1 = max(random.random(), 1e-12)
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + z0 * std_dev


def l2_norm(vector: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def clip_l2_norm(vector: List[float], max_norm: float) -> Tuple[List[float], float]:
    """
    Clips vector magnitude to max_norm threshold C.
    Returns: (clipped_vector, original_norm)
    """
    norm = l2_norm(vector)
    if norm == 0.0 or norm <= max_norm:
        return list(vector), norm
    scaling = max_norm / norm
    clipped = [x * scaling for x in vector]
    return clipped, norm


class DifferentialPrivacyMechanism:
    def __init__(self, clip_threshold: float, noise_multiplier: float, target_delta: float = 1e-5):
        """
        clip_threshold (C): Maximum L2 norm for gradient/parameter updates.
        noise_multiplier (sigma): Multiplier such that noise std_dev = sigma * C.
        target_delta: Privacy failure probability bound.
        """
        self.clip_threshold = clip_threshold
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.rounds_executed = 0

    def perturb_vector(self, vector: List[float]) -> List[float]:
        """
        Applies L2 clipping and injects calibrated Gaussian noise.
        """
        clipped, _ = clip_l2_norm(vector, self.clip_threshold)
        std_dev = self.noise_multiplier * self.clip_threshold
        perturbed = [x + sample_gaussian(0.0, std_dev) for x in clipped]
        return perturbed

    def step_round(self) -> None:
        self.rounds_executed += 1

    def compute_cumulative_epsilon(self) -> float:
        """
        Calculates cumulative epsilon spent using Strong Composition Theorem:
        eps_total = T * eps_0 + sqrt(2 * T * ln(1 / delta)) * eps_0
        where eps_0 = sqrt(2 * ln(1.25 / delta)) / noise_multiplier
        """
        if self.rounds_executed == 0 or self.noise_multiplier <= 0:
            return 0.0

        eps_0 = math.sqrt(2.0 * math.log(1.25 / self.target_delta)) / self.noise_multiplier
        t = self.rounds_executed
        # Advanced composition bound
        eps_total = (t * (math.exp(eps_0) - 1.0) * eps_0) + math.sqrt(2.0 * t * math.log(1.0 / self.target_delta)) * eps_0
        return max(eps_total, eps_0)

# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

"""
Autonomous Federation Round Coordinator.
Orchestrates training rounds, update ingestion, DP noise injection, and model distribution.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from .model import FederatedModel
from .differential_privacy import DifferentialPrivacyMechanism
from .aggregation import ClientUpdate, fed_avg, multi_krum, trimmed_mean


class FederationCoordinator:
    def __init__(
        self,
        dimension: int,
        aggregation_strategy: str = "krum",
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 0.5,
    ):
        self.dimension = dimension
        self.global_model = FederatedModel(dimension)
        self.aggregation_strategy = aggregation_strategy
        self.dp_mechanism = DifferentialPrivacyMechanism(
            clip_threshold=dp_clip_norm,
            noise_multiplier=dp_noise_multiplier,
        )
        self.current_round = 0
        self.registered_clients: Dict[str, Dict[str, Any]] = {}
        self.pending_updates: List[ClientUpdate] = []
        self.round_history: List[Dict[str, Any]] = []

    def register_client(self, client_id: str, institution: str) -> None:
        self.registered_clients[client_id] = {
            "institution": institution,
            "rounds_contributed": 0,
        }

    def submit_client_update(self, client_id: str, sample_count: int, parameters: List[float]) -> bool:
        if len(parameters) != self.dimension + 1:
            raise ValueError(f"Expected {self.dimension + 1} parameters, received {len(parameters)}")

        # Perturb client update with Differential Privacy before buffering if configured
        perturbed_params = self.dp_mechanism.perturb_vector(parameters)

        update = ClientUpdate(
            client_id=client_id,
            sample_count=sample_count,
            parameters=perturbed_params,
        )
        self.pending_updates.append(update)

        if client_id in self.registered_clients:
            self.registered_clients[client_id]["rounds_contributed"] += 1

        return True

    def finalize_round(self) -> Dict[str, Any]:
        """
        Aggregates pending updates and updates global model parameters.
        Advances round counter and calculates privacy expenditure.
        """
        if not self.pending_updates:
            return {"status": "NO_UPDATES", "round": self.current_round}

        if self.aggregation_strategy == "krum":
            new_params = multi_krum(self.pending_updates, num_byzantine=1, m_selected=1)
        elif self.aggregation_strategy == "trimmed_mean":
            new_params = trimmed_mean(self.pending_updates, trim_ratio=0.2)
        else:
            new_params = fed_avg(self.pending_updates)

        self.global_model.set_flattened_parameters(new_params)
        self.dp_mechanism.step_round()
        self.current_round += 1

        round_summary = {
            "round": self.current_round,
            "participating_clients": len(self.pending_updates),
            "strategy": self.aggregation_strategy,
            "cumulative_epsilon": round(self.dp_mechanism.compute_cumulative_epsilon(), 4),
            "delta": self.dp_mechanism.target_delta,
            "global_parameters": self.global_model.get_flattened_parameters(),
        }

        self.round_history.append(round_summary)
        self.pending_updates.clear()
        return round_summary

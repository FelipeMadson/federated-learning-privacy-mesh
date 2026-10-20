# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

import unittest
from mesh.model import FederatedModel
from mesh.aggregation import ClientUpdate, fed_avg, multi_krum, trimmed_mean
from mesh.coordinator import FederationCoordinator


class TestFederationAggregation(unittest.TestCase):
    def test_federated_model_gradients_and_update(self):
        model = FederatedModel(dimension=2)
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [1, 0, 1]

        w_grads, b_grad, loss = model.compute_gradients(X, y)
        self.assertEqual(len(w_grads), 2)
        self.assertGreater(loss, 0.0)

        model.apply_gradients(w_grads, b_grad, lr=0.1)
        params = model.get_flattened_parameters()
        self.assertEqual(len(params), 3)

    def test_fed_avg_weighting(self):
        u1 = ClientUpdate(client_id="node-1", sample_count=100, parameters=[1.0, 2.0])
        u2 = ClientUpdate(client_id="node-2", sample_count=300, parameters=[3.0, 6.0])

        # Expected: 0.25 * [1, 2] + 0.75 * [3, 6] = [0.25 + 2.25, 0.5 + 4.5] = [2.5, 5.0]
        agg = fed_avg([u1, u2])
        self.assertAlmostEqual(agg[0], 2.5)
        self.assertAlmostEqual(agg[1], 5.0)

    def test_byzantine_resilient_krum_defense(self):
        # 5 honest nodes near [10.0, 10.0]
        honest_updates = [
            ClientUpdate(client_id=f"honest-{i}", sample_count=100, parameters=[10.0 + i * 0.1, 10.0 - i * 0.1])
            for i in range(5)
        ]
        # 1 malicious attacker submitting poisoned weights [9999.0, -9999.0]
        adversary = ClientUpdate(client_id="malicious-1", sample_count=100, parameters=[9999.0, -9999.0])

        all_updates = honest_updates + [adversary]
        # Multi-Krum should reject the poisoned outlier
        selected = multi_krum(all_updates, num_byzantine=1, m_selected=1)

        self.assertLess(selected[0], 20.0)
        self.assertGreater(selected[0], 5.0)
        self.assertLess(selected[1], 20.0)
        self.assertGreater(selected[1], 5.0)

    def test_federation_coordinator_full_cycle(self):
        coordinator = FederationCoordinator(dimension=2, aggregation_strategy="fedavg")
        coordinator.register_client("hospital-rnp-1", "Hospital Universitário Darcy Ribeiro")
        coordinator.register_client("hospital-rnp-2", "Hospital das Clínicas UFMG")

        coordinator.submit_client_update("hospital-rnp-1", 500, [0.5, -0.2, 0.1])
        coordinator.submit_client_update("hospital-rnp-2", 500, [0.7, -0.4, 0.3])

        summary = coordinator.finalize_round()
        self.assertEqual(summary["round"], 1)
        self.assertEqual(summary["participating_clients"], 2)
        self.assertGreater(summary["cumulative_epsilon"], 0.0)


if __name__ == "__main__":
    unittest.main()

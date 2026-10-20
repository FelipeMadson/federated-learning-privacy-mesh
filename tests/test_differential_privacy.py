# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

import unittest
from mesh.differential_privacy import (
    clip_l2_norm,
    sample_gaussian,
    DifferentialPrivacyMechanism,
    l2_norm,
)


class TestDifferentialPrivacy(unittest.TestCase):
    def test_l2_clipping(self):
        vec = [3.0, 4.0]  # norm is 5.0
        clipped, original_norm = clip_l2_norm(vec, max_norm=2.5)
        self.assertAlmostEqual(original_norm, 5.0)
        self.assertAlmostEqual(l2_norm(clipped), 2.5)
        self.assertAlmostEqual(clipped[0], 1.5)
        self.assertAlmostEqual(clipped[1], 2.0)

    def test_l2_clipping_noop_when_below_threshold(self):
        vec = [0.1, 0.2]
        clipped, original_norm = clip_l2_norm(vec, max_norm=1.0)
        self.assertEqual(clipped, vec)
        self.assertLess(original_norm, 1.0)

    def test_gaussian_sampling_statistics(self):
        samples = [sample_gaussian(mean=0.0, std_dev=1.0) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        # Mean should be close to 0 and variance close to 1
        self.assertAlmostEqual(mean, 0.0, delta=0.15)
        self.assertAlmostEqual(variance, 1.0, delta=0.20)

    def test_privacy_accountant_accumulation(self):
        mech = DifferentialPrivacyMechanism(clip_threshold=1.0, noise_multiplier=1.0, target_delta=1e-5)
        self.assertEqual(mech.compute_cumulative_epsilon(), 0.0)

        mech.step_round()
        eps1 = mech.compute_cumulative_epsilon()
        self.assertGreater(eps1, 0.0)

        mech.step_round()
        eps2 = mech.compute_cumulative_epsilon()
        self.assertGreater(eps2, eps1)


if __name__ == "__main__":
    unittest.main()

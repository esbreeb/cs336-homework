"""Checks for the toy-model metrics. These lock the definitions used in the go/no-go."""

import unittest

import torch

from toy_superposition import (
    Decision,
    feature_dimensionality,
    lag_index,
    decide,
)


class FeatureDimensionalityTest(unittest.TestCase):
    def test_dedicated_basis(self):
        eye = torch.eye(4)
        dimensionality, norms = feature_dimensionality(eye)
        self.assertTrue(torch.allclose(dimensionality, torch.ones(4), atol=1e-5))
        self.assertTrue(torch.allclose(norms, torch.ones(4), atol=1e-5))

    def test_antipodal_pair_is_half(self):
        weights = torch.tensor([[1.0, -1.0]])
        dimensionality, _ = feature_dimensionality(weights)
        self.assertTrue(torch.allclose(dimensionality, torch.tensor([0.5, 0.5]), atol=1e-5))

    def test_unlearned_feature_is_zero(self):
        weights = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        dimensionality, _ = feature_dimensionality(weights)
        self.assertAlmostEqual(float(dimensionality[0]), 1.0, places=5)
        self.assertEqual(float(dimensionality[1]), 0.0)

    def test_dimensions_sum_to_hidden_when_packed(self):
        # Two antipodal pairs in orthogonal directions: 4 features, 2 dimensions.
        weights = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        )
        dimensionality, _ = feature_dimensionality(weights)
        self.assertTrue(torch.allclose(dimensionality, torch.full((4,), 0.5), atol=1e-5))
        self.assertAlmostEqual(float(dimensionality.sum()), 2.0, places=4)


class LagAndDecisionTest(unittest.TestCase):
    def test_lag_index_sign_and_scale(self):
        self.assertAlmostEqual(lag_index(1.0, 0.0), 1.0, places=5)
        self.assertAlmostEqual(lag_index(0.5, 0.5), 0.0, places=5)
        self.assertAlmostEqual(lag_index(0.75, 0.25), 0.5, places=5)

    def test_washout_is_no_go(self):
        decision = decide(
            effect_ab=0.02,
            effect_ba=0.01,
            gap_ab=0.02,
            gap_ba=0.01,
            sigma_lag=0.03,
            sigma_gap=0.02,
            mean_abs_lag_sim=0.02,
            phase1_dedicated=0.95,
        )
        self.assertEqual(decision.outcome, "no_go")

    def test_both_orders_clear_the_noise_floor(self):
        decision = decide(
            effect_ab=0.55,
            effect_ba=0.48,
            gap_ab=0.40,
            gap_ba=0.36,
            sigma_lag=0.02,
            sigma_gap=0.02,
            mean_abs_lag_sim=0.03,
            phase1_dedicated=0.92,
        )
        self.assertEqual(decision.outcome, "go")
        self.assertGreaterEqual(decision.threshold_lag, 0.15)

    def test_one_order_only_is_not_a_go(self):
        decision = decide(
            effect_ab=0.60,
            effect_ba=0.02,
            gap_ab=0.40,
            gap_ba=0.01,
            sigma_lag=0.02,
            sigma_gap=0.02,
            mean_abs_lag_sim=0.02,
            phase1_dedicated=0.9,
        )
        self.assertEqual(decision.outcome, "no_go")

    def test_phase1_failure_is_invalid(self):
        decision = decide(
            effect_ab=0.0,
            effect_ba=0.0,
            gap_ab=0.0,
            gap_ba=0.0,
            sigma_lag=0.01,
            sigma_gap=0.01,
            mean_abs_lag_sim=0.0,
            phase1_dedicated=0.2,
        )
        self.assertEqual(decision.outcome, "invalid")

    def test_decision_fields_are_recorded(self):
        decision = decide(
            effect_ab=0.4,
            effect_ba=0.4,
            gap_ab=0.3,
            gap_ba=0.3,
            sigma_lag=0.01,
            sigma_gap=0.01,
            mean_abs_lag_sim=0.01,
            phase1_dedicated=0.9,
        )
        self.assertIsInstance(decision, Decision)
        self.assertIn("threshold_gap", decision.as_dict())


if __name__ == "__main__":
    unittest.main()

"""Locks the H3 grouping and the progressive-distillation comparison."""

import unittest

from toy_efficiency import (
    acquisition_order,
    cluster_waves,
    decide_efficiency,
    interquartile_range,
    wave_separation,
)


class AcquisitionOrderTest(unittest.TestCase):
    def test_earliest_feature_is_first(self):
        curve = [
            (0, [1.0, 1.0, 1.0, 1.0]),
            (100, [0.1, 1.0, 0.4, 1.0]),
            (200, [0.05, 0.2, 0.05, 0.9]),
        ]
        order = acquisition_order(curve, acquired_below=0.3)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[-1], 3)

    def test_waves_follow_the_large_gap(self):
        acquired = [200] * 16 + [300] * 8 + [800, 800, 1000, 1100, 1200, 1200, 1400, 1400]
        waves = cluster_waves(acquired, min_gap=400)
        self.assertEqual(len(waves), 2)
        self.assertEqual(len(waves[0]), 24)
        self.assertEqual(sorted(waves[1]), [24, 25, 26, 27, 28, 29, 30, 31])
        self.assertGreaterEqual(wave_separation(waves, acquired), 400)

    def test_iqr(self):
        self.assertEqual(interquartile_range([100, 200, 300, 400]), 150)


class EfficiencyDecisionTest(unittest.TestCase):
    def test_clear_token_saving_is_go(self):
        decision = decide_efficiency(
            steps_dev=[800, 900, 700, 800, 850, 750, 900, 800],
            steps_prog=[1600, 1700, 1500, 1800, 1600, 1700, 1500, 1650],
            cap=4000,
        )
        self.assertEqual(decision.outcome, "go")

    def test_tie_is_no_go(self):
        steps = [1000, 1100, 900, 1200, 1000, 1050, 950, 1000]
        decision = decide_efficiency(steps_dev=steps, steps_prog=steps, cap=4000)
        self.assertEqual(decision.outcome, "no_go")

    def test_mass_censoring_is_inconclusive(self):
        decision = decide_efficiency(
            steps_dev=[4000, 4000, 4000, 4000, 2000, 4000, 4000, 4000],
            steps_prog=[4000, 4000, 4000, 4000, 4000, 4000, 3000, 4000],
            cap=4000,
        )
        self.assertEqual(decision.outcome, "inconclusive")


if __name__ == "__main__":
    unittest.main()

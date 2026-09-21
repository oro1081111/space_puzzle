"""Regression coverage for aggregated opponents and independent frozen planners."""
import unittest
import numpy as np
from training.engine import Game
from training.verify import Oracle


class ThreePlayerTests(unittest.TestCase):
    def test_observation(self):
        for count in (2, 3, 4):
            game = Game(count=count, size=15)
            for seat in range(count):
                obs = game.observation(seat)
                np.testing.assert_array_equal(obs[0], game.board == seat)
                np.testing.assert_array_equal(obs[1], (game.board >= 0) & (game.board != seat))
                self.assertEqual(obs[3].sum(), 1)
                self.assertEqual(obs[4].sum(), count - 1)

    def test_frozen_match_matches_existing_controller(self):
        oracle = Oracle()
        try:
            for seat in range(3):
                prior = [[.1, .2, .3, .4]] * 3
                oracle.call(op='reset', size=15, count=3, seed=71, styles=['atlas'] * 3)
                expected = oracle.call(op='atlas', logits=prior)
                oracle.call(op='reset', size=15, count=3, seed=71, styles=['atlas'] * 3)
                actual = oracle.call(op='match3', logits=prior, opponent_logits=prior, seat=seat)
                self.assertEqual(actual, expected)
        finally:
            oracle.close()


if __name__ == '__main__':
    unittest.main()

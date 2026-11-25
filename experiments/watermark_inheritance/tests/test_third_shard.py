import unittest

from experiments.watermark_inheritance.generate import double_prime_seed, sampling_seed


class NullControlTests(unittest.TestCase):
    def test_double_prime_seed_is_deterministic_and_fresh(self):
        config = {"seed": 1, "shard_seed": 7}
        value = double_prime_seed(config, "pair_00001")
        self.assertEqual(value, double_prime_seed(config, "pair_00001"))
        self.assertNotEqual(value, sampling_seed(7, "full", "pair_00001", 0))
        self.assertNotEqual(value, sampling_seed(7, "full", "pair_00001", 1))

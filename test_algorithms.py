"""Tests for the recursive and memoized Fibonacci implementations."""

import unittest

from reference_algorithms import (
    fibonacci_baseline,
    fibonacci_memoized,
)

KNOWN_VALUES = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    5: 5,
    10: 55,
    20: 6765,
    25: 75025,
    30: 832040,
    35: 9227465,
}


class FibonacciAlgorithmTests(unittest.TestCase):
    def test_known_answers_are_independently_checked(self) -> None:
        for n, expected in KNOWN_VALUES.items():
            with self.subTest(n=n, algorithm="baseline"):
                self.assertEqual(fibonacci_baseline(n), expected)
            with self.subTest(n=n, algorithm="memoized"):
                self.assertEqual(fibonacci_memoized(n, {}), expected)

    def test_negative_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than or equal to zero"):
            fibonacci_baseline(-1)
        with self.assertRaisesRegex(ValueError, "greater than or equal to zero"):
            fibonacci_memoized(-1, {})

    def test_non_integer_and_boolean_inputs_are_rejected(self) -> None:
        for invalid in (3.0, "3", True):
            with self.subTest(
                value=invalid, algorithm="baseline"
            ), self.assertRaisesRegex(TypeError, "non-boolean integer"):
                fibonacci_baseline(invalid)  # type: ignore[arg-type]
            with self.subTest(
                value=invalid, algorithm="memoized"
            ), self.assertRaisesRegex(TypeError, "non-boolean integer"):
                fibonacci_memoized(invalid, {})  # type: ignore[arg-type]

    def test_non_mapping_cache_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "mutable mapping"):
            fibonacci_memoized(5, [])  # type: ignore[arg-type]

    def test_fresh_call_records_the_complete_subproblem_chain(self) -> None:
        cache: dict[int, int] = {}
        self.assertEqual(fibonacci_memoized(10, cache), 55)
        self.assertEqual(set(cache), set(range(11)))
        self.assertEqual(cache[0], 0)
        self.assertEqual(cache[1], 1)
        self.assertEqual(cache[10], 55)

    def test_shared_cache_is_reused_without_recomputing_prior_keys(self) -> None:
        cache: dict[int, int] = {}
        self.assertEqual(fibonacci_memoized(10, cache), 55)
        snapshot = dict(cache)

        self.assertEqual(fibonacci_memoized(11, cache), 89)
        self.assertEqual({key: cache[key] for key in snapshot}, snapshot)
        self.assertEqual(set(cache) - set(snapshot), {11})

    def test_separate_caches_do_not_share_state(self) -> None:
        first: dict[int, int] = {}
        second: dict[int, int] = {}

        self.assertEqual(fibonacci_memoized(5, first), 5)
        self.assertEqual(second, {})
        self.assertEqual(fibonacci_memoized(3, second), 2)
        self.assertNotEqual(first, second)

    def test_valid_prepopulated_entry_is_used(self) -> None:
        cache = {10: 55}
        self.assertEqual(fibonacci_memoized(10, cache), 55)
        self.assertEqual(cache, {10: 55})


if __name__ == "__main__":
    unittest.main()

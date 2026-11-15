"""Recursive and memoized Fibonacci implementations for a bounded study."""

from collections.abc import MutableMapping


def _validate_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be a non-boolean integer")
    if n < 0:
        raise ValueError("n must be greater than or equal to zero")


def _fibonacci_baseline_recursive(n: int) -> int:
    if n < 2:
        return n
    return _fibonacci_baseline_recursive(n - 1) + _fibonacci_baseline_recursive(n - 2)


def fibonacci_baseline(n: int) -> int:
    """Return the nth Fibonacci number using direct recursive expansion."""
    _validate_n(n)
    return _fibonacci_baseline_recursive(n)


def _fibonacci_memoized_recursive(
    n: int,
    cache: MutableMapping[int, int],
) -> int:
    if n in cache:
        return cache[n]
    if n < 2:
        cache[n] = n
        return n
    cache[n] = _fibonacci_memoized_recursive(
        n - 1, cache
    ) + _fibonacci_memoized_recursive(n - 2, cache)
    return cache[n]


def fibonacci_memoized(
    n: int,
    cache: MutableMapping[int, int],
) -> int:
    """Return the nth Fibonacci number using the caller's shared cache."""
    _validate_n(n)
    if not isinstance(cache, MutableMapping):
        raise TypeError("cache must be a mutable mapping")
    return _fibonacci_memoized_recursive(n, cache)

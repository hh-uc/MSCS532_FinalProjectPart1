"""Correctness and measurement support for the Fibonacci experiment.

The harness validates known answers, measures repeated calls, gives the
memoized function a new empty cache for every call, and writes one raw CSV row
for every timed batch.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import platform
import sys
import time
import uuid
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BaselineFunction = Callable[[int], int]
MemoizedFunction = Callable[[int, MutableMapping[int, int]], int]

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
    32: 2178309,
    34: 5702887,
    35: 9227465,
}
MAX_NAIVE_N = 35
DEFAULT_CHECK_CASES = (0, 1, 2, 3, 5, 10, 20)
DEFAULT_SIZES = (20, 25, 30)
CSV_FIELDS = (
    "run_id",
    "recorded_at_utc",
    "algorithm",
    "n",
    "repetition",
    "order_in_pair",
    "batch_size",
    "elapsed_ns_total",
    "elapsed_ns_per_invocation",
    "result",
    "cache_mode",
    "cache_entries_after",
    "warmups",
    "repetitions",
    "timer",
    "python_version",
    "python_implementation",
    "platform_system",
    "platform_release",
    "platform_machine",
)


class HarnessError(RuntimeError):
    """Raised when an implementation or experiment violates the contract."""


@dataclass(frozen=True)
class Implementations:
    baseline: BaselineFunction
    memoized: MemoizedFunction


def load_implementations(module_name: str) -> Implementations:
    """Load the two required functions from a Python module."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover, exact import failures vary
        raise HarnessError(f"Could not import module {module_name!r}: {exc}") from exc

    functions: dict[str, Callable[..., int]] = {}
    for name in ("fibonacci_baseline", "fibonacci_memoized"):
        value = getattr(module, name, None)
        if not callable(value):
            raise HarnessError(f"Required callable {name!r} is missing")
        functions[name] = value

    return Implementations(
        baseline=functions["fibonacci_baseline"],
        memoized=functions["fibonacci_memoized"],
    )


def _expected_value(n: int) -> int:
    try:
        return KNOWN_VALUES[n]
    except KeyError as exc:
        choices = ", ".join(str(value) for value in sorted(KNOWN_VALUES))
        raise HarnessError(
            f"No independently recorded expected value exists for n={n}. "
            f"Use one of: {choices}, or add a separately verified value first."
        ) from exc


def _validate_result(label: str, n: int, result: object, expected: int) -> int:
    if isinstance(result, bool) or not isinstance(result, int):
        raise HarnessError(
            f"{label} returned {type(result).__name__} for n={n}; expected int"
        )
    if result != expected:
        raise HarnessError(
            f"{label} returned {result} for n={n}; expected {expected}"
        )
    return result


def _run_baseline(function: BaselineFunction, n: int, expected: int) -> int:
    try:
        result = function(n)
    except NotImplementedError as exc:
        raise HarnessError("The baseline algorithm is still unimplemented") from exc
    except Exception as exc:
        raise HarnessError(f"The baseline algorithm failed for n={n}: {exc}") from exc
    return _validate_result("baseline", n, result, expected)


def _run_memoized(
    function: MemoizedFunction,
    n: int,
    expected: int,
) -> tuple[int, int]:
    cache: dict[int, int] = {}
    try:
        result = function(n, cache)
    except NotImplementedError as exc:
        raise HarnessError("The memoized algorithm is still unimplemented") from exc
    except Exception as exc:
        raise HarnessError(f"The memoized algorithm failed for n={n}: {exc}") from exc
    validated = _validate_result("memoized", n, result, expected)
    _validate_memo_cache(cache, n)
    return validated, len(cache)


def _validate_memo_cache(cache: MutableMapping[int, int], n: int) -> None:
    """Require evidence that recursive subproblems passed through the cache."""
    if n < 2:
        return
    if n not in cache:
        raise HarnessError(
            f"The memoized algorithm did not store the requested result for n={n}"
        )
    minimum_entries = max(1, n - 1)
    if len(cache) < minimum_entries:
        raise HarnessError(
            "The supplied cache does not contain enough subproblem results for "
            f"n={n}: found {len(cache)}, expected at least {minimum_entries}. "
            "Ensure recursive calls pass through the memoized function."
        )


def validate_implementations(
    implementations: Implementations,
    cases: Iterable[int] = DEFAULT_CHECK_CASES,
) -> None:
    """Reject missing, failing, or incorrect implementations."""
    for n in cases:
        expected = _expected_value(n)
        _run_baseline(implementations.baseline, n, expected)
        _run_memoized(implementations.memoized, n, expected)


def _validate_experiment_inputs(
    sizes: Sequence[int],
    repetitions: int,
    warmups: int,
    memoized_batch_size: int,
) -> None:
    if not sizes:
        raise HarnessError("At least one input size is required")
    if repetitions < 1:
        raise HarnessError("Repetitions must be at least 1")
    if warmups < 0:
        raise HarnessError("Warmups cannot be negative")
    if memoized_batch_size < 1:
        raise HarnessError("Memoized batch size must be at least 1")
    for n in sizes:
        if n < 0:
            raise HarnessError("Input sizes cannot be negative")
        if n > MAX_NAIVE_N:
            raise HarnessError(
                f"n={n} exceeds the naive-workload safety bound of {MAX_NAIVE_N}"
            )
        _expected_value(n)


def _environment_fields() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }


def collect_measurements(
    implementations: Implementations,
    sizes: Sequence[int],
    repetitions: int,
    warmups: int,
    memoized_batch_size: int = 200,
) -> list[dict[str, object]]:
    """Collect validated raw timings without summarizing or interpreting them."""
    _validate_experiment_inputs(
        sizes, repetitions, warmups, memoized_batch_size
    )
    validation_cases = tuple(dict.fromkeys((*DEFAULT_CHECK_CASES, *sizes)))
    validate_implementations(implementations, validation_cases)

    for n in sizes:
        expected = _expected_value(n)
        for _ in range(warmups):
            _run_baseline(implementations.baseline, n, expected)
            _run_memoized(implementations.memoized, n, expected)

    run_id = str(uuid.uuid4())
    recorded_at = datetime.now(timezone.utc).isoformat()
    environment = _environment_fields()
    rows: list[dict[str, object]] = []

    for n in sizes:
        expected = _expected_value(n)
        for repetition in range(1, repetitions + 1):
            order = ("baseline", "memoized")
            if repetition % 2 == 0:
                order = tuple(reversed(order))

            for order_in_pair, algorithm in enumerate(order, start=1):
                if algorithm == "baseline":
                    started = time.perf_counter_ns()
                    try:
                        raw_result = implementations.baseline(n)
                    except NotImplementedError as exc:
                        raise HarnessError(
                            "The baseline algorithm is still unimplemented"
                        ) from exc
                    except Exception as exc:
                        raise HarnessError(
                            f"The baseline algorithm failed for n={n}: {exc}"
                        ) from exc
                    elapsed_ns_total = time.perf_counter_ns() - started
                    result = _validate_result(
                        "baseline", n, raw_result, expected
                    )
                    batch_size = 1
                    cache_mode = "not_applicable"
                    cache_entries_after = ""
                else:
                    caches: list[dict[int, int]] = [
                        {} for _ in range(memoized_batch_size)
                    ]
                    raw_results: list[object] = []
                    started = time.perf_counter_ns()
                    try:
                        for cache in caches:
                            raw_results.append(
                                implementations.memoized(n, cache)
                            )
                    except NotImplementedError as exc:
                        raise HarnessError(
                            "The memoized algorithm is still unimplemented"
                        ) from exc
                    except Exception as exc:
                        raise HarnessError(
                            f"The memoized algorithm failed for n={n}: {exc}"
                        ) from exc
                    elapsed_ns_total = time.perf_counter_ns() - started
                    for raw_result, cache in zip(raw_results, caches, strict=True):
                        _validate_result("memoized", n, raw_result, expected)
                        _validate_memo_cache(cache, n)
                    result = expected
                    batch_size = memoized_batch_size
                    cache_mode = "fresh_empty_per_call"
                    cache_sizes = {len(cache) for cache in caches}
                    cache_entries_after = (
                        cache_sizes.pop()
                        if len(cache_sizes) == 1
                        else f"{min(cache_sizes)}-{max(cache_sizes)}"
                    )

                row: dict[str, object] = {
                    "run_id": run_id,
                    "recorded_at_utc": recorded_at,
                    "algorithm": algorithm,
                    "n": n,
                    "repetition": repetition,
                    "order_in_pair": order_in_pair,
                    "batch_size": batch_size,
                    "elapsed_ns_total": elapsed_ns_total,
                    "elapsed_ns_per_invocation": elapsed_ns_total / batch_size,
                    "result": result,
                    "cache_mode": cache_mode,
                    "cache_entries_after": cache_entries_after,
                    "warmups": warmups,
                    "repetitions": repetitions,
                    "timer": "time.perf_counter_ns",
                    **environment,
                }
                rows.append(row)

    return rows


def write_csv(
    rows: Sequence[dict[str, object]],
    output: Path,
    overwrite: bool = False,
) -> None:
    """Write raw rows, refusing to replace prior evidence by default."""
    if output.exists() and not overwrite:
        raise HarnessError(
            f"Output already exists: {output}. Choose a new name or pass --overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and measure the Fibonacci implementations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="check known answers without recording timings"
    )
    check_parser.add_argument("--module", default="reference_algorithms")

    measure_parser = subparsers.add_parser(
        "measure", help="validate and record repeated raw timings"
    )
    measure_parser.add_argument("--module", default="reference_algorithms")
    measure_parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES)
    )
    measure_parser.add_argument("--repetitions", type=int, default=7)
    measure_parser.add_argument("--warmups", type=int, default=1)
    measure_parser.add_argument("--memoized-batch-size", type=int, default=200)
    measure_parser.add_argument("--output", type=Path, required=True)
    measure_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        implementations = load_implementations(args.module)
        if args.command == "check":
            validate_implementations(implementations)
            print("PASS: both implementations matched all known check cases")
            return 0

        rows = collect_measurements(
            implementations=implementations,
            sizes=args.sizes,
            repetitions=args.repetitions,
            warmups=args.warmups,
            memoized_batch_size=args.memoized_batch_size,
        )
        write_csv(rows, args.output, overwrite=args.overwrite)
        print(f"WROTE: {len(rows)} raw timing rows to {args.output}")
        return 0
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

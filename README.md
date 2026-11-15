# Fibonacci Memoization Study

This project compares direct recursive Fibonacci evaluation with a memoized version that reuses results through a caller-supplied dictionary.

## Requirements

- Python 3.11 or later

## Run the checks

```text
python -m unittest -v test_algorithms.py
python benchmark_harness.py check
```

## Record measurements

```text
python benchmark_harness.py measure \
  --sizes 20 25 30 \
  --repetitions 7 \
  --warmups 1 \
  --memoized-batch-size 200 \
  --output timings.csv
```

The memoized function trusts existing entries in the supplied cache, so the caller is responsible for their correctness. The harness gives the function a fresh empty dictionary for each measured invocation. Direct recursion is intentionally limited to n <= 35 because its work grows exponentially.

## Acknowledgment

OpenAI. (2026). *Codex* [Large language model; assisted with implementation, testing, analysis, and drafting]. https://openai.com/codex/

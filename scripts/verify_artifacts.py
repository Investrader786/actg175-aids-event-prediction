"""Reload every saved pipeline in a fresh Python process and confirm it reproduces
results/metrics_comparison.csv (the numbers in README.md) on test_data.csv.

Run from anywhere:  python scripts/verify_artifacts.py
Exit code 0 = all five models reproduce the reported metrics exactly.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import actg_eval as ev  # noqa: E402

TOLERANCE = 1e-9  # same code, same data -> identical floats expected


def main() -> int:
    test = pd.read_csv(ROOT / "test_data.csv")
    X, y, problems, _ = ev.check_schema(test)
    if problems:
        print("test_data.csv failed the schema check:", *problems, sep="\n  ")
        return 1

    expected = pd.read_csv(ROOT / "results" / "metrics_comparison.csv", index_col=0)
    failures = 0
    for name, file_name in ev.MODEL_REGISTRY.items():
        path = ROOT / "model" / file_name
        model = joblib.load(path)
        got, _, _ = ev.evaluate_model(model, X, y)
        off = {m: (got[m], expected.loc[name, m]) for m in ev.METRIC_NAMES
               if abs(got[m] - expected.loc[name, m]) > TOLERANCE}
        size_kb = path.stat().st_size / 1024
        print(f"{'OK ' if not off else 'BAD'} {name:<23} {size_kb:6.1f} KB  "
              + "  ".join(f"{m} {got[m]:.4f}" for m in ev.METRIC_NAMES))
        if off:
            failures += 1
            print("    differs:", off)

    print("\nAll five saved models reproduce results/metrics_comparison.csv."
          if not failures else f"\n{failures} model(s) do not reproduce the reported metrics.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

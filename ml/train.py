"""Train the learned surrogate on a dataset from dataset.py.

Output is the joblib bundle sim/surrogate/learned.py loads:
{"model", "feature_schema", "targets"}.

    uv run python train.py ../data/surrogate.npz --out ../data/surrogate.joblib
    # then: cd ../sim && uv run python server.py --model-path ../data/surrogate.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def load(path: Path) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    d = np.load(path)
    return d["x"], d["y"], str(d["feature_schema"]), [str(t) for t in d["targets"]]


def train(
    x: np.ndarray, y: np.ndarray, targets: list[str], seed: int = 0
) -> tuple[MultiOutputRegressor, dict[str, float]]:
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=seed))
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    metrics = {
        f"mae_{t}": float(mean_absolute_error(y_te[:, i], pred[:, i]))
        for i, t in enumerate(targets)
    }
    metrics["n_train"], metrics["n_test"] = len(x_tr), len(x_te)
    return model, metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    x, y, schema, targets = load(args.dataset)
    model, metrics = train(x, y, targets, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_schema": schema, "targets": targets}, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and benchmark LightGBM on the Kaggle credit-card fraud dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "ml-benchmark" / "creditcard.csv",
        help="Path to Kaggle's creditcard.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="Where to write the JSON benchmark results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Download creditcard.csv with the Kaggle CLI or pass --data PATH."
        )

    load_start = time.perf_counter()
    data = pd.read_csv(args.data)
    load_time_sec = time.perf_counter() - load_start

    if "Class" not in data.columns:
        raise SystemExit("Expected the Kaggle dataset to contain a 'Class' column.")

    features = data.drop(columns=["Class"])
    target = data["Class"].astype(int)
    if set(target.unique()) != {0, 1}:
        raise SystemExit("Expected 'Class' labels 0 (legitimate) and 1 (fraud).")

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=args.seed,
        stratify=target,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.125,
        random_state=args.seed,
        stratify=y_train_full,
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=negatives / positives,
        n_jobs=args.threads,
        random_state=args.seed,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )

    training_start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=0),
        ],
    )
    training_time_sec = time.perf_counter() - training_start
    best_iteration = int(model.best_iteration_ or model.n_estimators)

    probabilities = model.predict_proba(x_test, num_iteration=best_iteration)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }

    prediction_options = {"num_iteration": best_iteration}
    one_row = x_test.iloc[[0]]
    batch_rows = min(1000, len(x_test))
    batch = x_test.iloc[:batch_rows]
    model.predict_proba(batch.iloc[: min(10, batch_rows)], **prediction_options)

    single_row_ms = []
    for _ in range(200):
        start = time.perf_counter()
        model.predict_proba(one_row, **prediction_options)
        single_row_ms.append((time.perf_counter() - start) * 1000.0)

    batch_times_sec = []
    for _ in range(10):
        start = time.perf_counter()
        model.predict_proba(batch, **prediction_options)
        batch_times_sec.append(time.perf_counter() - start)

    mean_batch_time_sec = float(np.mean(batch_times_sec))
    inference = {
        "single_row_latency_ms_median": float(np.median(single_row_ms)),
        "single_row_latency_ms_mean": float(np.mean(single_row_ms)),
        "single_row_repeats": len(single_row_ms),
        "batch_rows": batch_rows,
        "batch_latency_sec_mean": mean_batch_time_sec,
        "batch_throughput_rows_per_sec": float(batch_rows / mean_batch_time_sec),
        "batch_repeats": len(batch_times_sec),
    }

    result = {
        "dataset": str(args.data),
        "dataset_rows": int(len(data)),
        "fraud_rows": int(target.sum()),
        "feature_count": int(features.shape[1]),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "test_rows": int(len(x_test)),
        "load_time_sec": float(load_time_sec),
        "training_time_sec": float(training_time_sec),
        "best_iteration": best_iteration,
        **metrics,
        **inference,
        "prediction_threshold": 0.5,
        "random_seed": args.seed,
        "threads": args.threads,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset: {args.data} ({len(data):,} rows, {features.shape[1]} features)")
    print(f"Data load time:              {load_time_sec:.3f} sec")
    print(f"Training time:               {training_time_sec:.3f} sec")
    print(f"Best iteration:              {best_iteration}")
    print(f"AUC-ROC:                     {metrics['auc_roc']:.6f}")
    print(f"Accuracy:                    {metrics['accuracy']:.6f}")
    print(f"F1-Score:                    {metrics['f1_score']:.6f}")
    print(f"Precision:                   {metrics['precision']:.6f}")
    print(f"Recall:                      {metrics['recall']:.6f}")
    print(f"Inference latency (1 row):   {inference['single_row_latency_ms_median']:.3f} ms median")
    print(
        f"Inference throughput ({batch_rows:,} rows): "
        f"{inference['batch_throughput_rows_per_sec']:.1f} rows/sec"
    )
    print(f"Results written to:          {args.output.resolve()}")


if __name__ == "__main__":
    main()

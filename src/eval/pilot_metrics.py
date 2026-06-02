from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import load_cfg


def _select_columns(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        for p in prefixes:
            if c.startswith(p):
                cols.append(c)
                break
    return cols


def get_policy(cfg: dict, policy_name: str | None):
    pols = (cfg.get("eval", {}).get("coverage_policies", {})) or {}
    default = pols.get("default", "A")
    name = policy_name or default
    params = pols.get(name, None)
    if params is None:
        # safe fallback defaults
        params = {"drop_no_news_days": False, "min_ticker_coverage_pct": None}
    return name, params




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", default=None, help="Coverage policy override (A/B/C). Defaults to cfg eval.coverage_policies.default")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_id = cfg["project"]["run_id"]
    data_dir = Path(cfg["project"]["data_dir"])
    processed = data_dir / "processed" / run_id

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ds_path = processed / "model_dataset.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing: {ds_path} (run build_model_dataset first)")

    df = pd.read_parquet(ds_path).copy()
    if "date" not in df.columns:
        raise ValueError("model_dataset must include a 'date' column")
    if "target_up_t1" not in df.columns or "target_ret_t1" not in df.columns:
        raise ValueError("model_dataset must include target_up_t1 and target_ret_t1")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    # ---- coverage policy (A/B/C) toggles ----
    policy_name, pol = get_policy(cfg, args.policy)

    drop_no_news = bool(pol.get("drop_no_news_days", False))
    min_cov = pol.get("min_ticker_coverage_pct", None)
    min_cov = None if min_cov in (None, "null") else float(min_cov)

    if (drop_no_news or (min_cov is not None)) and ("has_news" not in df.columns):
        raise ValueError("model_dataset must include has_news for coverage policies (run build_features_daily + build_model_dataset).")

    if "has_news" in df.columns:
        # normalize to int 0/1 for stable groupby mean and filtering
        df["has_news"] = df["has_news"].astype(int)

    # Filter by per-ticker coverage (mean has_news)
    if min_cov is not None:
        cov = df.groupby("ticker")["has_news"].mean()
        keep = cov[cov >= min_cov].index
        df = df[df["ticker"].isin(keep)].copy()

    # Optionally drop no-news days entirely
    if drop_no_news:
        df = df[df["has_news"] == 1].copy()

    # time split: first 70% unique dates train, remaining test
    dates = sorted(df["date"].unique().tolist())
    if len(dates) < 10:
        raise ValueError(f"Pilot window too small for metrics (only {len(dates)} unique dates).")
    split_idx = max(1, int(len(dates) * 0.7))
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train = df[df["date"].isin(train_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()

    splits_info: Dict[str, str] = {
        "run_id": run_id,
        "train_start": str(min(train["date"])),
        "train_end": str(max(train["date"])),
        "test_start": str(min(test["date"])),
        "test_end": str(max(test["date"])),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "policy_name": policy_name,
        "policy_params": pol,
        "policy_drop_no_news_days": drop_no_news,
        "policy_min_ticker_coverage_pct": min_cov,
    }

    # feature sets
    price_only_cols = _select_columns(df, ["px_"])
    vader_cols = [c for c in ["has_news", "n_articles", "sent_vader_mean", "sent_vader_sum"] if c in df.columns]
    finbert_cols = [
        c
        for c in [
            "has_news",
            "n_articles",
            "finbert_pos_mean",
            "finbert_neg_mean",
            "finbert_neu_mean",
            "finbert_score_mean",
        ]
        if c in df.columns
    ]

    variants = [
        ("price_only", price_only_cols),
        ("price_plus_vader", sorted(set(price_only_cols + vader_cols))),
        ("price_plus_finbert", sorted(set(price_only_cols + finbert_cols))),
    ]

    results = []
    policy_meta = {
        "policy_name": policy_name,
        "policy_drop_no_news_days": drop_no_news,
        "policy_min_ticker_coverage_pct": min_cov,
    }


    y_train_cls = train["target_up_t1"].astype(int).values
    y_test_cls = test["target_up_t1"].astype(int).values
    y_train_reg = train["target_ret_t1"].astype(float).values
    y_test_reg = test["target_ret_t1"].astype(float).values

    for name, cols in variants:
        if not cols:
            row = {"variant": name, "status": "skipped_no_features"}
            row.update(policy_meta)
            results.append(row)
            continue

        X_train = train[cols]
        X_test = test[cols]

        # If finbert is all-NaN in this pilot, skip gracefully
        if name == "price_plus_finbert":
            fin_cols_present = [c for c in cols if c.startswith("finbert_")]
            if not fin_cols_present:
                row = {"variant": name, "status": "skipped_finbert_not_available"}
                row.update(policy_meta)
                results.append(row)
                continue

            any_signal = False
            for c in fin_cols_present:
                series = pd.to_numeric(df[c], errors="coerce")
                if series.notna().any() and float(series.abs().sum()) > 0:
                    any_signal = True
                    break

            if not any_signal:
                row = {"variant": name, "status": "skipped_finbert_not_available"}
                row.update(policy_meta)
                results.append(row)
                continue

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scaler", StandardScaler()),
                ]), cols)
            ],
            remainder="drop",
        )

        # classification
        clf = Pipeline(steps=[
            ("pre", pre),
            ("model", LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=args.seed,
            )),
        ])
        clf.fit(X_train, y_train_cls)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # regression
        reg = Pipeline(steps=[
            ("pre", pre),
            ("model", Ridge(alpha=1.0, random_state=args.seed)),
        ])
        reg.fit(X_train, y_train_reg)
        yhat = reg.predict(X_test)

        row = {
            "variant": name,
            "status": "ok",
            "n_features": len(cols),

            "roc_auc": float(roc_auc_score(y_test_cls, proba)),
            "balanced_acc": float(balanced_accuracy_score(y_test_cls, pred)),
            "f1": float(f1_score(y_test_cls, pred)),

            "r2": float(r2_score(y_test_reg, yhat)),
            "mae": float(mean_absolute_error(y_test_reg, yhat)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_reg, yhat))),
        }
        row.update(policy_meta)
        results.append(row)

    metrics_path = reports_dir / "pilot_metrics_table.csv"
    splits_path = reports_dir / "pilot_splits.json"

    pd.DataFrame(results).to_csv(metrics_path, index=False)
    splits_path.write_text(json.dumps(splits_info, indent=2), encoding="utf-8")

    print(f"[OK] wrote {metrics_path}")
    print(f"[OK] wrote {splits_path}")


if __name__ == "__main__":
    main()

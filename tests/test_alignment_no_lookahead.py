from __future__ import annotations

from datetime import time as dtime

import pandas as pd
import pytest


def parse_cutoff(cutoff_str: str) -> dtime:
    hh, mm = cutoff_str.split(":")
    return dtime(int(hh), int(mm))


def compute_effective_date(published_london: pd.Series, cutoff: dtime) -> pd.Series:
    local_midnight = published_london.dt.normalize()
    after_cutoff = published_london.dt.time > cutoff
    effective_midnight = local_midnight + pd.to_timedelta(after_cutoff.astype(int), unit="D")
    return effective_midnight.dt.date


def test_effective_date_cutoff_rule_is_clone_safe() -> None:
    published = pd.Series(
        pd.to_datetime(
            [
                "2024-11-04 09:15:00+00:00",
                "2024-11-04 16:30:00+00:00",
                "2024-11-04 16:30:01+00:00",
            ]
        )
    )

    expected = pd.Series(
        pd.to_datetime(
            [
                "2024-11-04",
                "2024-11-04",
                "2024-11-05",
            ]
        ).date
    )

    actual = compute_effective_date(published, parse_cutoff("16:30"))

    assert actual.reset_index(drop=True).equals(expected)


def test_target_return_is_aligned_at_feature_date_t() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["HSBA.L", "HSBA.L", "HSBA.L"],
            "date": pd.to_datetime(["2024-11-01", "2024-11-04", "2024-11-05"]).date,
            "close": [100.0, 110.0, 99.0],
        }
    )
    features = prices[["ticker", "date"]].copy()
    features["px_ret_1d"] = [0.00, 0.10, -0.10]

    prices = prices.sort_values(["ticker", "date"])
    grouped = prices.groupby("ticker", group_keys=False)
    prices["target_ret_t1"] = grouped["close"].pct_change().shift(-1)
    prices["target_up_t1"] = (prices["target_ret_t1"] > 0).astype(int)

    model_dataset = features.merge(
        prices[["ticker", "date", "target_ret_t1", "target_up_t1"]],
        on=["ticker", "date"],
        how="left",
    ).dropna(subset=["target_ret_t1"])

    first_row = model_dataset.loc[model_dataset["date"] == pd.to_datetime("2024-11-01").date()].iloc[0]
    second_row = model_dataset.loc[model_dataset["date"] == pd.to_datetime("2024-11-04").date()].iloc[0]

    assert first_row["target_ret_t1"] == pytest.approx(0.10)
    assert first_row["target_up_t1"] == 1
    assert second_row["target_ret_t1"] == pytest.approx(-0.10)
    assert second_row["target_up_t1"] == 0
    assert pd.to_datetime("2024-11-05").date() not in set(model_dataset["date"])

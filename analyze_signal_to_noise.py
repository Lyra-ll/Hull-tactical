# -*- coding: utf-8 -*-
"""Comprehensive signal/noise diagnostics for target columns.

This script inspects a list of continuous or binary target columns and prints
summary statistics, annual breakdowns, rolling behaviour (for continuous
series), and lightweight autocorrelation diagnostics. It is intended for quick
EDA from the command line without plotting.

Example
-------
    python analyze_signal_to_noise.py --input train_final_features.csv \
        --columns forward_returns,market_forward_excess_returns,
                   resp_1d,resp_3d,resp_5d,
                   action_1d,action_3d,action_5d,
                   dls_target_1d,dls_target_3d,dls_target_5d,
                   sample_weight
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

try:  # Optional but improves p-values for small samples
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover - SciPy not available
    stats = None  # type: ignore

DEFAULT_COLUMNS: Sequence[str] = (
    "forward_returns",
    "market_forward_excess_returns",
    "resp_1d",
    "resp_3d",
    "resp_5d",
    "action_1d",
    "action_3d",
    "action_5d",
    "dls_target_1d",
    "dls_target_3d",
    "dls_target_5d",
    "sample_weight",
)


def _normal_two_tailed_pvalue(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


@dataclass
class ContinuousStats:
    segment: str
    n: int
    mean: float
    std: float
    snr: float
    annual_sharpe: float
    t_stat: float
    p_value: float


def compute_continuous_stats(series: pd.Series, label: str) -> ContinuousStats:
    values = series.dropna().values
    n = len(values)
    if n == 0:
        return ContinuousStats(label, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    snr = mean / std if std else float("inf")
    annual_sharpe = snr * math.sqrt(252.0)
    t_stat = snr * math.sqrt(n)

    if stats is not None and n > 2:
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        p_value = _normal_two_tailed_pvalue(t_stat)

    return ContinuousStats(label, n, mean, std, snr, annual_sharpe, t_stat, p_value)


def ljung_box_portmanteau(series: pd.Series, lags: Iterable[int]) -> List[tuple[int, float, float]]:
    clean = series.dropna().values
    n = len(clean)
    if n == 0:
        return []

    out: List[tuple[int, float, float]] = []
    for lag in lags:
        if lag >= n:
            break
        autocorr = pd.Series(clean).autocorr(lag)
        if np.isnan(autocorr):
            continue
        q_stat = n * (n + 2) * (autocorr ** 2) / (n - lag)
        if stats is not None:
            p_value = float(stats.chi2.sf(q_stat, df=lag))
        else:
            p_value = _normal_two_tailed_pvalue(math.sqrt(q_stat))
        out.append((lag, autocorr, p_value))
    return out


def analyse_continuous(
    df: pd.DataFrame,
    column: str,
    date_col: str,
    rolling_window: int,
    min_non_na: int,
) -> None:
    print("\n" + "#" * 80)
    print(f"CONTINUOUS TARGET :: {column}")
    series = df[column].astype(float)
    if date_col in df.columns:
        df = df.sort_values(date_col)
        series = df[column].astype(float)

    global_stats = compute_continuous_stats(series, "GLOBAL")
    print("=== GLOBAL STATISTICS ===")
    for field in ("n", "mean", "std", "snr", "annual_sharpe", "t_stat", "p_value"):
        value = getattr(global_stats, field)
        print(f"{field:>14}: {value:.6f}")

    if date_col in df.columns:
        print("\n=== CALENDAR SEGMENTS (YEARLY) ===")
        date_series = df[date_col]
        if pd.api.types.is_numeric_dtype(date_series):
            base_ts = pd.Timestamp("2000-01-01")
            datetime_idx = base_ts + pd.to_timedelta(date_series.astype(int), unit="D")
        else:
            datetime_idx = pd.to_datetime(date_series, errors="coerce")
        year_series = df.assign(_year=datetime_idx.dt.year.fillna(-1).astype(int))
        records = [
            compute_continuous_stats(group[column], str(year)).__dict__
            for year, group in year_series.groupby("_year", sort=True)
        ]
        seg_df = pd.DataFrame(records)
        print(seg_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    if rolling_window > 0:
        print(f"\n=== ROLLING WINDOW SNR (window={rolling_window}) ===")
        rolling = (
            series.rolling(window=rolling_window, min_periods=min_non_na)
            .apply(lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) else np.nan)
            .dropna()
        )
        if rolling.empty:
            print("Not enough data for rolling SNR.")
        else:
            desc = rolling.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            print(desc.to_string(float_format=lambda x: f"{x: .6f}"))

    print("\n=== AUTOCORRELATION & LJUNG-BOX ===")
    lb = ljung_box_portmanteau(series, lags=[1, 2, 3, 5])
    if lb:
        print("lag  autocorr      p-value")
        for lag, autocorr, p_value in lb:
            print(f"{lag:>3d}  {autocorr: .6f}  {p_value: .6f}")
    else:
        print("Insufficient data for Ljung-Box style check.")

    print("\n=== DISTRIBUTION SNAPSHOT ===")
    quantiles = series.dropna().quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(quantiles.to_string(float_format=lambda x: f"{x: .6f}"))


def analyse_binary(df: pd.DataFrame, column: str, date_col: str) -> None:
    print("\n" + "#" * 80)
    print(f"BINARY TARGET :: {column}")
    series = df[column].dropna()
    n = len(series)
    pos_rate = float(series.mean())
    neg_rate = 1.0 - pos_rate
    se = math.sqrt(pos_rate * neg_rate / n) if 0 < pos_rate < 1 else 0.0
    z = (pos_rate - 0.5) / se if se > 0 else float("inf")
    p_value = _normal_two_tailed_pvalue(z) if se > 0 else 0.0
    logit = math.log(pos_rate / neg_rate) if 0 < pos_rate < 1 else float("inf")

    print("=== GLOBAL STATISTICS ===")
    print(f"         count: {n}")
    print(f"    pos_ratio: {pos_rate:.6f}")
    print(f"    neg_ratio: {neg_rate:.6f}")
    print(f"         logit: {logit:.6f}")
    print(f"             z: {z:.6f}")
    print(f"        p_value: {p_value:.6f}")

    if date_col in df.columns:
        print("\n=== CALENDAR SEGMENTS (YEARLY) ===")
        date_series = df[date_col]
        if pd.api.types.is_numeric_dtype(date_series):
            base_ts = pd.Timestamp("2000-01-01")
            datetime_idx = base_ts + pd.to_timedelta(date_series.astype(int), unit="D")
        else:
            datetime_idx = pd.to_datetime(date_series, errors="coerce")
        year_series = df.assign(_year=datetime_idx.dt.year.fillna(-1).astype(int))
        rows = []
        for year, group in year_series.groupby("_year", sort=True):
            clean = group[column].dropna()
            n_year = len(clean)
            if n_year == 0:
                rows.append((year, 0, np.nan, np.nan, np.nan))
                continue
            pos = float(clean.mean())
            neg = 1 - pos
            se_year = math.sqrt(pos * neg / n_year) if 0 < pos < 1 else 0.0
            z_year = (pos - 0.5) / se_year if se_year > 0 else float("inf")
            p_year = _normal_two_tailed_pvalue(z_year) if se_year > 0 else 0.0
            rows.append((year, n_year, pos, z_year, p_year))
        seg_df = pd.DataFrame(rows, columns=["segment", "n", "pos_ratio", "z_stat", "p_value"])
        print(seg_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\n=== DISTRIBUTION SNAPSHOT ===")
    counts = series.value_counts(dropna=False).sort_index()
    for value, count in counts.items():
        print(f"  value={value!r} count={count} ratio={count / n:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signal-to-noise diagnostics for multiple targets")
    parser.add_argument("--input", required=True, help="CSV file containing the features/targets")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to inspect. If omitted, a default target set is used.",
    )
    parser.add_argument(
        "--date-column",
        default="date_id",
        help="Column representing chronological order. Use '' to disable date-based analysis.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=252,
        help="Rolling window length for SNR (continuous targets). Set 0 to disable.",
    )
    parser.add_argument(
        "--min-non-na",
        type=int,
        default=30,
        help="Minimum non-NA observations inside the rolling window to compute SNR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        columns = [c for c in DEFAULT_COLUMNS if c in df.columns]

    if not columns:
        raise ValueError("No valid columns to analyse. Provide --columns explicitly.")

    date_col = args.date_column if args.date_column and args.date_column in df.columns else ""

    for column in columns:
        if column not in df.columns:
            print(f"\nWARNING: column '{column}' missing in input; skipping.")
            continue

        series = df[column]
        unique_non_na = series.dropna().unique()
        if len(unique_non_na) <= 2 and set(np.round(unique_non_na, 8)).issubset({0.0, 1.0}):
            analyse_binary(df, column, date_col)
        else:
            analyse_continuous(df, column, date_col, args.rolling_window, args.min_non_na)


if __name__ == "__main__":
    main()

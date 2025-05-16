#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

def log_rmse(y_true_log, y_pred_log):
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))

def mape_orig(y_true_orig, y_pred_orig):
    mask = y_true_orig != 0
    return float(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask]).mean())

def load_wide_multivar(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    drop_cols = {"date", "city", "city_full", "zipcode"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    def _pivot(f):
        sub = df[["date", "zipcode", f]]
        pvt = sub.pivot(index="date", columns="zipcode", values=f)
        pvt.columns = [f"{f}_{z}" for z in pvt.columns]
        return pvt
    wide = pd.concat([_pivot(f) for f in numeric_cols], axis=1).sort_index().asfreq("M")
    const_cols = [c for c in wide.columns if wide[c].nunique() <= 1]
    if const_cols:
        print(f"[INFO] Dropping {len(const_cols)} constant columns.")
        wide = wide.drop(columns=const_cols)
    return wide

def _pca_transform(df_log, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log.values)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_scaled)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_Z = pd.DataFrame(Z, index=df_log.index, columns=cols)
    return df_Z, pca, scaler

def run_arima_pca_forecast(df_wide_log, p, pred_len, n_components=10):
    N = len(df_wide_log)
    train_N = int(N * 0.5)
    val_N = int((N - train_N) * 0.4)
    df_train = df_wide_log.iloc[:train_N]
    df_test = df_wide_log.iloc[train_N + val_N:]
    df_Z_train, pca, scaler = _pca_transform(df_train, n_components)
    fc_pca = np.zeros((pred_len, n_components))
    for i, col in enumerate(df_Z_train.columns):
        res = ARIMA(df_Z_train[col], order=(p, 0, 0)).fit()
        fc_pca[:, i] = res.forecast(steps=pred_len).values
    fc_log = scaler.inverse_transform(pca.inverse_transform(fc_pca))
    df_fc_log = pd.DataFrame(fc_log, index=df_test.index[:pred_len], columns=df_wide_log.columns)
    return _evaluate(df_wide_log, df_fc_log, pred_len, "arima", p)

def run_var_pca_forecast(df_wide_log, seq_len, pred_len, n_components=10):
    N = len(df_wide_log)
    train_N = int(N * 0.5)
    val_N = int((N - train_N) * 0.4)
    df_train = df_wide_log.iloc[:train_N]
    df_test = df_wide_log.iloc[train_N + val_N:]
    df_Z_train, pca, scaler = _pca_transform(df_train, n_components)
    model = VAR(df_Z_train)
    res = model.fit(maxlags=seq_len, ic=None)
    fc_pca = res.forecast(df_Z_train.values, steps=pred_len)
    fc_log = scaler.inverse_transform(pca.inverse_transform(fc_pca))
    df_fc_log = pd.DataFrame(fc_log, index=df_test.index[:pred_len], columns=df_wide_log.columns)
    return _evaluate(df_wide_log, df_fc_log, pred_len, "var", seq_len)

def _evaluate(df_wide_log, df_fc_log, pred_len, model, seq_len):
    df_test = df_wide_log.loc[df_fc_log.index]
    df_fc_orig = np.expm1(df_fc_log)
    df_test_orig = np.expm1(df_test)
    price_cols = [c for c in df_wide_log.columns if c.startswith("price_")]
    metrics = {"model": model, "seq_len": seq_len, "pred_len": pred_len, "log_RMSE": [], "MAPE": []}
    for pc in price_cols:
        metrics["log_RMSE"].append(log_rmse(df_test[pc].values[:pred_len], df_fc_log[pc].values))
        metrics["MAPE"].append(mape_orig(df_test_orig[pc].values, df_fc_orig[pc].values))
    for k in ("log_RMSE", "MAPE"):
        metrics[k] = float(np.mean(metrics[k])) if metrics[k] else np.nan
    return metrics

def sweep_models(df_wide_log, models, combos, n_components):
    results = []
    for model in models:
        for seq_len, pred_len in combos:
            print(f"\n=== {model.upper()} seq={seq_len} pred={pred_len} ===")
            if model == "arima":
                m = run_arima_pca_forecast(df_wide_log, seq_len, pred_len, n_components)
            elif model == "var":
                m = run_var_pca_forecast(df_wide_log, seq_len, pred_len, n_components)
            else:
                raise ValueError(model)
            print(f"log_RMSE={m['log_RMSE']:.4f} MAPE={m['MAPE']:.2f}")
            results.append(m)
    return pd.DataFrame(results)

def _parse_combos(s):
    pairs = re.findall(r"\(\s*(\d+)\s*[,:]\s*(\d+)\s*\)", s)
    if pairs:
        return [(int(a), int(b)) for a, b in pairs]
    nums = [int(x) for x in re.split(r"[;,\s]+", s.strip()) if x]
    if len(nums) % 2:
        raise ValueError
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model", default="all", choices=["arima", "var", "all"])
    p.add_argument("--combos", default="(6,3),(6,6),(6,12),(12,3),(12,6),(12,12)")
    p.add_argument("--n_components", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    df_wide_log = load_wide_multivar(args.csv)
    models = [args.model] if args.model != "all" else ["arima", "var"]
    combos = _parse_combos(args.combos)
    summary = sweep_models(df_wide_log, models, combos, args.n_components)
    print("\n===== SUMMARY =====")
    print(summary[["model", "seq_len", "pred_len", "log_RMSE", "MAPE"]].to_string(index=False))

if __name__ == "__main__":
    main()

# python arima_var_pca_forecast.py \
#     --csv /home/shengkun/KDD/Benchmark/Datasets/HouseTS_log.csv \
#     --model all \
#     --combos "(6,3),(6,6),(6,12),(12,3),(12,6),(12,12)" \
#     --n_components 10
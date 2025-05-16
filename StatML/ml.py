import argparse, re, warnings
warnings.filterwarnings("ignore", category=Warning)

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def log_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape_orig(a, b):
    m = a != 0
    return float(np.abs((a[m] - b[m]) / a[m]).mean())


def build_lag_xy(series, p, h):
    X, Y = [], []
    for t in range(p, len(series) - h + 1):
        X.append(series[t - p: t])
        Y.append(series[t: t + h])
    return np.asarray(X), np.asarray(Y)


def load_wide(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    drop = {"date", "city", "city_full", "zipcode"}
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]

    def piv(feat):
        sub = df[["date", "zipcode", feat]].pivot(index="date", columns="zipcode", values=feat)
        sub.columns = [f"{feat}_{z}" for z in sub.columns]
        return sub

    wide = pd.concat([piv(f) for f in num], axis=1).sort_index().asfreq("M")
    const = [c for c in wide.columns if wide[c].nunique() <= 1]
    if const:
        print(f"[INFO] Dropping {len(const)} constant columns.")
        wide = wide.drop(columns=const)
    return wide


def _pca(df, n_components):
    sc = StandardScaler()
    X = sc.fit_transform(df.values)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    return Z, pca, sc


def _evaluate(df_wide_log, df_fc_log, pred_len, model, seq_len):
    df_test = df_wide_log.loc[df_fc_log.index]
    df_fc_orig = np.expm1(df_fc_log)
    df_test_orig = np.expm1(df_test)
    price_cols = [c for c in df_wide_log.columns if c.startswith("price_")]
    metr = {"model": model, "seq_len": seq_len, "pred_len": pred_len, "log_RMSE": [], "MAPE": []}
    for pc in price_cols:
        metr["log_RMSE"].append(log_rmse(df_test[pc].values[:pred_len], df_fc_log[pc].values))
        metr["MAPE"].append(mape_orig(df_test_orig[pc].values, df_fc_orig[pc].values))
    for k in ("log_RMSE", "MAPE"):
        metr[k] = float(np.mean(metr[k])) if metr[k] else np.nan
    return metr


def run_rf(df_wide_log, p, h, n_components):
    N = len(df_wide_log)
    tr = int(N * 0.5)
    va = int((N - tr) * 0.4)
    df_tr = df_wide_log.iloc[:tr]
    df_te = df_wide_log.iloc[tr + va:]

    Z_tr, pca, sc = _pca(df_tr, n_components)
    fc = np.zeros((h, n_components))
    for j in range(n_components):
        s = Z_tr[:, j]
        X_tr, Y_tr = build_lag_xy(s, p, h)
        base = RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base)
        model.fit(X_tr, Y_tr)
        fc[:, j] = model.predict(s[-p:].reshape(1, -1)).flatten()
    fc_log = sc.inverse_transform(pca.inverse_transform(fc))
    df_fc_log = pd.DataFrame(fc_log, index=df_te.index[:h], columns=df_wide_log.columns)
    return _evaluate(df_wide_log, df_fc_log, h, "rf", p)


def run_xgb(df_wide_log, p, h, n_components):
    N = len(df_wide_log)
    tr = int(N * 0.5)
    va = int((N - tr) * 0.4)
    df_tr = df_wide_log.iloc[:tr]
    df_te = df_wide_log.iloc[tr + va:]

    Z_tr, pca, sc = _pca(df_tr, n_components)
    fc = np.zeros((h, n_components))
    for j in range(n_components):
        s = Z_tr[:, j]
        X_tr, Y_tr = build_lag_xy(s, p, h)
        base = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, objective="reg:squarederror", random_state=42)
        model = MultiOutputRegressor(base)
        model.fit(X_tr, Y_tr)
        fc[:, j] = model.predict(s[-p:].reshape(1, -1)).flatten()
    fc_log = sc.inverse_transform(pca.inverse_transform(fc))
    df_fc_log = pd.DataFrame(fc_log, index=df_te.index[:h], columns=df_wide_log.columns)
    return _evaluate(df_wide_log, df_fc_log, h, "xgb", p)


def sweep(df_wide_log, models, combos, n_components):
    res = []
    for m in models:
        for p, h in combos:
            print(f"\n=== {m.upper()} seq={p} pred={h} ===")
            if m == "rf":
                r = run_rf(df_wide_log, p, h, n_components)
            elif m == "xgb":
                r = run_xgb(df_wide_log, p, h, n_components)
            else:
                raise ValueError(m)
            print(f"log_RMSE={r['log_RMSE']:.4f} MAPE={r['MAPE']:.2f}")
            res.append(r)
    return pd.DataFrame(res)


def parse_combos(s):
    pairs = re.findall(r"\(\s*(\d+)\s*[,:]\s*(\d+)\s*\)", s)
    if pairs:
        return [(int(a), int(b)) for a, b in pairs]
    nums = [int(x) for x in re.split(r"[;,\s]+", s.strip()) if x]
    if len(nums) % 2:
        raise ValueError
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="all", choices=["rf", "xgb", "all"])
    ap.add_argument("--combos", default="(6,3),(6,6),(6,12),(12,3),(12,6),(12,12)")
    ap.add_argument("--n_components", type=int, default=10)
    args = ap.parse_args()

    df_wide_log = load_wide(args.csv)
    models = [args.model] if args.model != "all" else ["rf", "xgb"]
    combos = parse_combos(args.combos)

    summary = sweep(df_wide_log, models, combos, args.n_components)
    print("\n===== SUMMARY =====")
    print(summary[["model", "seq_len", "pred_len", "log_RMSE", "MAPE"]].to_string(index=False))


if __name__ == "__main__":
    main()

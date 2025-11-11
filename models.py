"""
Run forecasting baselines + core predictive regression with rolling OOS evaluation.

Inputs (already created by your preprocessing):
  - Oil_Model_Weekly_Core.csv

Models:
  1) Naive: 0 return
  2) AR(1): r_t = c + φ r_{t-1}
  3) OLS predictive (expanding window) with selected core predictors

Outputs (printed to console):
  - Metrics per model: RMSE, MAE, HitRate, TheilU
  - Simple trading P&L stats (long/flat based on sign of forecast)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

# Fix scipy/statsmodels compatibility issue
# _centered was removed in scipy 1.8.0+ but older statsmodels still needs it
try:
    import scipy.signal
    import scipy.signal.signaltools
    # Patch _centered before statsmodels tries to import it
    if not hasattr(scipy.signal.signaltools, '_centered'):
        def _centered(arr, newsize):
            """Compatibility shim for removed scipy._centered function"""
            # Return the first newsize elements, matching original behavior
            return arr[:newsize]
        # Inject into the module namespace
        scipy.signal.signaltools._centered = _centered
        # Also ensure it's accessible via from ... import
        import sys
        if 'scipy.signal.signaltools' in sys.modules:
            sys.modules['scipy.signal.signaltools']._centered = _centered
except Exception:
    pass  # If scipy isn't available, let it fail naturally

import statsmodels.api as sm

# ------------------ Config ------------------
DATA = Path("Oil_Model_Weekly_Core.csv")
# last N% as test (you can switch to a fixed date if you prefer)
TEST_FRAC = 0.20

# Core predictors to try (only those present will be used)
CORE_PREDICTORS = [
    "cl1_ret_l1", "cl1_ret_l2", "cl1_ret_l3", "cl1_ret_l4",
    "brent_ret", "cl2_ret", "bdiy_index_ret", "spx_ret", "dxy_returns",
    "weekly_change_in_crude_stock_diff", "weekly_rig_count_diff", "weekly_commercial_long_diff",
    "weekly_commercial_short_diff", "weekly_total_open_interest_diff", "weekly_prod_crude_diff",
    "3m_yield_diff", "10y_yield_diff", "term_spread_diff", "vix_diff",
    "dow_jones_us_oilandgas_index_ret", "exxonmobil_ret", "conocophilips_ret",
    "chevron_ret", "bp_ret", "totalenergies_ret"
]

# ----------------- Helpers -------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    e = y_true - y_pred
    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))
    # Theil's U: compare to naive (predict 0)
    rmse_naive = float(np.sqrt(np.mean(y_true**2)))
    theilu = rmse / rmse_naive if rmse_naive > 0 else np.nan
    # Directional accuracy
    hit = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {"RMSE": rmse, "MAE": mae, "HitRate": hit, "TheilU": theilu}

def toy_pnl(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Long/flat rule: if forecast > 0 -> long for 1 week; else flat.
    P&L = position * realized return (no costs).
    """
    pos = (y_pred > 0).astype(float)
    ret = pos * y_true
    cum = np.cumsum(ret)
    sharpe = (ret.mean() / ret.std()) * np.sqrt(52) if ret.std() > 0 else np.nan
    return {
        "AnnRet(approx)": float(ret.mean() * 52),
        "AnnVol(approx)": float(ret.std() * np.sqrt(52)),
        "Sharpe(approx)": float(sharpe),
        "CumulativeRet": float(cum.iloc[-1])
    }

def expanding_ols_forecast(df: pd.DataFrame, target: str, Xcols: List[str], split_idx: int) -> pd.Series:
    """
    Refit OLS each step on [0:t) to predict t, for t in [split_idx .. end).
    """
    preds = []
    idxs = []
    for t in range(split_idx, len(df)):
        train = df.iloc[:t]
        testX = df.iloc[[t]][Xcols]
        # Drop rows with NAs in the expanding train
        trn = train.dropna(subset=[target] + Xcols)
        if trn.empty:
            preds.append(np.nan); idxs.append(df.index[t]); continue
        y = trn[target].values
        X_train = trn[Xcols]
        train_means = X_train.mean()
        train_stds = X_train.std(ddof=0).replace(0, 1)
        X_train_std = (X_train - train_means) / train_stds
        X = sm.add_constant(X_train_std.values, has_constant='add')
        mdl = sm.OLS(y, X).fit()
        # Ensure testX has the same columns and handle NaNs
        testX_clean = testX.copy()
        for col in Xcols:
            if pd.isna(testX_clean[col].iloc[0]):
                testX_clean[col].iloc[0] = train_means[col]
        testX_std = (testX_clean - train_means) / train_stds
        X_test = sm.add_constant(testX_std.values, has_constant='add')
        pred = float(mdl.predict(X_test)[0])
        preds.append(pred); idxs.append(df.index[t])
    return pd.Series(preds, index=idxs, name="ols_pred")

def expanding_ar1_forecast(df: pd.DataFrame, target: str, split_idx: int) -> pd.Series:
    tmp = df[[target]].copy()
    tmp["lag1"] = tmp[target].shift(1)

    preds, idxs = [], []
    for t in range(split_idx, len(tmp)):
        trn = tmp.iloc[:t].dropna()
        if trn.empty:
            preds.append(np.nan); idxs.append(tmp.index[t]); continue
        y = trn[target]
        X = trn[["lag1"]]
        X_const = sm.add_constant(X, has_constant='add')
        mdl = sm.OLS(y.values, X_const.values).fit()

        x_t = tmp.iloc[[t]][["lag1"]]
        x_t = x_t.fillna(X["lag1"].mean())  # if needed
        x_t_const = sm.add_constant(x_t, has_constant='add')
        x_t_const = x_t_const[X_const.columns]

        preds.append(float(mdl.predict(x_t_const.values)[0]))
        idxs.append(tmp.index[t])

    return pd.Series(preds, index=idxs, name="ar1_pred")



# ----------------- Main -------------------
def main():
    df = pd.read_csv(DATA, parse_dates=[0], index_col=0)
    target = "cl1_ret"
    if target not in df.columns:
        raise ValueError("cl1_ret not found in dataset.")

    # Train/test split index
    split_idx = int((1 - TEST_FRAC) * len(df))
    y = df[target]

    # NAIVE forecast = 0
    naive_pred = pd.Series(0.0, index=df.index[split_idx:], name="naive_pred")

    # AR(1) expanding forecast
    ar1_pred = expanding_ar1_forecast(df, target, split_idx)

    # OLS predictive regression (select only available predictors)
    Xcols = [c for c in CORE_PREDICTORS if c in df.columns]
    if not Xcols:
        raise ValueError("No core predictors found in dataset. Check column names.")
    ols_pred = expanding_ols_forecast(df, target, Xcols, split_idx)

    # Align true and preds on test range
    y_test = y.loc[naive_pred.index]
    preds = {
        "Naive": naive_pred.loc[y_test.index],
        "AR(1)": ar1_pred.loc[y_test.index],
        "OLS(core)": ols_pred.loc[y_test.index],
    }

    # Metrics
    rows = []
    for name, p in preds.items():
        valid = y_test.dropna().index.intersection(p.dropna().index)
        m = metrics(y_test.loc[valid].values, p.loc[valid].values)
        pnl = toy_pnl(y_test.loc[valid], p.loc[valid])
        row = {"Model": name, **m, **pnl, "N_test": len(valid)}
        rows.append(row)

    res = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
    print("\n=== Rolling OOS Results (expanding window) ===")
    print(res.round(4))

    # Optional: save the results
    out_path = Path("OOS_results.csv")
    res.to_csv(out_path)
    print(f"\nSaved results to: {out_path.resolve()}")

    # (Optional) peek at last few predictions
    tail = pd.DataFrame({
        "y_true": y_test.tail(10),
        "naive": preds["Naive"].tail(10),
        "ar1": preds["AR(1)"].tail(10),
        "ols": preds["OLS(core)"].tail(10),
    })
    print("\nTail of test set predictions:")
    print(tail.round(5))

if __name__ == "__main__":
    main()

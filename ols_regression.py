"""
OLS exploratory modelling for the Oil coursework project.

Loads the model-ready weekly dataset, fits a baseline OLS using a curated
set of predictors, and reports regression diagnostics plus VIFs to flag
multicollinearity.

Run:
    python ols_regression.py
"""

from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_FILE = Path("Oil_Model_Weekly_Core.csv")
OUTPUT_DIR = Path("ols_diagnostics")
TEST_FRAC = 0.20  # Fraction of data to use for out-of-sample testing

# Predictors chosen for a trader-friendly, interpretable model
# (all series are already standardized in the core dataset).
CANDIDATE_PREDICTORS: List[str] = [
    # Autoregressive structure
    # "cl1_ret_l1",
    # "cl1_ret_l2",
    # "cl1_ret_l3",
    "cl1_ret_l4",
    # Cross-asset flows
    # "brent_ret",
    # "cl2_ret",
    # "spx_ret",
    # "dxy_returns",
    # Weekly oil fundamentals
    "weekly_change_in_crude_stock_diff",
    # "weekly_rig_count_diff",
    # "weekly_prod_crude_diff",
    # "bdiy_index_ret",
    # "weekly_commercial_long_diff",
    # "weekly_commercial_short_diff", #because both are included in the total open interest
    # "weekly_total_open_interest_diff",
    # Macro sentiment & rates
    # "3m_yield_diff",
    # "10y_yield_diff",
    # "term_spread_diff",
    # "vix_diff",
    # Energy equities
    # "dow_jones_us_oilandgas_index_ret",
    # "exxonmobil_ret",
    # "conocophilips_ret",
    # "chevron_ret",
    # "bp_ret",
    # "totalenergies_ret",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run process_oil_data.py to regenerate the dataset."
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def prepare_design(
    df: pd.DataFrame, predictors: List[str]
) -> Tuple[pd.Series, pd.DataFrame, List[str], List[str]]:
    available = [c for c in predictors if c in df.columns]
    missing = sorted(set(predictors) - set(available))

    design = df[["cl1_ret"] + available].dropna()
    y = design["cl1_ret"]
    X = sm.add_constant(design[available])
    return y, X, available, missing


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # Skip the constant column when calculating VIFs
    cols = [c for c in X.columns if c != "const"]
    X_numeric = X[cols].values

    vif_data = []
    for i, col in enumerate(cols):
        vif = variance_inflation_factor(X_numeric, i)
        vif_data.append({"variable": col, "VIF": vif})

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_residual_plots(model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    ensure_output_dir(OUTPUT_DIR)
    resid = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(10, 4))
    plt.plot(resid, color="steelblue", linewidth=0.7)
    plt.title("OLS Residuals (time series)")
    plt.ylabel("Residual")
    plt.xlabel("Observation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_timeseries.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, resid, s=10, color="darkorange", alpha=0.6)
    plt.axhline(0, color="gray", linewidth=1, linestyle="--")
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_vs_fitted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=40, color="slateblue", edgecolor="white", alpha=0.85)
    plt.title("Residual histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_hist.png", dpi=150)
    plt.close()


def save_acf_pacf(resid: pd.Series, lags: int = 24) -> None:
    ensure_output_dir(OUTPUT_DIR)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(resid, lags=lags, ax=ax[0])
    ax[0].set_title(f"Residual ACF (lags 1–{lags})")
    plot_pacf(resid, lags=lags, ax=ax[1], method="ywm")
    ax[1].set_title(f"Residual PACF (lags 1–{lags})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals_acf_pacf.png", dpi=150)
    plt.close(fig)


def run_ljung_box(resid: pd.Series, lags: List[int]) -> pd.DataFrame:
    lb = acorr_ljungbox(resid, lags=lags, return_df=True)
    lb.index.name = "lag"
    lb.reset_index(inplace=True)
    lb.rename(columns={"lb_stat": "LB stat", "lb_pvalue": "p-value"}, inplace=True)
    return lb


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, HitRate, TheilU."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    hit_rate = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    rmse_naive = float(np.sqrt(np.mean(y_true ** 2)))
    theil_u = rmse / rmse_naive if rmse_naive > 0 else np.nan
    return {
        "RMSE": rmse,
        "MAE": mae,
        "HitRate": hit_rate,
        "TheilU": theil_u,
    }


def compute_pnl(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Toy P&L: long if forecast > 0, else flat."""
    pos = (y_pred > 0).astype(float)
    ret = pos * y_true
    sharpe = (
        (ret.mean() / ret.std()) * np.sqrt(52) if ret.std() > 0 else np.nan
    )
    return {
        "AnnRet(approx)": float(ret.mean() * 52),
        "AnnVol(approx)": float(ret.std() * np.sqrt(52)),
        "Sharpe(approx)": float(sharpe),
        "CumulativeRet": float(ret.sum()),
    }


def expanding_ols_forecast(
    df: pd.DataFrame, target: str, predictors: List[str], split_idx: int
) -> pd.Series:
    """Expanding-window OLS forecast: refit on [0:t) to predict t."""
    preds = []
    idxs = []
    
    for t in range(split_idx, len(df)):
        train = df.iloc[:t]
        test_row = df.iloc[[t]]
        
        # Prepare training data
        trn = train.dropna(subset=[target] + predictors)
        if trn.empty or len(trn) < len(predictors) + 2:
            preds.append(np.nan)
            idxs.append(df.index[t])
            continue
        
        # Standardize predictors using training sample only
        y_trn = trn[target].values
        X_trn = trn[predictors].copy()
        train_means = X_trn.mean()
        train_stds = X_trn.std(ddof=0).replace(0, 1)
        X_trn_std = (X_trn - train_means) / train_stds
        X_trn_const = sm.add_constant(X_trn_std.values, has_constant='add')
        
        # Fit model
        try:
            mdl = sm.OLS(y_trn, X_trn_const).fit()
        except:
            preds.append(np.nan)
            idxs.append(df.index[t])
            continue
        
        # Prepare test data (standardize using training stats)
        X_test = test_row[predictors].copy()
        X_test = X_test.fillna(train_means)
        X_test_std = (X_test - train_means) / train_stds
        X_test_const = sm.add_constant(X_test_std.values, has_constant='add')
        
        # Predict
        try:
            pred = float(mdl.predict(X_test_const)[0])
        except:
            pred = np.nan
        
        preds.append(pred)
        idxs.append(df.index[t])
    
    return pd.Series(preds, index=idxs, name="ols_pred")


def expanding_ar1_forecast(df: pd.DataFrame, target: str, split_idx: int) -> pd.Series:
    """Expanding-window AR(1) forecast."""
    tmp = df[[target]].copy()
    tmp["lag1"] = tmp[target].shift(1)
    
    preds, idxs = [], []
    for t in range(split_idx, len(tmp)):
        trn = tmp.iloc[:t].dropna()
        if trn.empty:
            preds.append(np.nan)
            idxs.append(tmp.index[t])
            continue
        
        y_trn = trn[target].values
        X_trn = sm.add_constant(trn[["lag1"]].values, has_constant='add')
        mdl = sm.OLS(y_trn, X_trn).fit()
        
        x_test = tmp.iloc[[t]][["lag1"]].fillna(trn["lag1"].mean())
        x_test_const = sm.add_constant(x_test.values, has_constant='add')
        
        try:
            pred = float(mdl.predict(x_test_const)[0])
        except:
            pred = np.nan
        
        preds.append(pred)
        idxs.append(tmp.index[t])
    
    return pd.Series(preds, index=idxs, name="ar1_pred")


def main():
    df = load_data(DATA_FILE)
    y, X, used_predictors, missing_predictors = prepare_design(
        df, CANDIDATE_PREDICTORS
    )

    print("=" * 72)
    print("OLS EXPLORATORY MODEL")
    print("=" * 72)
    print(f"Total observations in dataset   : {len(df):,}")
    print(f"Sample size after dropna        : {len(y):,}")
    print(f"Predictors used ({len(used_predictors)}): {used_predictors}")
    if missing_predictors:
        print(
            f"⚠️  Missing columns ({len(missing_predictors)}): {missing_predictors}"
        )
    print()

    model = sm.OLS(y, X).fit()
    print(model.summary())

    print("\nVariance Inflation Factors (VIF):")
    vif_table = compute_vif(X)
    print(vif_table.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

    save_residual_plots(model)
    save_acf_pacf(model.resid)
    lb_table = run_ljung_box(model.resid, lags=[4, 8, 12, 24])

    print("\nLjung-Box test on residuals:")
    print(lb_table.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print(f"\nDiagnostic plots saved to: {OUTPUT_DIR.resolve()}")
    print("Review residual shape, serial correlation, and Ljung-Box p-values.")
    print("Consider robust errors or feature tweaks if diagnostics flag issues.")
    
    # ========== OUT-OF-SAMPLE EVALUATION ==========
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE EVALUATION (Expanding Window)")
    print("=" * 72)
    
    target = "cl1_ret"
    split_idx = int((1 - TEST_FRAC) * len(df))
    y_test = df[target].iloc[split_idx:]
    
    # Naive forecast (always 0)
    naive_pred = pd.Series(0.0, index=y_test.index, name="naive_pred")
    
    # AR(1) forecast
    ar1_pred = expanding_ar1_forecast(df, target, split_idx)
    
    # OLS forecast with final predictor set
    ols_pred = expanding_ols_forecast(df, target, used_predictors, split_idx)
    
    # Align all predictions
    valid_idx = y_test.dropna().index.intersection(
        naive_pred.dropna().index.intersection(
            ar1_pred.dropna().index.intersection(ols_pred.dropna().index)
        )
    )
    
    if len(valid_idx) == 0:
        print("⚠️  No valid predictions after alignment. Check data.")
        return
    
    y_test_aligned = y_test.loc[valid_idx]
    naive_aligned = naive_pred.loc[valid_idx]
    ar1_aligned = ar1_pred.loc[valid_idx]
    ols_aligned = ols_pred.loc[valid_idx]
    
    # Compute metrics for each model
    results = []
    for name, pred in [
        ("Naive", naive_aligned),
        ("AR(1)", ar1_aligned),
        ("OLS(final)", ols_aligned),
    ]:
        m = compute_metrics(y_test_aligned.values, pred.values)
        pnl = compute_pnl(y_test_aligned, pred)
        results.append({
            "Model": name,
            **m,
            **pnl,
            "N_test": len(valid_idx),
        })
    
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n=== Out-of-Sample Performance ===")
    print(results_df.round(4).to_string())
    
    # Save results
    oos_file = OUTPUT_DIR / "oos_results.csv"
    ensure_output_dir(OUTPUT_DIR)
    results_df.to_csv(oos_file)
    print(f"\nSaved OOS results to: {oos_file.resolve()}")
    
    # Show tail of predictions
    tail_df = pd.DataFrame({
        "y_true": y_test_aligned.tail(10),
        "naive": naive_aligned.tail(10),
        "ar1": ar1_aligned.tail(10),
        "ols": ols_aligned.tail(10),
    })
    print("\nTail of test set predictions:")
    print(tail_df.round(5).to_string())


if __name__ == "__main__":
    main()


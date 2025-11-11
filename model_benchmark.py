"""
Comprehensive model benchmarking for Oil returns forecasting.

Runs all candidate models (Naive, AR(1), OLS, ARIMAX) with consistent
train/test splits and evaluation metrics. Generates unified comparison tables
and diagnostic outputs.

Run:
    python model_benchmark.py
"""

from pathlib import Path
from typing import List, Dict, Tuple
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


DATA_FILE = Path("Oil_Model_Weekly_Core.csv")
OUTPUT_DIR = Path("model_benchmarks")
TEST_FRAC = 0.20

# Final OLS predictors (from screening stage)
OLS_PREDICTORS = [
    "cl1_ret_l4",
    "weekly_change_in_crude_stock_diff",
]

# Extended OLS predictors (to capture lag-8 autocorrelation)
OLS_PREDICTORS_EXTENDED = OLS_PREDICTORS + ["cl1_ret_l8"]  # Add lag 8 to capture autocorrelation

# ARIMAX configuration
ARIMAX_EXOG = OLS_PREDICTORS.copy()
# Expanded grid to capture residual autocorrelation better
NON_SEASONAL_GRID = [
    (0, 0), (1, 0), (0, 1), (1, 1),
    (2, 0), (0, 2), (2, 1), (1, 2), (2, 2),  # Higher orders
    (3, 0), (0, 3), (3, 1), (1, 3),  # Even higher orders
]
SEASONAL_CONFIGS = [
    {"period": 7, "orders": [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]},
    {"period": 13, "orders": [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1)]},
    {"period": 8, "orders": [(1, 0), (0, 1), (1, 1)]},  # Try lag 8 specifically
]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run process_oil_data.py to regenerate the dataset."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


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


def naive_forecast(y_test: pd.Series) -> pd.Series:
    """Always predict zero."""
    return pd.Series(0.0, index=y_test.index, name="naive_pred")


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


def expanding_ols_forecast(
    df: pd.DataFrame, target: str, predictors: List[str], split_idx: int
) -> pd.Series:
    """Expanding-window OLS forecast: refit on [0:t) to predict t."""
    preds = []
    idxs = []
    
    for t in range(split_idx, len(df)):
        train = df.iloc[:t]
        test_row = df.iloc[[t]]
        
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
        
        try:
            pred = float(mdl.predict(X_test_const)[0])
        except:
            pred = np.nan
        
        preds.append(pred)
        idxs.append(df.index[t])
    
    return pd.Series(preds, index=idxs, name="ols_pred")


def grid_search_arimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float, SARIMAX]:
    """Grid search for best ARIMAX order."""
    best_order = None
    best_seasonal_order = None
    best_aic = np.inf
    best_model = None

    for p, q in NON_SEASONAL_GRID:
        order = (p, 0, q)
        for cfg in SEASONAL_CONFIGS:
            s = cfg["period"]
            for P, Q in cfg["orders"]:
                seasonal_order = (P, 0, Q, s)
                try:
                    model = SARIMAX(
                        y_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        exog=exog_train,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = order
                        best_seasonal_order = seasonal_order
                        best_model = results
                except Exception:
                    continue

    if best_model is None:
        raise RuntimeError("ARIMAX grid search failed for all candidate orders.")

    return best_order, best_seasonal_order, best_aic, best_model


def arimax_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    exog_cols: List[str],
) -> Tuple[pd.Series, Dict]:
    """Fit ARIMAX on training set and forecast test set."""
    # Prepare training data
    trn = train_df.dropna(subset=[target] + exog_cols)
    if trn.empty:
        raise ValueError("No valid training data for ARIMAX.")
    
    y_trn = trn[target]
    exog_trn = trn[exog_cols].copy()
    
    # Standardize exog using training sample
    train_means = exog_trn.mean()
    train_stds = exog_trn.std(ddof=0).replace(0, 1)
    exog_trn_std = (exog_trn - train_means) / train_stds
    
    # Grid search on training data
    print("   Grid searching ARIMAX orders...")
    best_order, best_seasonal, best_aic, best_model = grid_search_arimax(
        y_trn, exog_trn_std
    )
    
    model_info = {
        "order": best_order,
        "seasonal_order": best_seasonal,
        "aic": best_aic,
    }
    
    # Prepare test exog
    exog_test = test_df[exog_cols].copy()
    exog_test = exog_test.fillna(train_means)
    exog_test_std = (exog_test - train_means) / train_stds
    
    # Forecast
    print(f"   Fitted ARIMAX{best_order} × {best_seasonal} (AIC={best_aic:.2f})")
    print("   Generating test forecasts...")
    
    forecasts = best_model.get_forecast(steps=len(test_df), exog=exog_test_std)
    preds = forecasts.predicted_mean
    
    # Align index with test_df
    preds.index = test_df.index[:len(preds)]
    preds.name = "arimax_pred"
    
    return preds, model_info


def fit_ols_train(
    train_df: pd.DataFrame, target: str, predictors: List[str], use_robust: bool = True
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.Series]:
    """Fit OLS on training data and return model + residuals.
    
    Args:
        use_robust: If True, use HAC (Newey-West) robust standard errors to account
                   for heteroskedasticity and autocorrelation.
    """
    trn = train_df.dropna(subset=[target] + predictors)
    if trn.empty:
        raise ValueError("No valid training data for OLS.")
    
    y_trn = trn[target].values
    X_trn = trn[predictors].copy()
    
    # Standardize
    train_means = X_trn.mean()
    train_stds = X_trn.std(ddof=0).replace(0, 1)
    X_trn_std = (X_trn - train_means) / train_stds
    X_const = sm.add_constant(X_trn_std.values, has_constant='add')
    
    model = sm.OLS(y_trn, X_const).fit()
    
    # Use robust standard errors if requested (HAC - Newey-West)
    if use_robust:
        model = model.get_robustcov_results(cov_type='HAC', maxlags=8)
    
    resid = pd.Series(model.resid, index=trn.index)
    
    return model, resid


def fit_arimax_train(
    train_df: pd.DataFrame, target: str, exog_cols: List[str]
) -> Tuple[object, pd.Series]:
    """Fit ARIMAX on training data and return model + residuals."""
    trn = train_df.dropna(subset=[target] + exog_cols)
    if trn.empty:
        raise ValueError("No valid training data for ARIMAX.")
    
    y_trn = trn[target]
    exog_trn = trn[exog_cols].copy()
    
    # Standardize
    train_means = exog_trn.mean()
    train_stds = exog_trn.std(ddof=0).replace(0, 1)
    exog_trn_std = (exog_trn - train_means) / train_stds
    
    # Grid search
    best_order, best_seasonal, _, best_model = grid_search_arimax(
        y_trn, exog_trn_std
    )
    
    resid = pd.Series(best_model.resid, index=trn.index)
    
    return best_model, resid


def run_diagnostic_tests(resid: pd.Series, model_name: str) -> Dict:
    """Run comprehensive diagnostic tests on residuals."""
    resid_clean = resid.dropna()
    if len(resid_clean) == 0:
        return {}
    
    results = {"model": model_name, "n_obs": len(resid_clean)}
    
    # Ljung-Box test for autocorrelation
    try:
        lb = acorr_ljungbox(resid_clean, lags=[4, 8, 12, 24], return_df=True)
        # lb DataFrame has index as lags, columns as lb_stat and lb_pvalue
        results["ljung_box"] = {}
        for lag in [4, 8, 12, 24]:
            if lag in lb.index:
                results["ljung_box"][f"lag_{lag}"] = {
                    "stat": float(lb.loc[lag, "lb_stat"]),
                    "pval": float(lb.loc[lag, "lb_pvalue"]),
                }
            else:
                results["ljung_box"][f"lag_{lag}"] = {"stat": np.nan, "pval": np.nan}
    except Exception as e:
        results["ljung_box"] = {"error": str(e)}
    
    # Jarque-Bera test for normality
    try:
        jb_result = jarque_bera(resid_clean)
        if isinstance(jb_result, tuple) and len(jb_result) >= 2:
            jb_stat = float(jb_result[0])
            jb_pval = float(jb_result[1])
        else:
            jb_stat = np.nan
            jb_pval = np.nan
        results["jarque_bera"] = {"stat": jb_stat, "pval": jb_pval}
    except Exception as e:
        results["jarque_bera"] = {"stat": np.nan, "pval": np.nan, "error": str(e)}
    
    # ARCH test for heteroskedasticity
    try:
        arch_result = het_arch(resid_clean, nlags=4)
        # het_arch returns tuple: (lm_stat, lm_pval, f_stat, f_pval)
        if isinstance(arch_result, tuple) and len(arch_result) >= 2:
            arch_stat = float(arch_result[0])
            arch_pval = float(arch_result[1])
        else:
            arch_stat = np.nan
            arch_pval = np.nan
        results["arch"] = {"stat": arch_stat, "pval": arch_pval}
    except Exception as e:
        results["arch"] = {"stat": np.nan, "pval": np.nan, "error": str(e)}
    
    # Basic statistics
    results["residual_stats"] = {
        "mean": float(resid_clean.mean()),
        "std": float(resid_clean.std()),
        "skew": float(resid_clean.skew()),
        "kurtosis": float(resid_clean.kurtosis()),
    }
    
    return results


def save_diagnostic_plots(
    resid: pd.Series, fitted: pd.Series, model_name: str, output_dir: Path
) -> None:
    """Generate and save diagnostic plots."""
    resid_clean = resid.dropna()
    if len(resid_clean) == 0:
        return
    
    # Time series plot
    plt.figure(figsize=(10, 4))
    plt.plot(resid_clean, color="steelblue", linewidth=0.7)
    plt.title(f"{model_name} Residuals (Time Series)")
    plt.ylabel("Residual")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_residuals_ts.png", dpi=150)
    plt.close()
    
    # Residuals vs Fitted
    if len(fitted) == len(resid_clean):
        plt.figure(figsize=(6, 4))
        plt.scatter(fitted, resid_clean, s=10, color="darkorange", alpha=0.6)
        plt.axhline(0, color="gray", linewidth=1, linestyle="--")
        plt.title(f"{model_name} Residuals vs Fitted")
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_residuals_vs_fitted.png", dpi=150)
        plt.close()
    
    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(resid_clean, bins=40, color="slateblue", edgecolor="white", alpha=0.85)
    plt.title(f"{model_name} Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_residuals_hist.png", dpi=150)
    plt.close()
    
    # ACF/PACF
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(resid_clean, lags=24, ax=ax[0])
    ax[0].set_title(f"{model_name} Residual ACF")
    plot_pacf(resid_clean, lags=24, ax=ax[1], method="ywm")
    ax[1].set_title(f"{model_name} Residual PACF")
    fig.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_residuals_acf_pacf.png", dpi=150)
    plt.close(fig)


def run_comprehensive_diagnostics(
    train_df: pd.DataFrame,
    target: str,
    ols_predictors: List[str],
    arimax_exog: List[str],
    output_dir: Path,
) -> Dict:
    """Run comprehensive diagnostics on training fits."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL DIAGNOSTICS (Training Set)")
    print("=" * 80)
    
    diagnostics = {}
    
    # OLS Diagnostics (standard and extended)
    print("\n1. Fitting OLS (standard) on training data...")
    try:
        ols_model, ols_resid = fit_ols_train(train_df, target, ols_predictors, use_robust=True)
        ols_fitted = pd.Series(ols_model.fittedvalues, index=ols_resid.index)
        
        print(f"   OLS R²: {ols_model.rsquared:.4f}")
        print(f"   OLS AIC: {ols_model.aic:.2f}")
        print("   Using robust (HAC) standard errors")
        print("   Running diagnostic tests...")
        ols_diag = run_diagnostic_tests(ols_resid, "OLS")
        diagnostics["OLS"] = {
            "model_summary": str(ols_model.summary()),
            "diagnostics": ols_diag,
        }
        
        print("   Saving diagnostic plots...")
        save_diagnostic_plots(ols_resid, ols_fitted, "OLS", output_dir)
        
        print("\n   OLS Diagnostic Results:")
        if "ljung_box" in ols_diag and "error" not in ols_diag["ljung_box"]:
            lb_lag8 = ols_diag["ljung_box"].get("lag_8", {})
            print(f"     Ljung-Box (lag 8): stat={lb_lag8.get('stat', np.nan):.2f}, p={lb_lag8.get('pval', np.nan):.4f}")
        else:
            print(f"     Ljung-Box: Error occurred")
        if "jarque_bera" in ols_diag and "error" not in ols_diag["jarque_bera"]:
            print(f"     Jarque-Bera: stat={ols_diag['jarque_bera'].get('stat', np.nan):.2f}, p={ols_diag['jarque_bera'].get('pval', np.nan):.4f}")
        if "arch" in ols_diag and "error" not in ols_diag["arch"]:
            print(f"     ARCH (heteroskedasticity): stat={ols_diag['arch'].get('stat', np.nan):.2f}, p={ols_diag['arch'].get('pval', np.nan):.4f}")
        
    except Exception as e:
        print(f"   ⚠️  OLS diagnostics failed: {e}")
        diagnostics["OLS"] = {"error": str(e)}
    
    # Try extended OLS with lag-8
    print("\n1b. Fitting OLS (extended with lag-8) on training data...")
    try:
        # Check if lag-8 exists in data, create it if needed
        train_df_work = train_df.copy()
        if "cl1_ret_l8" not in train_df_work.columns and target in train_df_work.columns:
            train_df_work["cl1_ret_l8"] = train_df_work[target].shift(8)
            print("   Created cl1_ret_l8 (lag-8) on the fly")
        
        extended_preds = [p for p in ols_predictors + ["cl1_ret_l8"] if p in train_df_work.columns]
        if len(extended_preds) > len(ols_predictors):
            ols_ext_model, ols_ext_resid = fit_ols_train(train_df_work, target, extended_preds, use_robust=True)
            ols_ext_fitted = pd.Series(ols_ext_model.fittedvalues, index=ols_ext_resid.index)
            
            print(f"   OLS Extended R²: {ols_ext_model.rsquared:.4f}")
            print(f"   OLS Extended AIC: {ols_ext_model.aic:.2f}")
            print("   Running diagnostic tests...")
            ols_ext_diag = run_diagnostic_tests(ols_ext_resid, "OLS_Extended")
            diagnostics["OLS_Extended"] = {
                "model_summary": str(ols_ext_model.summary()),
                "diagnostics": ols_ext_diag,
            }
            
            print("\n   OLS Extended Diagnostic Results:")
            if "ljung_box" in ols_ext_diag and "error" not in ols_ext_diag["ljung_box"]:
                lb_lag8 = ols_ext_diag["ljung_box"].get("lag_8", {})
                print(f"     Ljung-Box (lag 8): stat={lb_lag8.get('stat', np.nan):.2f}, p={lb_lag8.get('pval', np.nan):.4f}")
            if "jarque_bera" in ols_ext_diag and "error" not in ols_ext_diag["jarque_bera"]:
                print(f"     Jarque-Bera: stat={ols_ext_diag['jarque_bera'].get('stat', np.nan):.2f}, p={ols_ext_diag['jarque_bera'].get('pval', np.nan):.4f}")
        else:
            print("   ⚠️  cl1_ret_l8 not available in dataset, skipping extended OLS")
    except Exception as e:
        print(f"   ⚠️  Extended OLS diagnostics failed: {e}")
        diagnostics["OLS_Extended"] = {"error": str(e)}
    
    # ARIMAX Diagnostics
    print("\n2. Fitting ARIMAX on training data...")
    try:
        arimax_model, arimax_resid = fit_arimax_train(train_df, target, arimax_exog)
        
        print(f"   ARIMAX AIC: {arimax_model.aic:.2f}")
        print("   Running diagnostic tests...")
        arimax_diag = run_diagnostic_tests(arimax_resid, "ARIMAX")
        diagnostics["ARIMAX"] = {
            "model_summary": str(arimax_model.summary()),
            "diagnostics": arimax_diag,
        }
        
        print("   Saving diagnostic plots...")
        # ARIMAX doesn't have fittedvalues in same way, use zeros as placeholder
        arimax_fitted = pd.Series(0.0, index=arimax_resid.index)
        save_diagnostic_plots(arimax_resid, arimax_fitted, "ARIMAX", output_dir)
        
        print("\n   ARIMAX Diagnostic Results:")
        if "ljung_box" in arimax_diag and "error" not in arimax_diag["ljung_box"]:
            lb_lag8 = arimax_diag["ljung_box"].get("lag_8", {})
            print(f"     Ljung-Box (lag 8): stat={lb_lag8.get('stat', np.nan):.2f}, p={lb_lag8.get('pval', np.nan):.4f}")
        else:
            print(f"     Ljung-Box: Error occurred")
        if "jarque_bera" in arimax_diag and "error" not in arimax_diag["jarque_bera"]:
            print(f"     Jarque-Bera: stat={arimax_diag['jarque_bera'].get('stat', np.nan):.2f}, p={arimax_diag['jarque_bera'].get('pval', np.nan):.4f}")
        if "arch" in arimax_diag and "error" not in arimax_diag["arch"]:
            print(f"     ARCH (heteroskedasticity): stat={arimax_diag['arch'].get('stat', np.nan):.2f}, p={arimax_diag['arch'].get('pval', np.nan):.4f}")
        
    except Exception as e:
        print(f"   ⚠️  ARIMAX diagnostics failed: {e}")
        diagnostics["ARIMAX"] = {"error": str(e)}
    
    # Save diagnostics summary
    diag_file = output_dir / "diagnostics_summary.txt"
    with open(diag_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL DIAGNOSTICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, diag_data in diagnostics.items():
            f.write(f"\n{model_name} Diagnostics:\n")
            f.write("-" * 80 + "\n")
            if "error" in diag_data:
                f.write(f"Error: {diag_data['error']}\n")
            else:
                if "model_summary" in diag_data:
                    f.write(diag_data["model_summary"] + "\n\n")
                if "diagnostics" in diag_data:
                    d = diag_data["diagnostics"]
                    f.write(f"Ljung-Box Tests:\n")
                    if "ljung_box" in d and "error" not in d["ljung_box"]:
                        for lag in [4, 8, 12, 24]:
                            lag_key = f"lag_{lag}"
                            if lag_key in d["ljung_box"]:
                                stat = d["ljung_box"][lag_key].get("stat", np.nan)
                                pval = d["ljung_box"][lag_key].get("pval", np.nan)
                                f.write(f"  Lag {lag}: stat={stat:.2f}, p={pval:.4f}\n")
                            else:
                                f.write(f"  Lag {lag}: N/A\n")
                    else:
                        f.write(f"  Error occurred\n")
                    
                    if "jarque_bera" in d and "error" not in d["jarque_bera"]:
                        jb_stat = d["jarque_bera"].get("stat", np.nan)
                        jb_pval = d["jarque_bera"].get("pval", np.nan)
                        f.write(f"\nJarque-Bera (Normality): stat={jb_stat:.2f}, p={jb_pval:.4f}\n")
                    else:
                        f.write(f"\nJarque-Bera (Normality): Error occurred\n")
                    
                    if "arch" in d and "error" not in d["arch"]:
                        arch_stat = d["arch"].get("stat", np.nan)
                        arch_pval = d["arch"].get("pval", np.nan)
                        f.write(f"ARCH (Heteroskedasticity): stat={arch_stat:.2f}, p={arch_pval:.4f}\n")
                    else:
                        f.write(f"ARCH (Heteroskedasticity): Error occurred\n")
                    
                    if "residual_stats" in d:
                        f.write(f"\nResidual Statistics:\n")
                        f.write(f"  Mean: {d['residual_stats']['mean']:.6f}\n")
                        f.write(f"  Std: {d['residual_stats']['std']:.6f}\n")
                        f.write(f"  Skew: {d['residual_stats']['skew']:.4f}\n")
                        f.write(f"  Kurtosis: {d['residual_stats']['kurtosis']:.4f}\n")
    
    print(f"\nDiagnostics summary saved to: {diag_file.resolve()}")
    
    return diagnostics


def main():
    ensure_output_dir(OUTPUT_DIR)
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL BENCHMARKING")
    print("=" * 80)
    
    # Load data
    df = load_data(DATA_FILE)
    target = "cl1_ret"
    
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset.")
    
    # Train/test split
    split_idx = int((1 - TEST_FRAC) * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    y_test = test_df[target]
    
    print(f"\nDataset: {len(df):,} total observations")
    print(f"Train: {len(train_df):,} observations ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Test:  {len(test_df):,} observations ({test_df.index.min().date()} to {test_df.index.max().date()})")
    print(f"\nTarget variable: {target}")
    print(f"OLS predictors: {OLS_PREDICTORS}")
    print(f"ARIMAX exogenous: {ARIMAX_EXOG}")
    
    # Generate forecasts
    print("\n" + "-" * 80)
    print("Generating forecasts...")
    print("-" * 80)
    
    print("1. Naive forecast (always 0)...")
    naive_pred = naive_forecast(y_test)
    
    print("2. AR(1) forecast (expanding window)...")
    ar1_pred = expanding_ar1_forecast(df, target, split_idx)
    
    print("3. OLS forecast (expanding window)...")
    ols_pred = expanding_ols_forecast(df, target, OLS_PREDICTORS, split_idx)
    
    print("4. ARIMAX forecast (fit on training, forecast test)...")
    arimax_pred, arimax_info = arimax_forecast(
        train_df, test_df, target, ARIMAX_EXOG
    )
    
    # Align all predictions
    valid_idx = (
        y_test.dropna().index
        .intersection(naive_pred.dropna().index)
        .intersection(ar1_pred.dropna().index)
        .intersection(ols_pred.dropna().index)
        .intersection(arimax_pred.dropna().index)
    )
    
    if len(valid_idx) == 0:
        print("⚠️  No valid predictions after alignment. Check data.")
        return
    
    y_test_aligned = y_test.loc[valid_idx]
    naive_aligned = naive_pred.loc[valid_idx]
    ar1_aligned = ar1_pred.loc[valid_idx]
    ols_aligned = ols_pred.loc[valid_idx]
    arimax_aligned = arimax_pred.loc[valid_idx]
    
    # Compute metrics for all models
    print("\n" + "-" * 80)
    print("Computing metrics...")
    print("-" * 80)
    
    results = []
    for name, pred in [
        ("Naive", naive_aligned),
        ("AR(1)", ar1_aligned),
        ("OLS(final)", ols_aligned),
        ("ARIMAX", arimax_aligned),
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
    
    # Print results
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE PERFORMANCE COMPARISON")
    print("=" * 80)
    print(results_df.round(4).to_string())
    
    # Sort by RMSE
    print("\n" + "-" * 80)
    print("Ranked by RMSE (lower is better):")
    print("-" * 80)
    print(results_df.sort_values("RMSE").round(4).to_string())
    
    # Save results
    results_file = OUTPUT_DIR / "benchmark_results.csv"
    results_df.to_csv(results_file)
    print(f"\nSaved results to: {results_file.resolve()}")
    
    # Save forecasts
    forecasts_df = pd.DataFrame({
        "y_true": y_test_aligned,
        "naive": naive_aligned,
        "ar1": ar1_aligned,
        "ols": ols_aligned,
        "arimax": arimax_aligned,
    })
    forecasts_file = OUTPUT_DIR / "all_forecasts.csv"
    forecasts_df.to_csv(forecasts_file)
    print(f"Saved forecasts to: {forecasts_file.resolve()}")
    
    # ARIMAX model info summary
    if arimax_info:
        print("\n" + "-" * 80)
        print("ARIMAX Model Summary:")
        print("-" * 80)
        print(f"Selected order: {arimax_info['order']} × {arimax_info['seasonal_order']}")
        print(f"AIC: {arimax_info['aic']:.2f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
    best_sharpe = results_df.loc[results_df["Sharpe(approx)"].idxmax()]
    print(f"Best RMSE: {best_rmse.name} ({best_rmse['RMSE']:.4f})")
    print(f"Best Sharpe: {best_sharpe.name} ({best_sharpe['Sharpe(approx)']:.4f})")
    print(f"Best Hit Rate: {results_df.loc[results_df['HitRate'].idxmax()].name} ({results_df['HitRate'].max():.2%})")
    
    # Run comprehensive diagnostics on training set
    diagnostics = run_comprehensive_diagnostics(
        train_df, target, OLS_PREDICTORS, ARIMAX_EXOG, OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARKING AND DIAGNOSTICS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("  - benchmark_results.csv: Performance metrics")
    print("  - all_forecasts.csv: Individual forecasts")
    print("  - diagnostics_summary.txt: Diagnostic test results")
    print("  - Diagnostic plots: *_residuals_*.png")


if __name__ == "__main__":
    main()


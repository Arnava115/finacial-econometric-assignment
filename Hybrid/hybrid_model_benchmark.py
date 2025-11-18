"""
Hybrid Model Benchmarking: Combining Best of Both Approaches

This script integrates:
1. Improved 7-variable OLS model with squared lagged returns (from Diagnostics.ipynb)
2. Expanding window forecasting methodology (from model_benchmark.py)
3. Comprehensive diagnostic tests (from Diagnostics.ipynb)
4. Seasonal ARIMAX with sophisticated grid search (from model_benchmark.py)
5. P&L metrics for financial performance (from model_benchmark.py)

Run:
    python hybrid_model_benchmark.py
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, het_arch, het_breuschpagan, 
    linear_reset, acorr_breusch_godfrey
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import shapiro, f
from scipy import stats

# ==================== CONFIGURATION ====================

# Data file - try Updated folder first, then root
DATA_FILE_OPTIONS = [
    Path("../Updated/Oil_Model_Weekly_Core.csv"),
    Path("../Oil_Model_Weekly_Core.csv"),
    Path("Oil_Model_Weekly_Core.csv"),
]

OUTPUT_DIR = Path("hybrid_results")
TEST_FRAC = 0.20

# IMPROVED OLS PREDICTORS (from Diagnostics.ipynb - 7 variables with squared term)
OLS_PREDICTORS = [
    'dow_jones_us_oilandgas_index_ret',
    'exxonmobil_ret',
    'cl1_ret_l2',
    'weekly_commercial_short_diff',
    'weekly_commercial_long_diff',
    'weekly_change_in_crude_stock_diff',
    # Note: cl1_ret_l2_squared will be created on the fly
]

# ARIMAX uses same exogenous variables
ARIMAX_EXOG = OLS_PREDICTORS.copy()

# ARIMAX grid search configuration (OPTIMIZED for speed)
# Reduced grid to focus on most common and effective orders
NON_SEASONAL_GRID = [
    (0, 0), (1, 0), (0, 1), (1, 1),  # Basic orders
    (2, 0), (0, 2), (2, 1), (1, 2), (2, 2),  # Common higher orders
    # Removed (3,0), (0,3), (3,1), (1,3) to speed up
]

# Simplified seasonal configs - focus on most promising periods
SEASONAL_CONFIGS = [
    {"period": 7, "orders": [(1, 0), (0, 1), (1, 1)]},  # Weekly seasonality
    {"period": 13, "orders": [(1, 0), (0, 1), (1, 1)]},  # Quarterly seasonality
    # Removed period 8 and higher order seasonal terms to speed up
]

# Grid search frequency: only every N steps (higher = faster but less adaptive)
# Set to None to do grid search only once at the beginning (fastest)
ARIMAX_GRID_SEARCH_FREQUENCY = 50  # Changed from 20 to 50 for speed
# Alternative: Set to None for single grid search: ARIMAX_GRID_SEARCH_FREQUENCY = None

# ==================== UTILITY FUNCTIONS ====================

def ensure_output_dir(path: Path) -> None:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def find_data_file() -> Path:
    """Find the data file in available locations."""
    for path in DATA_FILE_OPTIONS:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Data file not found. Tried: {[str(p) for p in DATA_FILE_OPTIONS]}"
    )


def load_data(path: Path) -> pd.DataFrame:
    """Load and sort data."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def prepare_improved_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare predictors including squared lagged returns term.
    This addresses the functional form misspecification identified by RESET test.
    """
    df_work = df.copy()
    
    # Create squared lagged returns if cl1_ret_l2 exists
    if 'cl1_ret_l2' in df_work.columns:
        df_work['cl1_ret_l2_squared'] = df_work['cl1_ret_l2'] ** 2
    
    return df_work


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, HitRate, TheilU, R²."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    hit_rate = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    rmse_naive = float(np.sqrt(np.mean(y_true ** 2)))
    theil_u = rmse / rmse_naive if rmse_naive > 0 else np.nan
    
    # R² score
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "HitRate": hit_rate,
        "TheilU": theil_u,
        "R2": r2,
    }


def compute_pnl(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute P&L metrics: long if forecast > 0, else flat."""
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


# ==================== FORECASTING FUNCTIONS ====================

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
    df: pd.DataFrame, target: str, predictors: List[str], split_idx: int, use_robust: bool = True
) -> pd.Series:
    """
    Expanding-window OLS forecast with improved 7-variable model.
    Refits model on [0:t) to predict t, for t in [split_idx .. end).
    """
    # Prepare data with squared term
    df_work = prepare_improved_predictors(df)
    
    # Add squared term to predictors if not already present
    predictors_final = predictors.copy()
    if 'cl1_ret_l2_squared' not in predictors_final and 'cl1_ret_l2' in df_work.columns:
        predictors_final.append('cl1_ret_l2_squared')
    
    preds = []
    idxs = []
    
    for t in range(split_idx, len(df_work)):
        train = df_work.iloc[:t]
        test_row = df_work.iloc[[t]]
        
        trn = train.dropna(subset=[target] + predictors_final)
        if trn.empty or len(trn) < len(predictors_final) + 2:
            preds.append(np.nan)
            idxs.append(df_work.index[t])
            continue
        
        # Standardize predictors using training sample only
        y_trn = trn[target].values
        X_trn = trn[predictors_final].copy()
        train_means = X_trn.mean()
        train_stds = X_trn.std(ddof=0).replace(0, 1)
        X_trn_std = (X_trn - train_means) / train_stds
        X_trn_const = sm.add_constant(X_trn_std.values, has_constant='add')
        
        try:
            mdl = sm.OLS(y_trn, X_trn_const).fit()
            # Use robust standard errors if requested
            if use_robust:
                mdl = mdl.get_robustcov_results(cov_type='HAC', maxlags=12)
        except:
            preds.append(np.nan)
            idxs.append(df_work.index[t])
            continue
        
        # Prepare test data (standardize using training stats)
        X_test = test_row[predictors_final].copy()
        X_test = X_test.fillna(train_means)
        X_test_std = (X_test - train_means) / train_stds
        X_test_const = sm.add_constant(X_test_std.values, has_constant='add')
        
        try:
            pred = float(mdl.predict(X_test_const)[0])
        except:
            pred = np.nan
        
        preds.append(pred)
        idxs.append(df_work.index[t])
    
    return pd.Series(preds, index=idxs, name="ols_pred")


def grid_search_arimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]], float, SARIMAX]:
    """
    Grid search for best ARIMAX order with seasonal components.
    Returns best non-seasonal order, seasonal order, AIC, and fitted model.
    """
    best_order = None
    best_seasonal_order = None
    best_aic = np.inf
    best_model = None

    # Try non-seasonal first
    for p, q in NON_SEASONAL_GRID:
        order = (p, 0, q)
        try:
            model = SARIMAX(
                y_train,
                order=order,
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_seasonal_order = None
                best_model = results
        except Exception:
            continue

    # Try seasonal configurations
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


def expanding_arimax_forecast(
    df: pd.DataFrame,
    target: str,
    exog_cols: List[str],
    split_idx: int,
) -> Tuple[pd.Series, Dict]:
    """
    Expanding-window ARIMAX forecast.
    Refits ARIMAX on [0:t) to predict t, for t in [split_idx .. end).
    """
    # Prepare data with squared term
    df_work = prepare_improved_predictors(df)
    
    # Add squared term to exog if needed
    exog_cols_final = exog_cols.copy()
    if 'cl1_ret_l2_squared' not in exog_cols_final and 'cl1_ret_l2' in df_work.columns:
        exog_cols_final.append('cl1_ret_l2_squared')
    
    preds = []
    idxs = []
    model_info_list = []
    
    print(f"   Generating ARIMAX forecasts (expanding window)...")
    print(f"   This may take a while...")
    
    for t in range(split_idx, len(df_work)):
        train = df_work.iloc[:t]
        test_row = df_work.iloc[[t]]
        
        trn = train.dropna(subset=[target] + exog_cols_final)
        if trn.empty or len(trn) < len(exog_cols_final) + 10:
            preds.append(np.nan)
            idxs.append(df_work.index[t])
            continue
        
        y_trn = trn[target]
        exog_trn = trn[exog_cols_final].copy()
        
        # Standardize exog using training sample
        train_means = exog_trn.mean()
        train_stds = exog_trn.std(ddof=0).replace(0, 1)
        exog_trn_std = (exog_trn - train_means) / train_stds
        
        try:
            # Grid search (only on first iteration or periodically to save time)
            # Use ARIMAX_GRID_SEARCH_FREQUENCY to control how often we re-optimize
            # If None, only do grid search once at the beginning
            do_grid_search = (
                t == split_idx or 
                (ARIMAX_GRID_SEARCH_FREQUENCY is not None and 
                 (t - split_idx) % ARIMAX_GRID_SEARCH_FREQUENCY == 0)
            )
            
            if do_grid_search:
                if t == split_idx:
                    print(f"   Running initial ARIMAX grid search (this may take 1-2 minutes)...")
                best_order, best_seasonal, best_aic, best_model = grid_search_arimax(
                    y_trn, exog_trn_std
                )
                model_info_list.append({
                    'index': t,
                    'order': best_order,
                    'seasonal_order': best_seasonal,
                    'aic': best_aic,
                })
                if t == split_idx:
                    print(f"   Initial ARIMAX order: {best_order}, seasonal: {best_seasonal}, AIC: {best_aic:.2f}")
            else:
                # Reuse previous order for speed
                if len(model_info_list) > 0:
                    prev_info = model_info_list[-1]
                    best_order = prev_info['order']
                    best_seasonal = prev_info['seasonal_order']
                else:
                    best_order, best_seasonal, best_aic, best_model = grid_search_arimax(
                        y_trn, exog_trn_std
                    )
                
                # Fit with previous order
                if best_seasonal is not None:
                    model = SARIMAX(
                        y_trn,
                        order=best_order,
                        seasonal_order=best_seasonal,
                        exog=exog_trn_std,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                else:
                    model = SARIMAX(
                        y_trn,
                        order=best_order,
                        exog=exog_trn_std,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                best_model = model.fit(disp=False)
            
            # Prepare test exog
            exog_test = test_row[exog_cols_final].copy()
            exog_test = exog_test.fillna(train_means)
            exog_test_std = (exog_test - train_means) / train_stds
            
            # Forecast one step ahead
            forecast = best_model.get_forecast(steps=1, exog=exog_test_std)
            pred = float(forecast.predicted_mean.iloc[0])
            
        except Exception as e:
            pred = np.nan
        
        preds.append(pred)
        idxs.append(df_work.index[t])
        
        if (t - split_idx + 1) % 50 == 0:
            print(f"   Completed {t - split_idx + 1} / {len(df_work) - split_idx} forecasts")
    
    pred_series = pd.Series(preds, index=idxs, name="arimax_pred")
    
    # Get final model info
    final_model_info = {
        "order": model_info_list[-1]['order'] if model_info_list else None,
        "seasonal_order": model_info_list[-1]['seasonal_order'] if model_info_list else None,
        "aic": model_info_list[-1]['aic'] if model_info_list else None,
    }
    
    return pred_series, final_model_info


# ==================== DIAGNOSTIC FUNCTIONS ====================

def run_comprehensive_diagnostics(
    train_df: pd.DataFrame,
    target: str,
    predictors: List[str],
    output_dir: Path,
) -> Dict:
    """
    Run comprehensive diagnostic tests on training set OLS model.
    Includes: RESET, Breusch-Godfrey, Breusch-Pagan, Jarque-Bera, VIF, Chow test.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL DIAGNOSTICS (Training Set)")
    print("=" * 80)
    
    # Prepare data
    df_work = prepare_improved_predictors(train_df)
    predictors_final = predictors.copy()
    if 'cl1_ret_l2_squared' not in predictors_final and 'cl1_ret_l2' in df_work.columns:
        predictors_final.append('cl1_ret_l2_squared')
    
    trn = df_work.dropna(subset=[target] + predictors_final)
    if trn.empty:
        print("⚠️  No valid training data for diagnostics.")
        return {}
    
    y_trn = trn[target].values
    X_trn = trn[predictors_final].copy()
    
    # Standardize
    train_means = X_trn.mean()
    train_stds = X_trn.std(ddof=0).replace(0, 1)
    X_trn_std = (X_trn - train_means) / train_stds
    X_const = sm.add_constant(X_trn_std.values, has_constant='add')
    
    # Fit model
    model = sm.OLS(y_trn, X_const).fit()
    model_hac = model.get_robustcov_results(cov_type='HAC', maxlags=12)
    residuals = pd.Series(model.resid, index=trn.index)
    
    diagnostics = {}
    
    # 1. Ramsey RESET Test
    print("\n1. RAMSEY RESET TEST (Functional Form)")
    print("-" * 80)
    try:
        reset_result = linear_reset(model)
        diagnostics['reset'] = {
            'statistic': float(reset_result.statistic),
            'pvalue': float(reset_result.pvalue),
        }
        print(f"   Test Statistic: {reset_result.statistic:.6f}")
        print(f"   P-value: {reset_result.pvalue:.6f}")
        if reset_result.pvalue > 0.05:
            print("   ✓ PASS - Functional form is appropriate")
        else:
            print("   ✗ FAIL - Functional form misspecification detected")
    except Exception as e:
        print(f"   ⚠️  RESET test failed: {e}")
        diagnostics['reset'] = {'error': str(e)}
    
    # 2. Breusch-Godfrey Test (Autocorrelation)
    print("\n2. BREUSCH-GODFREY TEST (Autocorrelation)")
    print("-" * 80)
    try:
        bg_result = acorr_breusch_godfrey(model, nlags=10)
        diagnostics['breusch_godfrey'] = {
            'statistic': float(bg_result[0]),
            'pvalue': float(bg_result[1]),
        }
        print(f"   Test Statistic: {bg_result[0]:.6f}")
        print(f"   P-value: {bg_result[1]:.6f}")
        if bg_result[1] > 0.05:
            print("   ✓ PASS - No autocorrelation detected")
        else:
            print("   ✗ FAIL - Autocorrelation detected (using HAC robust SE)")
    except Exception as e:
        print(f"   ⚠️  Breusch-Godfrey test failed: {e}")
        diagnostics['breusch_godfrey'] = {'error': str(e)}
    
    # 3. Breusch-Pagan Test (Heteroskedasticity)
    print("\n3. BREUSCH-PAGAN TEST (Heteroskedasticity)")
    print("-" * 80)
    try:
        bp_result = het_breuschpagan(residuals, X_const)
        diagnostics['breusch_pagan'] = {
            'lm_statistic': float(bp_result[0]),
            'lm_pvalue': float(bp_result[1]),
            'f_statistic': float(bp_result[2]),
            'f_pvalue': float(bp_result[3]),
        }
        print(f"   LM Statistic: {bp_result[0]:.6f}")
        print(f"   LM P-value: {bp_result[1]:.6f}")
        if bp_result[1] > 0.05:
            print("   ✓ PASS - Homoskedasticity")
        else:
            print("   ✗ FAIL - Heteroskedasticity detected (using HAC robust SE)")
    except Exception as e:
        print(f"   ⚠️  Breusch-Pagan test failed: {e}")
        diagnostics['breusch_pagan'] = {'error': str(e)}
    
    # 4. Jarque-Bera Test (Normality)
    print("\n4. JARQUE-BERA TEST (Normality)")
    print("-" * 80)
    try:
        jb_result = jarque_bera(residuals)
        diagnostics['jarque_bera'] = {
            'statistic': float(jb_result[0]),
            'pvalue': float(jb_result[1]),
        }
        print(f"   Test Statistic: {jb_result[0]:.6f}")
        print(f"   P-value: {jb_result[1]:.6f}")
        print(f"   Skewness: {residuals.skew():.6f}")
        print(f"   Kurtosis: {residuals.kurtosis():.6f}")
        if jb_result[1] > 0.05:
            print("   ✓ PASS - Residuals approximately normal")
        else:
            print("   ✗ FAIL - Non-normal residuals (common in financial data)")
    except Exception as e:
        print(f"   ⚠️  Jarque-Bera test failed: {e}")
        diagnostics['jarque_bera'] = {'error': str(e)}
    
    # 5. VIF (Multicollinearity)
    print("\n5. VARIANCE INFLATION FACTOR (Multicollinearity)")
    print("-" * 80)
    try:
        vif_data = []
        for i, col in enumerate(predictors_final):
            vif_val = variance_inflation_factor(X_trn_std.values, i)
            vif_data.append({'variable': col, 'vif': float(vif_val)})
        
        diagnostics['vif'] = vif_data
        print(f"{'Variable':<45} {'VIF':<10} {'Flag'}")
        print("-" * 70)
        for item in vif_data:
            v = item['vif']
            if v <= 5:
                flag = "✓ OK"
            elif v <= 10:
                flag = "⚠ Moderate"
            else:
                flag = "✗ High"
            print(f"{item['variable']:<45} {v:<10.2f} {flag}")
        
        max_vif = max([item['vif'] for item in vif_data])
        if max_vif <= 10:
            print("\n   ✓ PASS - No severe multicollinearity")
        else:
            print("\n   ✗ FAIL - High multicollinearity detected")
    except Exception as e:
        print(f"   ⚠️  VIF calculation failed: {e}")
        diagnostics['vif'] = {'error': str(e)}
    
    # 6. Chow Test (Structural Breaks)
    print("\n6. CHOW TEST (Structural Breaks)")
    print("-" * 80)
    try:
        breakpoint = int(len(trn) * 0.7)
        y1, X1 = y_trn[:breakpoint], X_const[:breakpoint]
        y2, X2 = y_trn[breakpoint:], X_const[breakpoint:]
        
        model1 = sm.OLS(y1, X1).fit()
        model2 = sm.OLS(y2, X2).fit()
        
        RSS_pooled = model.ssr
        RSS_1 = model1.ssr
        RSS_2 = model2.ssr
        
        k = X_const.shape[1]
        n1 = len(y1)
        n2 = len(y2)
        
        F_chow = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n1 + n2 - 2*k))
        p_chow = 1 - f.cdf(F_chow, k, (n1 + n2 - 2*k))
        
        diagnostics['chow'] = {
            'statistic': float(F_chow),
            'pvalue': float(p_chow),
        }
        print(f"   F-statistic: {F_chow:.6f}")
        print(f"   P-value: {p_chow:.6f}")
        if p_chow > 0.05:
            print("   ✓ PASS - No structural break detected")
        else:
            print("   ✗ FAIL - Structural break detected")
    except Exception as e:
        print(f"   ⚠️  Chow test failed: {e}")
        diagnostics['chow'] = {'error': str(e)}
    
    # Save diagnostics summary
    diag_file = output_dir / "diagnostics_summary.txt"
    with open(diag_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL DIAGNOSTICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: OLS with {len(predictors_final)} predictors\n")
        f.write(f"Sample size: {len(trn)}\n\n")
        
        for test_name, test_result in diagnostics.items():
            f.write(f"\n{test_name.upper()}:\n")
            f.write("-" * 80 + "\n")
            if isinstance(test_result, dict):
                if 'error' in test_result:
                    f.write(f"Error: {test_result['error']}\n")
                else:
                    for key, value in test_result.items():
                        if isinstance(value, list):
                            f.write(f"{key}:\n")
                            for item in value:
                                f.write(f"  {item}\n")
                        else:
                            f.write(f"{key}: {value}\n")
    
    print(f"\nDiagnostics summary saved to: {diag_file.resolve()}")
    
    return diagnostics


# ==================== MAIN FUNCTION ====================

def main():
    ensure_output_dir(OUTPUT_DIR)
    
    print("=" * 80)
    print("HYBRID MODEL BENCHMARKING")
    print("Combining Best of Both Approaches")
    print("=" * 80)
    
    # Load data
    data_file = find_data_file()
    print(f"\nLoading data from: {data_file}")
    df = load_data(data_file)
    target = "cl1_ret"
    
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset.")
    
    # Prepare data with improved predictors
    df = prepare_improved_predictors(df)
    
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
    print(f"  + cl1_ret_l2_squared (created automatically)")
    
    # Run comprehensive diagnostics on training set
    diagnostics = run_comprehensive_diagnostics(
        train_df, target, OLS_PREDICTORS, OUTPUT_DIR
    )
    
    # Generate forecasts
    print("\n" + "-" * 80)
    print("Generating forecasts (expanding window)...")
    print("-" * 80)
    
    print("1. Naive forecast (always 0)...")
    naive_pred = naive_forecast(y_test)
    
    print("2. AR(1) forecast (expanding window)...")
    ar1_pred = expanding_ar1_forecast(df, target, split_idx)
    
    print("3. OLS forecast (expanding window with improved model)...")
    ols_pred = expanding_ols_forecast(df, target, OLS_PREDICTORS, split_idx, use_robust=True)
    
    print("4. ARIMAX forecast (expanding window)...")
    arimax_pred, arimax_info = expanding_arimax_forecast(
        df, target, ARIMAX_EXOG, split_idx
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
        ("OLS(Improved)", ols_aligned),
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
    
    # ARIMAX model info
    if arimax_info:
        print("\n" + "-" * 80)
        print("ARIMAX Model Summary:")
        print("-" * 80)
        print(f"Selected order: {arimax_info['order']}")
        if arimax_info['seasonal_order']:
            print(f"Seasonal order: {arimax_info['seasonal_order']}")
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
    
    print("\n" + "=" * 80)
    print("HYBRID BENCHMARKING COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("  - benchmark_results.csv: Performance metrics")
    print("  - all_forecasts.csv: Individual forecasts")
    print("  - diagnostics_summary.txt: Comprehensive diagnostic test results")


if __name__ == "__main__":
    main()


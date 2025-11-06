# STEP 1: Imports
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt

# Try to import BaiPerron (may not be available in all linearmodels versions)
try:
    from linearmodels.breaks import BaiPerron
    BAI_PERRON_AVAILABLE = True
except ImportError:
    BAI_PERRON_AVAILABLE = False
    print("Note: BaiPerron test not available. Will use alternative breakpoint detection methods.")

# STEP 2: Load data
DATA_FILE = Path("Oil_Model_Weekly_Core.csv")  # Model-ready dataset

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file {DATA_FILE} not found. "
        "Please run process_oil_data.py first to generate the data file."
    )

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
df = df.sort_index()

print(f"Loaded data: {len(df)} observations")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nAvailable columns: {list(df.columns)}")

# STEP 3: Prepare data
target = 'cl1_ret'  # Dependent variable: oil returns

# Select predictors (you can customize this based on your model)
# Option 1: Use AR lags only
predictor_cols = [col for col in df.columns if col.startswith('cl1_ret_l')]

# Option 2: Use AR lags + key cross-asset returns (uncomment to use)
# predictor_cols = [col for col in df.columns if col.startswith('cl1_ret_l')]
# additional_predictors = ['brent_ret', 'spx_ret']  # Add more as needed
# predictor_cols.extend([p for p in additional_predictors if p in df.columns])

# Option 3: Use all predictors except target (uncomment to use)
# predictor_cols = [col for col in df.columns if col != target]

if not predictor_cols:
    raise ValueError("No predictor columns found. Check your data file.")

print(f"\nTarget variable: {target}")
print(f"Number of predictors: {len(predictor_cols)}")
print(f"Predictors: {predictor_cols}")

# Align data and drop any remaining NaN
data_aligned = df[[target] + predictor_cols].dropna()
Y = data_aligned[target]
X_data = data_aligned[predictor_cols]
X = sm.add_constant(X_data)  # Add constant term

print(f"\nFinal sample size after dropping NaN: {len(Y)} observations")

# STEP 4: Fit baseline OLS
ols_model = sm.OLS(Y, X).fit()
print("\n" + "="*70)
print("BASELINE OLS MODEL")
print("="*70)
print(ols_model.summary())

# STEP 5: CUSUM test for parameter stability
cusum_stat, cusum_pvalue, _ = breaks_cusumolsresid(ols_model.resid, ddof=len(X.columns))
print("\n" + "="*70)
print("CUSUM TEST FOR PARAMETER STABILITY")
print("="*70)
print(f"CUSUM Test p-value: {cusum_pvalue:.4f}")
if cusum_pvalue < 0.05:
    print("⚠️  Possible structural instability detected (reject H0)")
else:
    print("✅ No significant structural break detected (fail to reject H0)")

# STEP 6: Breakpoint detection tests
print("\n" + "="*70)
print("BREAKPOINT DETECTION TESTS")
print("="*70)

if BAI_PERRON_AVAILABLE:
    # Bai–Perron multiple breakpoint test (detects unknown breakpoints)
    print("\nBai-Perron Multiple Breakpoint Test:")
    try:
        bp = BaiPerron(Y, X, lags=0)
        bp_fitted = bp.fit()
        print(f"Detected breakpoints: {bp_fitted.breaks}")
        if len(bp_fitted.breaks) > 0:
            print(f"Breakpoint dates: {[data_aligned.index[int(b)] for b in bp_fitted.breaks]}")
        bp_fitted.plot()
        plt.tight_layout()
        plt.savefig('bai_perron_breakpoints.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'bai_perron_breakpoints.png'")
        plt.show()
    except Exception as e:
        print(f"Error running Bai-Perron test: {e}")
        print("This may occur if the data structure doesn't meet the test requirements.")
else:
    # Alternative: Test multiple potential breakpoints using CUSUM
    print("\nMultiple Breakpoint Test (Alternative Method):")
    try:
        # Test multiple potential breakpoints
        n = len(Y)
        test_dates = []
        test_stats = []
        test_pvals = []
        
        # Test breakpoints at 25%, 33%, 50%, 67%, 75% of sample
        breakpoint_indices = [int(n * p) for p in [0.25, 0.33, 0.50, 0.67, 0.75]]
        
        for idx in breakpoint_indices:
            if idx < 20 or idx > n - 20:  # Need enough observations on each side
                continue
                
            # Split data
            y_before = Y.iloc[:idx]
            x_before = X.iloc[:idx]
            y_after = Y.iloc[idx:]
            x_after = X.iloc[idx:]
            
            # Fit models
            model_before = sm.OLS(y_before, x_before).fit()
            model_after = sm.OLS(y_after, x_after).fit()
            
            # CUSUM test on residuals
            cusum_stat, cusum_pval, _ = breaks_cusumolsresid(
                pd.concat([model_before.resid, model_after.resid]), 
                ddof=len(X.columns)
            )
            
            test_dates.append(data_aligned.index[idx])
            test_stats.append(cusum_stat)
            test_pvals.append(cusum_pval)
        
        if test_dates:
            results_df = pd.DataFrame({
                'Breakpoint Date': test_dates,
                'CUSUM Statistic': test_stats,
                'P-value': test_pvals
            })
            print("\nBreakpoint Test Results:")
            print(results_df.to_string(index=False))
            
            # Find most significant breakpoint
            min_p_idx = np.argmin(test_pvals)
            print(f"\nMost significant potential breakpoint:")
            print(f"  Date: {test_dates[min_p_idx]}")
            print(f"  CUSUM p-value: {test_pvals[min_p_idx]:.4f}")
    except Exception as e:
        print(f"Error running alternative breakpoint tests: {e}")

# STEP 7: Visualize rolling coefficients
print("\n" + "="*70)
print("ROLLING COEFFICIENT ANALYSIS")
print("="*70)
window = 52  # weekly data = 1-year rolling window

# Calculate rolling coefficients for the first predictor
if len(predictor_cols) > 0:
    first_pred = predictor_cols[0]
    print(f"\nCalculating rolling coefficients for: {first_pred}")
    
    rolling_betas = []
    rolling_dates = []
    
    for i in range(window, len(data_aligned)):
        window_data = data_aligned.iloc[i-window:i]
        y_window = window_data[target]
        x_window = sm.add_constant(window_data[[first_pred]])
        
        try:
            model_window = sm.OLS(y_window, x_window).fit()
            beta = model_window.params[first_pred]
            rolling_betas.append(beta)
            rolling_dates.append(data_aligned.index[i])
        except:
            rolling_betas.append(np.nan)
            rolling_dates.append(data_aligned.index[i])
    
    rolling_series = pd.Series(rolling_betas, index=rolling_dates, name=f'Rolling Beta ({first_pred})')
    
    plt.figure(figsize=(12, 5))
    rolling_series.plot(title=f"Rolling Beta ({first_pred}) — Detect Structural Drifts", 
                        color='steelblue', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel('Coefficient Value')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_coefficients.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'rolling_coefficients.png'")
    plt.show()
    
    print(f"\nRolling coefficient statistics:")
    print(f"  Mean: {rolling_series.mean():.4f}")
    print(f"  Std:  {rolling_series.std():.4f}")
    print(f"  Min:  {rolling_series.min():.4f}")
    print(f"  Max:  {rolling_series.max():.4f}")
else:
    print("No predictors available for rolling coefficient analysis.")

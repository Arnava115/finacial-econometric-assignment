# Comparison: Updated Notebooks vs Original model_benchmark.py

## Executive Summary

The updated notebooks (`Diagnostics.ipynb` and `model_selection.ipynb`) represent a **significant methodological improvement** over the original `model_benchmark.py`, particularly in model specification and diagnostic rigor. However, they differ in forecasting methodology and variable selection.

---

## Key Differences

### 1. **Variable Selection**

| Aspect | model_benchmark.py | Updated Notebooks |
|--------|-------------------|-------------------|
| **OLS Predictors** | 2 variables:<br>- `cl1_ret_l4`<br>- `weekly_change_in_crude_stock_diff` | 7 variables:<br>- `dow_jones_us_oilandgas_index_ret`<br>- `exxonmobil_ret`<br>- `cl1_ret_l2`<br>- `weekly_commercial_short_diff`<br>- `weekly_commercial_long_diff`<br>- `weekly_change_in_crude_stock_diff`<br>- `cl1_ret_l2_squared` ⭐ |
| **Rationale** | Minimalist approach (2 variables) | Comprehensive + non-linear term |
| **Key Addition** | None | **Squared lagged returns** (fixes RESET test) |

**Assessment**: ✅ Updated notebooks use a more comprehensive variable set with **non-linear term** that addresses functional form misspecification.

---

### 2. **Model Specification & Diagnostics**

| Aspect | model_benchmark.py | Updated Notebooks |
|--------|-------------------|-------------------|
| **Functional Form Test** | ❌ Not performed | ✅ **Ramsey RESET test** - identifies misspecification |
| **Fix Applied** | None | ✅ Added `cl1_ret_l2_squared` term |
| **RESET p-value** | N/A | Original: 0.0164 (FAIL)<br>Improved: 0.4692 (PASS) |
| **Autocorrelation Tests** | ✅ Ljung-Box (lags 4,8,12,24) | ✅ Breusch-Godfrey (multiple lags) |
| **Heteroskedasticity** | ✅ ARCH test | ✅ Breusch-Pagan + White tests |
| **Normality** | ✅ Jarque-Bera | ✅ Jarque-Bera + Shapiro-Wilk |
| **Multicollinearity** | ❌ Not tested | ✅ **VIF analysis** (all < 5) |
| **Structural Breaks** | ❌ Not tested | ✅ **Chow test** (70% split) |
| **Robust Inference** | ✅ HAC (Newey-West) | ✅ HAC (Newey-West) |

**Assessment**: ✅ Updated notebooks are **significantly more comprehensive** in diagnostics, especially:
- **RESET test** identifies and fixes functional form issues
- **VIF** ensures no multicollinearity problems
- **Chow test** checks for structural breaks

---

### 3. **Forecasting Methodology**

| Aspect | model_benchmark.py | Updated Notebooks |
|--------|-------------------|-------------------|
| **Forecast Type** | ✅ **Expanding window**<br>(refit model at each step) | ❌ **Static forecast**<br>(fit once, forecast all) |
| **Train/Test Split** | 80/20 split | Fixed date split (2018) |
| **Real-world Validity** | ✅ More realistic (simulates real-time forecasting) | ⚠️ Less realistic (uses all training data) |
| **Computational Cost** | Higher (refits many times) | Lower (fits once) |

**Assessment**: ⚠️ **model_benchmark.py has superior forecasting methodology** - expanding window is more appropriate for time series forecasting and avoids look-ahead bias.

---

### 4. **ARIMA/ARIMAX Implementation**

| Aspect | model_benchmark.py | Updated Notebooks |
|--------|-------------------|-------------------|
| **ARIMA Grid Search** | ✅ Comprehensive:<br>- Non-seasonal: (0,0) to (3,3)<br>- **Seasonal components**: periods 7, 8, 13 | ⚠️ Simpler:<br>- Non-seasonal: (0,0) to (3,2)<br>- **No seasonal components** |
| **Stationarity Test** | ❌ Not shown | ✅ **ADF test** (confirms d=0) |
| **Model Selection** | AIC-based | AIC-based |
| **ARIMAX Exogenous** | Same as OLS (2 vars) | All 7 variables |

**Assessment**: 
- ✅ Updated notebooks: Better **diagnostics** (ADF test)
- ⚠️ model_benchmark.py: More **sophisticated ARIMAX** (seasonal components)

---

### 5. **Performance Metrics**

| Metric | model_benchmark.py | Updated Notebooks |
|--------|-------------------|-------------------|
| **Core Metrics** | RMSE, MAE, HitRate, TheilU | MSE, RMSE, MAE, R² |
| **Additional** | ✅ P&L metrics:<br>- Annualized Return<br>- Annualized Volatility<br>- Sharpe Ratio<br>- Cumulative Return | ✅ Directional Accuracy<br>❌ No P&L metrics |
| **Comparison** | ✅ Unified comparison table | ✅ Comparison table |

**Assessment**: ⚠️ **model_benchmark.py includes P&L metrics** which are valuable for financial applications.

---

### 6. **Diagnostic Outputs**

| Aspect | model_benchmark.py | Updated Notebooks |
|--------|-------------------|-------------------|
| **Residual Plots** | ✅ Comprehensive:<br>- Time series<br>- vs Fitted<br>- Histogram<br>- ACF/PACF | ❌ Limited plots |
| **Diagnostic Summary** | ✅ Saved to text file | ✅ Inline output |
| **Model Summaries** | ✅ Saved | ✅ Displayed |

**Assessment**: ✅ **model_benchmark.py has better visualization** and saved outputs.

---

## Critical Findings

### ✅ **Strengths of Updated Notebooks**

1. **Model Specification Fix**: 
   - Identifies functional form misspecification via RESET test
   - Fixes it with squared term (Adj R² improves: 0.0386 → 0.0449)
   - AIC improves: -3791.93 → -3798.78

2. **Comprehensive Diagnostics**:
   - Tests all major OLS assumptions
   - VIF analysis prevents multicollinearity issues
   - Chow test checks structural stability

3. **Better Variable Set**:
   - Uses 7 variables vs 2 in original
   - Includes market indicators (Dow Jones, ExxonMobil)
   - Includes non-linear term

### ⚠️ **Weaknesses of Updated Notebooks**

1. **Forecasting Methodology**:
   - Uses static forecast (fit once) vs expanding window
   - Less realistic for real-world application
   - May overstate performance

2. **ARIMAX Underperformance**:
   - ARIMAX performs identically to OLS (MSE = 0.003931)
   - Negative R² (-0.033193) - worse than baseline
   - Suggests exogenous variables don't add value

3. **Autocorrelation Remains**:
   - Breusch-Godfrey fails at multiple lags
   - HAC standard errors help inference but don't fix underlying issue

### ✅ **Strengths of model_benchmark.py**

1. **Superior Forecasting**:
   - Expanding window methodology
   - More realistic simulation
   - Avoids look-ahead bias

2. **Better ARIMAX**:
   - Includes seasonal components
   - More sophisticated grid search
   - Better captures time series patterns

3. **P&L Metrics**:
   - Financial performance metrics
   - Sharpe ratio, annualized returns
   - More relevant for trading applications

---

## Recommendations

### 🔄 **Integration Strategy**

1. **Use Updated Model Specification**:
   - Adopt the 7-variable model with squared term from `Diagnostics.ipynb`
   - This fixes functional form issues

2. **Adopt Expanding Window Forecasting**:
   - Use the expanding window methodology from `model_benchmark.py`
   - More appropriate for time series

3. **Combine ARIMAX Approaches**:
   - Use seasonal components from `model_benchmark.py`
   - Use comprehensive diagnostics from updated notebooks

4. **Add P&L Metrics**:
   - Include financial performance metrics from `model_benchmark.py`

### 📊 **Hybrid Approach**

**Best of Both Worlds**:
```
Updated Model Specification (7 vars + squared term)
    +
Expanding Window Forecasting (from model_benchmark.py)
    +
Comprehensive Diagnostics (from Diagnostics.ipynb)
    +
Seasonal ARIMAX (from model_benchmark.py)
    +
P&L Metrics (from model_benchmark.py)
```

---

## Conclusion

**Updated Notebooks**: Superior in **model specification** and **diagnostic rigor**
- ✅ Fixes functional form issues
- ✅ Comprehensive diagnostic testing
- ✅ Better variable selection

**model_benchmark.py**: Superior in **forecasting methodology** and **practical application**
- ✅ Expanding window forecasts
- ✅ P&L metrics
- ✅ Better ARIMAX implementation

**Recommendation**: **Integrate both approaches** - use the improved model specification from the notebooks with the superior forecasting methodology from `model_benchmark.py`.


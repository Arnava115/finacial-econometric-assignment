# Hybrid Model Benchmarking

This folder contains a **hybrid implementation** that combines the best features from both the original `model_benchmark.py` and the updated notebooks (`Diagnostics.ipynb` and `model_selection.ipynb`).

## 🎯 What This Hybrid Approach Does

### **Combines Best of Both Worlds:**

1. ✅ **Improved Model Specification** (from `Diagnostics.ipynb`)
   - Uses 7-variable OLS model with squared lagged returns
   - Addresses functional form misspecification identified by RESET test
   - Better model fit (Adj R²: 0.0386 → 0.0449)

2. ✅ **Expanding Window Forecasting** (from `model_benchmark.py`)
   - Refits model at each time step
   - More realistic simulation of real-world forecasting
   - Avoids look-ahead bias

3. ✅ **Comprehensive Diagnostics** (from `Diagnostics.ipynb`)
   - Ramsey RESET test (functional form)
   - Breusch-Godfrey test (autocorrelation)
   - Breusch-Pagan test (heteroskedasticity)
   - Jarque-Bera test (normality)
   - VIF analysis (multicollinearity)
   - Chow test (structural breaks)

4. ✅ **Seasonal ARIMAX** (from `model_benchmark.py`)
   - Sophisticated grid search with seasonal components
   - Periods: 7, 8, 13 weeks
   - Better captures time series patterns

5. ✅ **P&L Metrics** (from `model_benchmark.py`)
   - Annualized returns
   - Annualized volatility
   - Sharpe ratio
   - Cumulative returns

## 📋 Model Specification

### **OLS Predictors (7 variables):**
1. `dow_jones_us_oilandgas_index_ret` - Oil & gas sector returns
2. `exxonmobil_ret` - ExxonMobil stock returns
3. `cl1_ret_l2` - Lagged crude oil returns (2 weeks)
4. `weekly_commercial_short_diff` - Commercial short positions change
5. `weekly_commercial_long_diff` - Commercial long positions change
6. `weekly_change_in_crude_stock_diff` - Crude stock changes
7. `cl1_ret_l2_squared` - **Squared lagged returns** (non-linear term)

### **Why the Squared Term?**
- Ramsey RESET test identified functional form misspecification (p=0.0164)
- Adding squared term fixes this (RESET p=0.4692)
- Captures non-linear momentum/reversion effects
- Improves model fit and reduces specification error

## 🚀 Usage

### **Prerequisites:**
```bash
pip install pandas numpy statsmodels matplotlib scipy
```

### **Run the Benchmark:**
```bash
cd Hybrid
python hybrid_model_benchmark.py
```

### **Data Requirements:**
The script will look for data files in this order:
1. `../Updated/Oil_Model_Weekly_Core.csv`
2. `../Oil_Model_Weekly_Core.csv`
3. `Oil_Model_Weekly_Core.csv`

## 📊 Outputs

All results are saved to `hybrid_results/`:

1. **`benchmark_results.csv`**
   - Performance metrics for all models
   - RMSE, MAE, Hit Rate, Theil's U, R²
   - P&L metrics (Sharpe, Annualized Returns, etc.)

2. **`all_forecasts.csv`**
   - Individual forecasts for each model
   - Actual vs predicted values

3. **`diagnostics_summary.txt`**
   - Comprehensive diagnostic test results
   - All test statistics and p-values

## 🔍 Models Compared

1. **Naive**: Always predicts zero (baseline)
2. **AR(1)**: Simple autoregressive model
3. **OLS(Improved)**: 7-variable model with squared term + HAC robust SE
4. **ARIMAX**: ARIMA with exogenous variables + seasonal components

## 📈 Key Features

### **Expanding Window Methodology:**
- For each test period `t`, model is refit on data `[0:t)`
- Prediction is made for period `t`
- More realistic than static forecasting
- Simulates real-world forecasting scenario

### **Robust Inference:**
- Uses HAC (Newey-West) standard errors
- Accounts for autocorrelation and heteroskedasticity
- Valid inference even when assumptions are violated

### **Comprehensive Diagnostics:**
- Tests all major OLS assumptions
- Identifies and addresses model misspecification
- Ensures model validity

## 🎓 Academic Context

This hybrid approach addresses key econometric concerns:

1. **Model Specification**: RESET test ensures correct functional form
2. **Time Series Properties**: Expanding window respects temporal structure
3. **Robust Inference**: HAC standard errors handle violations
4. **Model Validation**: Comprehensive diagnostics ensure reliability

## 📝 Comparison with Original Approaches

| Feature | Original benchmark.py | Updated Notebooks | **Hybrid** |
|---------|----------------------|-------------------|------------|
| Model Spec | 2 variables | 7 vars + squared | ✅ 7 vars + squared |
| Forecasting | ✅ Expanding window | Static | ✅ Expanding window |
| Diagnostics | Basic | ✅ Comprehensive | ✅ Comprehensive |
| ARIMAX | ✅ Seasonal | Non-seasonal | ✅ Seasonal |
| P&L Metrics | ✅ Yes | No | ✅ Yes |

## 🔧 Customization

To modify the model:

1. **Change predictors**: Edit `OLS_PREDICTORS` list
2. **Adjust test fraction**: Change `TEST_FRAC` (default 0.20)
3. **Modify ARIMAX grid**: Edit `NON_SEASONAL_GRID` and `SEASONAL_CONFIGS`
4. **Control ARIMAX speed**: 
   - `ARIMAX_GRID_SEARCH_FREQUENCY = 50` (every 50 steps)
   - `ARIMAX_GRID_SEARCH_FREQUENCY = None` (only once at start - fastest)
5. **Add diagnostics**: Extend `run_comprehensive_diagnostics()` function

## 📚 References

- **Ramsey RESET Test**: Tests for functional form misspecification
- **HAC Standard Errors**: Newey-West robust standard errors
- **Expanding Window**: Time series cross-validation approach
- **ARIMAX**: ARIMA with exogenous variables

## ⚠️ Notes

- ARIMAX forecasting can be slow due to grid search
- Script uses periodic grid search (every 20 steps) to speed up
- All models use standardized predictors (z-scores)
- Missing values are handled by forward-filling with training means

## 🤝 Contributing

This is a hybrid implementation combining:
- Original work: `model_benchmark.py`
- Updated work: `Diagnostics.ipynb` and `model_selection.ipynb`

For questions or improvements, refer to the comparison analysis in `../Updated/COMPARISON_ANALYSIS.md`.


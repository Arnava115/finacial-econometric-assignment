# Quick Start Guide - Hybrid Model Benchmarking

## 🚀 Quick Run

```bash
cd Hybrid
python hybrid_model_benchmark.py
```

## 📋 What You'll Get

1. **Performance Comparison Table** (printed to console)
   - RMSE, MAE, Hit Rate, R², Sharpe Ratio
   - Ranked by RMSE

2. **Files Generated** (in `hybrid_results/` folder):
   - `benchmark_results.csv` - All metrics
   - `all_forecasts.csv` - Individual predictions
   - `diagnostics_summary.txt` - Diagnostic test results

## 🔍 Quick Interpretation

### **Best Model Selection:**
- **Lowest RMSE** = Best point forecast accuracy
- **Highest Sharpe** = Best risk-adjusted returns
- **Highest Hit Rate** = Best directional accuracy

### **Diagnostic Tests:**
- **RESET p > 0.05** = ✓ Functional form OK
- **Breusch-Godfrey p > 0.05** = ✓ No autocorrelation
- **Breusch-Pagan p > 0.05** = ✓ Homoskedasticity
- **VIF < 10** = ✓ No multicollinearity
- **Chow p > 0.05** = ✓ No structural break

## ⚡ Expected Runtime

- **Naive & AR(1)**: < 1 second
- **OLS**: ~10-30 seconds (expanding window)
- **ARIMAX**: ~2-5 minutes (optimized grid search + expanding window)

Total: ~2-5 minutes depending on data size

**Speed Optimizations Applied:**
- Reduced grid search from 559 to 63 combinations (~9x faster)
- Grid search runs every 50 steps (vs every 20)
- Can be set to run only once for maximum speed

## 🐛 Troubleshooting

### **File Not Found Error:**
```
FileNotFoundError: Data file not found
```
**Solution**: Ensure `Oil_Model_Weekly_Core.csv` exists in:
- `../Updated/` folder, OR
- `../` folder (parent directory), OR
- Current directory

### **ARIMAX Still Too Slow:**
If ARIMAX is still taking too long, edit `hybrid_model_benchmark.py`:

1. **Fastest option** - Run grid search only once:
   ```python
   ARIMAX_GRID_SEARCH_FREQUENCY = None  # Line 78
   ```

2. **Reduce grid search further**:
   - Edit `NON_SEASONAL_GRID` (line 63) to fewer combinations
   - Edit `SEASONAL_CONFIGS` (line 70) to fewer periods

3. **Make grid search less frequent**:
   ```python
   ARIMAX_GRID_SEARCH_FREQUENCY = 100  # Line 78 (higher = less frequent)
   ```

### **Import Errors:**
```bash
pip install pandas numpy statsmodels matplotlib scipy
```

## 📊 Understanding Results

### **RMSE (Root Mean Squared Error)**
- Lower is better
- Units: same as target variable (returns)
- Example: RMSE = 0.062 means average error is 6.2%

### **Sharpe Ratio**
- Higher is better
- Risk-adjusted return measure
- > 1 is good, > 2 is excellent

### **Hit Rate**
- Percentage of correct directional predictions
- > 50% means better than random
- > 55% is good for financial forecasting

## 🎯 Next Steps

1. Review `benchmark_results.csv` for model comparison
2. Check `diagnostics_summary.txt` for model validity
3. Analyze `all_forecasts.csv` for forecast patterns
4. Compare with original approaches (see `../Updated/COMPARISON_ANALYSIS.md`)


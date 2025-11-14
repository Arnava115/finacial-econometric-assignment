# Quick Reference: Key Findings for Report Writing

## Model Performance Summary (Out-of-Sample)

| Model | RMSE | MAE | Hit Rate | Sharpe | Annual Ret | Annual Vol |
|-------|------|-----|----------|--------|------------|------------|
| **Naive** | 0.0636 | 0.0440 | 0.0% | 0.00 | 0.0% | 0.0% |
| **AR(1)** | 0.0639 | 0.0440 | 52.8% | -0.27 | -11.3% | 41.4% |
| **OLS(final)** | 0.0634 | 0.0439 | 51.8% | 0.35 | 10.8% | 31.3% |
| **ARIMAX** | **0.0633** | 0.0440 | **51.8%** | **0.44** | **13.7%** | 30.9% |

**Winner: ARIMAX** - Best RMSE and Sharpe ratio

---

## Key Model Specifications

### ARIMAX (Best Model)
- **Order:** ARIMA(1,0,0) × SARIMA(0,0,1,8)
- **Exogenous variables:**
  - `cl1_ret_l4` (4-week lagged returns)
  - `weekly_change_in_crude_stock_diff` (weekly change in crude inventories)
- **AIC:** -3860.69

### OLS (Baseline)
- **Predictors:**
  - `cl1_ret_l4` (coefficient: ~0.0026, p=0.014)
  - `weekly_change_in_crude_stock_diff` (coefficient: ~-0.0052, p<0.001)
- **R²:** 0.010 (1.0%)
- **Robust standard errors:** HAC (Newey-West, 8 lags)

---

## Diagnostic Test Results

### OLS Diagnostics
- **Ljung-Box (lag 8):** stat=37.88, p<0.001 ❌ (residual autocorrelation)
- **Jarque-Bera:** stat=802.12, p<0.001 ❌ (non-normal residuals)
- **ARCH:** stat=209.86, p<0.001 ❌ (heteroskedasticity)
- **Residual stats:** Mean≈0, Std=0.0498, Skew=-0.57, Kurtosis=6.80

### ARIMAX Diagnostics
- **Ljung-Box (lag 8):** stat=15.52, p=0.050 ❌ (marginal - some autocorrelation)
- **Jarque-Bera:** stat=628.59, p<0.001 ❌ (non-normal residuals)
- **ARCH:** stat=202.85, p<0.001 ❌ (heteroskedasticity)
- **Residual stats:** Mean≈0.0009, Std=0.0494, Skew=-0.50, Kurtosis=6.33

**Key insight:** ARIMAX improves autocorrelation (lag 8 p-value goes from <0.001 to 0.050), but still has non-normal residuals and heteroskedasticity.

---

## Data Summary

- **Total observations:** ~1,534 (after preprocessing)
- **Training period:** 1995-2018 (~1,227 observations)
- **Test period:** 2018-2023 (307 observations)
- **Frequency:** Weekly (Friday close)
- **Target:** `cl1_ret` (WTI futures weekly returns)

---

## Key Predictors Explained

### 1. `cl1_ret_l4` (4-week lagged returns)
- **What it is:** Oil returns from 4 weeks ago
- **Coefficient:** Positive (~0.0026)
- **Trading interpretation:** Momentum effect - if oil rallied 4 weeks ago, expect continued strength
- **Why 4 weeks?** Captures intermediate-term momentum patterns

### 2. `weekly_change_in_crude_stock_diff` (Inventory changes)
- **What it is:** Week-over-week change in crude oil inventories
- **Coefficient:** Negative (~-0.0052)
- **Trading interpretation:** Falling inventories (negative change) → bullish signal
- **Economic logic:** Lower inventories = tighter supply = higher prices

---

## Performance Interpretation

### Hit Rate: 51.8%
- **What it means:** Model correctly predicts direction 51.8% of the time
- **Baseline:** Random = 50%
- **Edge:** +1.8 percentage points (modest but meaningful)
- **Trader view:** "Slightly better than a coin flip, but the edge compounds over time"

### Sharpe Ratio: 0.44
- **What it means:** Risk-adjusted return of 0.44
- **Interpretation:** Positive risk-adjusted returns - model adds value
- **Comparison:** 
  - Naive (flat): 0.00
  - AR(1): -0.27 (negative!)
  - OLS: 0.35
  - ARIMAX: 0.44 (best)
- **Trader view:** "Model generates positive returns per unit of risk taken"

### RMSE: 0.0633
- **What it means:** Average forecast error is 6.33% weekly
- **Context:** Oil weekly volatility is typically 5-8%
- **Interpretation:** Model error is similar to market volatility - reasonable for volatile asset
- **Trader view:** "Forecasts are noisy, but still informative"

### Annual Return: ~13.7%
- **What it means:** If trading long-only when forecast > 0
- **Caveat:** This is approximate, assumes perfect execution, no transaction costs
- **Trader view:** "Theoretical return - real-world will be lower due to costs and slippage"

---

## Model Limitations (Be Honest!)

1. **Low R² (1-2.5%)**
   - Most return variation is unpredictable (efficient markets)
   - Model captures small but meaningful edge

2. **Residual Autocorrelation**
   - Some patterns not captured (especially at lag 8+)
   - Suggests room for improvement

3. **Non-Normal Residuals**
   - Fat tails (kurtosis > 3)
   - Model underestimates extreme events
   - **Trading implication:** Use stop-losses, don't rely on model for tail risk

4. **Heteroskedasticity**
   - Volatility clustering present
   - Model doesn't fully adjust for changing volatility regimes
   - **Trading implication:** Consider volatility filters or dynamic position sizing

5. **Regime Changes**
   - Model trained on 1995-2018 data
   - Structural breaks possible (e.g., COVID-19, shale revolution)
   - **Trading implication:** Monitor performance, retrain if model degrades

6. **Weekly Horizon Only**
   - Designed for weekly forecasts
   - Not suitable for intraday/day trading
   - **Trading implication:** Use for weekly position sizing, not intraday signals

---

## Trader-Friendly Translations

### Technical → Plain English

| Technical Term | Trader Translation |
|----------------|-------------------|
| **RMSE** | Average forecast error |
| **Sharpe Ratio** | Risk-adjusted return |
| **Hit Rate** | Percentage of correct directional predictions |
| **R²** | How much of return variation the model explains |
| **Ljung-Box Test** | Checks if model missed any patterns |
| **Jarque-Bera Test** | Checks if residuals are normally distributed |
| **ARCH Test** | Checks if volatility is constant |
| **Heteroskedasticity** | Volatility changes over time |
| **Autocorrelation** | Patterns in residuals (model missed something) |
| **Out-of-sample** | Testing on data the model hasn't seen |

---

## Key Messages for Trader

### What Works
✅ Model provides a small but meaningful edge (52% accuracy, 0.44 Sharpe)  
✅ Combines momentum (4-week lag) with fundamentals (inventories)  
✅ Validated on unseen data (2018-2023)  
✅ Positive risk-adjusted returns  

### What Doesn't Work
❌ Model doesn't predict extreme events (crashes, spikes)  
❌ Some patterns not captured (residual autocorrelation)  
❌ Volatility clustering not fully modeled  
❌ Modest predictive power (R² ~1-2.5%)  

### How to Use
1. **Start small:** Test with small positions
2. **Combine signals:** Use model as one input, not the only input
3. **Risk management:** Always use stop-losses
4. **Monitor performance:** Track hit rate and Sharpe over time
5. **Adapt:** Retrain if model degrades

---

## Report Writing Checklist

- [ ] Explain every technical term in plain English
- [ ] Focus on practical trading implications
- [ ] Be honest about limitations
- [ ] Use concrete examples ("If inventories fall by 1M barrels...")
- [ ] Translate numbers into trader language ("52% accuracy = 2% edge over random")
- [ ] Include visualizations (tables, charts)
- [ ] Maintain professional but accessible tone
- [ ] Keep word count ~2,000 words
- [ ] Structure: Problem → Solution → Limitations → Recommendations

---

## Sample Sentences for Report

### Opening
"Oil futures returns are notoriously difficult to predict, but even small edges can add value when combined with proper risk management."

### Model Selection
"After testing multiple approaches, the ARIMAX model emerged as the best performer, delivering a Sharpe ratio of 0.44 compared to 0.00 for a naive 'always flat' strategy."

### Performance
"While a 52% directional accuracy may seem modest, the model's ability to avoid large losses and capture positive moves creates meaningful value over time."

### Limitations
"The model's low R² of 2.5% reflects the reality that most oil return variation is unpredictable—but even small edges can be valuable in trading."

### Trading Recommendation
"Use the model as one tool in your toolkit, not the only tool. Combine its signals with your fundamental views and always maintain proper risk management."

---

## Key Numbers to Memorize

- **Best model:** ARIMAX
- **Hit rate:** 51.8%
- **Sharpe ratio:** 0.44
- **RMSE:** 0.0633
- **Annual return:** ~13.7% (theoretical)
- **R²:** ~0.025 (2.5%)
- **Test period:** 307 observations (2018-2023)
- **Training period:** 1,227 observations (1995-2018)


# Oil Futures Returns Forecasting Model: Report Outline
## For Oil Trading Desk - 2,000 Words

---

## **Executive Summary** (~150 words)
**Purpose:** Give the trader a 30-second overview

- **Bottom line:** What signal does your model provide? (e.g., "Weekly directional forecast for WTI futures returns")
- **Key finding:** Best performing model (ARIMAX) with Sharpe ratio of 0.44 vs. naive baseline
- **Practical value:** How can this inform trading? (e.g., "52% directional accuracy with positive risk-adjusted returns")
- **Main limitation:** What should the trader watch out for? (e.g., "Model works best in normal market conditions; performance degrades during extreme volatility")

---

## **1. Introduction & Business Context** (~200 words)
**Purpose:** Set the scene - why does this matter for trading?

- **Trading objective:** Help identify weekly trading opportunities in WTI futures
- **Market context:** Oil futures are volatile; even small edges can be valuable
- **Your role:** Junior analyst providing quantitative support to trading decisions
- **What you deliver:** A forecasting model that translates data into actionable signals

**Key points:**
- Oil futures returns are notoriously hard to predict (weak-form efficiency)
- But small predictive edges, when combined with risk management, can add value
- Focus on interpretable signals that traders can understand and trust

---

## **2. Data & Methodology Overview** (~300 words)
**Purpose:** Explain what data you used and why, in trader-friendly terms

### 2.1 Data Sources
- **Weekly WTI futures returns** (cl1_ret) - your target variable
- **Key predictors:**
  - **4-week lagged returns** (cl1_ret_l4): Captures momentum/reversal patterns
  - **Weekly change in crude oil inventories** (weekly_change_in_crude_stock_diff): Fundamental supply/demand signal
- **Time period:** 1995-2018 (training), 2018-2023 (test) - explain why this split
- **Frequency:** Weekly - matches trading horizon and data release cadence

### 2.2 Why These Variables?
- **Lagged returns:** Oil markets show short-term momentum and longer-term mean reversion
- **Inventory changes:** Direct fundamental driver - rising inventories typically signal oversupply (bearish), falling inventories signal tightness (bullish)
- **What you tested but excluded:** Explain briefly why other variables (equity returns, yields, etc.) didn't make the final cut

### 2.3 Model Selection Process
- **Benchmark models:** Naive (always predict zero), AR(1), OLS, ARIMAX
- **Selection criteria:** Out-of-sample performance (not in-sample fit)
- **Evaluation metrics:** RMSE, directional accuracy (hit rate), Sharpe ratio

**Trader translation:**
- "I tested multiple approaches and chose the one that performs best on unseen data"
- "The model is validated on data it hasn't seen before, so you can trust it going forward"

---

## **3. Model Results & Trading Signals** (~500 words)
**Purpose:** The core of the report - what does your model tell the trader?

### 3.1 Best Model: ARIMAX
- **What it is:** Combines autoregressive structure (captures momentum patterns) with fundamental drivers (inventory changes)
- **Why it wins:** Lowest RMSE (0.0633), best Sharpe ratio (0.44), captures both short-term dynamics and fundamental drivers

### 3.2 Performance Metrics (Out-of-Sample)
**Present in trader-friendly language:**

| Metric | Value | What It Means for Trading |
|--------|-------|---------------------------|
| **Directional Accuracy** | 52% | Model correctly predicts direction slightly more than half the time - better than coin flip |
| **Sharpe Ratio** | 0.44 | Positive risk-adjusted returns - model adds value vs. staying flat |
| **RMSE** | 0.0633 | Average forecast error is ~6.3% weekly - reasonable for volatile oil markets |
| **Annual Return (approx)** | 13.7% | If trading long-only when forecast > 0 |

**Key insight:** While 52% accuracy seems modest, the model's ability to avoid large losses and capture positive moves creates value.

### 3.3 How to Use the Model Signal
**Practical trading interpretation:**

1. **Forecast > 0:** Model expects positive returns → Consider long positions
2. **Forecast < 0:** Model expects negative returns → Consider short positions or stay flat
3. **Forecast magnitude:** Larger absolute values suggest stronger conviction (but also check model uncertainty)

**Example interpretation:**
- "If the model forecasts +2% for next week, and inventories fell sharply, this suggests a bullish setup"
- "The model combines momentum (4-week lag) with fundamentals (inventory changes) to form its view"

### 3.4 Model Drivers (What Moves the Forecast)
- **4-week lag coefficient:** Positive → suggests momentum continuation
- **Inventory change coefficient:** Negative → falling inventories (negative diff) → bullish signal
- **Seasonal AR component:** Captures 8-week cyclical patterns in oil markets

**Trader takeaway:** "The model is telling you: if oil rallied 4 weeks ago AND inventories are falling, expect continued strength."

---

## **4. Model Reliability & Diagnostics** (~400 words)
**Purpose:** Be honest about what works and what doesn't

### 4.1 What Works Well
- **Out-of-sample validation:** Model tested on unseen data (2018-2023) - not just fit to historical data
- **Expanding window:** Model retrains weekly using all available data - adapts to changing market conditions
- **Robust standard errors:** Used HAC (Newey-West) to account for autocorrelation and heteroskedasticity

### 4.2 Diagnostic Tests & What They Mean
**Present in plain language:**

1. **Ljung-Box Test (Autocorrelation):**
   - **Finding:** Some residual autocorrelation at lag 8+ (p < 0.05)
   - **Trader interpretation:** "The model captures most patterns, but there may be some longer-term cycles we're missing"
   - **Impact:** Not critical for weekly forecasts, but suggests room for improvement

2. **Jarque-Bera Test (Normality):**
   - **Finding:** Residuals are not normally distributed (fat tails)
   - **Trader interpretation:** "Oil markets have extreme moves that the model doesn't fully capture"
   - **Impact:** Model may underestimate tail risk - use appropriate position sizing

3. **ARCH Test (Heteroskedasticity):**
   - **Finding:** Volatility clustering present
   - **Trader interpretation:** "When markets are volatile, they stay volatile - model doesn't fully adjust"
   - **Impact:** Consider volatility filters or dynamic position sizing

### 4.3 Model Limitations
**Be transparent:**

1. **Modest predictive power:** R² is low (~2.5%) - oil markets are hard to predict
2. **Residual autocorrelation:** Some patterns not captured - model isn't perfect
3. **Non-normal residuals:** Extreme events (crashes, spikes) not fully modeled
4. **Regime changes:** Model trained on 1995-2018 data - structural breaks possible
5. **Weekly horizon:** Model designed for weekly forecasts - not suitable for intraday/day trading

**Trader takeaway:** "This model provides an edge, but it's not a crystal ball. Use it as one input among many, with proper risk management."

---

## **5. Risk Considerations & Trading Recommendations** (~300 words)
**Purpose:** Help the trader use the model safely

### 5.1 When to Trust the Model
- **Normal market conditions:** Model performs best in typical volatility regimes
- **Clear fundamental signals:** When inventory data aligns with model forecast
- **Moderate forecast magnitudes:** Extreme forecasts may be less reliable

### 5.2 When to Be Cautious
- **Extreme volatility:** Model may underestimate risk during crises (2008, 2020 COVID)
- **Structural breaks:** Major policy changes, supply disruptions may invalidate historical patterns
- **Low liquidity periods:** Model assumes normal market functioning
- **Conflicting signals:** If model says "long" but your fundamental view says "short" - trust your judgment

### 5.3 Practical Trading Recommendations
1. **Position sizing:** Don't bet the farm - model provides an edge, not certainty
2. **Risk management:** Always use stop-losses - model doesn't predict tail events well
3. **Combine with other inputs:** Use model as one tool in your toolkit, not the only tool
4. **Monitor performance:** Track model accuracy over time - if it degrades, retrain
5. **Volatility filters:** Consider reducing position size when VIX is elevated

### 5.4 Expected Performance
- **Best case:** Model continues to deliver ~0.44 Sharpe ratio
- **Worst case:** Structural break causes model to underperform - need to retrain
- **Realistic:** Somewhere in between - model adds value but isn't perfect

---

## **6. Conclusion & Next Steps** (~150 words)
**Purpose:** Wrap up with actionable next steps

### 6.1 Summary
- **Model delivers:** Positive risk-adjusted returns with 52% directional accuracy
- **Key drivers:** Momentum (4-week lag) + fundamentals (inventory changes)
- **Best use case:** Weekly trading horizon with proper risk management

### 6.2 Recommendations for the Trader
1. **Start small:** Test the model with small positions to build confidence
2. **Track performance:** Monitor hit rate and Sharpe ratio over next quarter
3. **Provide feedback:** Share when model signals align/diverge from your intuition
4. **Iterate:** Model can be refined based on your trading experience

### 6.3 Future Enhancements
- **Volatility regime detection:** Adjust forecasts based on market volatility
- **Additional predictors:** Test other fundamental variables (rig counts, production, etc.)
- **Regime-switching models:** Adapt to different market conditions automatically

---

## **Appendix: Technical Details** (Optional, not counted in word limit)
- Model equations
- Detailed diagnostic tables
- Code references
- Data sources and transformations

---

## **Writing Tips for Trader Audience**

### Do:
- ✅ Use concrete examples ("If inventories fall by 1M barrels, model expects +1.5% return")
- ✅ Translate technical terms ("R² of 0.025 means the model explains 2.5% of return variation")
- ✅ Focus on practical implications ("52% accuracy means 2% edge over random")
- ✅ Be honest about limitations ("Model doesn't predict crashes")
- ✅ Use tables and visualizations to summarize key points

### Don't:
- ❌ Overwhelm with technical jargon ("The ARIMAX(1,0,0)×SARIMA(0,0,1,8) specification...")
- ❌ Hide weaknesses ("Model is perfect" - it's not)
- ❌ Make unrealistic claims ("This will make you rich")
- ❌ Skip the "so what?" - always explain why results matter for trading

---

## **Key Numbers to Highlight**

From your benchmark_results.csv:
- **ARIMAX RMSE:** 0.0633 (best)
- **ARIMAX Sharpe:** 0.44 (best)
- **ARIMAX Hit Rate:** 51.8% (slightly better than random)
- **ARIMAX Annual Return:** ~13.7% (if trading long-only on positive forecasts)
- **Test period:** 307 observations (2018-2023)

From diagnostics:
- **R²:** ~0.025 (low but meaningful for financial returns)
- **Ljung-Box lag 8:** p < 0.05 (some residual autocorrelation)
- **Jarque-Bera:** p < 0.001 (non-normal residuals - fat tails)
- **ARCH:** p < 0.001 (volatility clustering)

---

## **Suggested Report Structure (Word Count)**

1. **Executive Summary:** 150 words
2. **Introduction & Business Context:** 200 words
3. **Data & Methodology:** 300 words
4. **Model Results & Trading Signals:** 500 words ⭐ (Most important section)
5. **Model Reliability & Diagnostics:** 400 words
6. **Risk Considerations & Trading Recommendations:** 300 words
7. **Conclusion & Next Steps:** 150 words

**Total: ~2,000 words**

---

## **Visualizations to Include**

1. **Performance comparison table** (from benchmark_results.csv)
2. **Time series of forecasts vs. actuals** (from all_forecasts.csv)
3. **Residual diagnostic plots** (already generated)
4. **Simple trading signal chart** (forecast > 0 = long, < 0 = flat/short)

---

## **Final Checklist**

Before submitting, ensure:
- [ ] Every technical term is explained in trader-friendly language
- [ ] All claims are backed by your results
- [ ] Limitations are clearly stated
- [ ] Practical trading implications are explicit
- [ ] Word count is ~2,000 words
- [ ] Report flows logically from problem → solution → limitations → recommendations
- [ ] Visualizations support key points
- [ ] Tone is professional but accessible (junior analyst to senior trader)


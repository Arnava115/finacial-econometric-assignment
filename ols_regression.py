"""
OLS exploratory modelling for the Oil coursework project.

Loads the model-ready weekly dataset, fits a baseline OLS using a curated
set of predictors, and reports regression diagnostics plus VIFs to flag
multicollinearity.

Run:
    python ols_regression.py
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_FILE = Path("Oil_Model_Weekly_Core.csv")
OUTPUT_DIR = Path("ols_diagnostics")

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
    "weekly_prod_crude_diff",
    # "bdiy_index_ret",
    # "weekly_commercial_long_diff",
    # "weekly_commercial_short_diff", #because both are included in the total open interest
    "weekly_total_open_interest_diff",
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


if __name__ == "__main__":
    main()


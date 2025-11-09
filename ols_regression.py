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

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_FILE = Path("Oil_Model_Weekly_Core.csv")

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

    print("\nReminder: Inspect residual plots and run diagnostic tests as needed.")
    print("Consider adjusting CANDIDATE_PREDICTORS if VIFs are excessive or")
    print("if trader feedback suggests a leaner feature set.")


if __name__ == "__main__":
    main()


"""
ARIMAX modelling pipeline for the Oil coursework dataset.

Steps
-----
1. Load `Oil_Model_Weekly_Core.csv`.
2. Split into train/test using TEST_FRAC.
3. Grid-search a small set of ARMA(p,q) orders (d=0) with selected exogenous
   predictors (those validated via OLS).
4. Fit the best model, generate test forecasts, and report accuracy metrics.
5. Save residual diagnostic plots (time series, histogram, residuals vs fitted,
   ACF/PACF) and Ljung–Box statistics.

Run:
    python arimax_model.py
"""

from pathlib import Path
from typing import List, Tuple
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_FILE = Path("Oil_Model_Weekly_Core.csv")
OUTPUT_DIR = Path("arimax_diagnostics")
TEST_FRAC = 0.20

# Exogenous predictors selected from the OLS screening stage
EXOG_COLUMNS: List[str] = [
    "cl1_ret_l4",
    "weekly_change_in_crude_stock_diff",
    "weekly_prod_crude_diff",
    "weekly_total_open_interest_diff",
]

# Candidate ARMA orders (p, q) for d=0
NON_SEASONAL_GRID: List[Tuple[int, int]] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
]

SEASONAL_CONFIGS = [
    {"period": 7, "orders": [(0, 1)]},
    {"period": 13, "orders": [(1, 0), (0, 1), (1, 1)]},
]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run process_oil_data.py to regenerate the dataset."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def toy_pnl(y_true: pd.Series, y_pred: pd.Series) -> dict:
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


def select_exog(df: pd.DataFrame, columns: List[str]) -> List[str]:
    available = [c for c in columns if c in df.columns]
    missing = sorted(set(columns) - set(available))
    if missing:
        print(f"⚠️  Missing exogenous columns: {missing}")
    if not available:
        raise ValueError("No exogenous columns found in dataset for ARIMAX.")
    return available


def train_test_split(df: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int((1 - frac) * len(df))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Invalid TEST_FRAC; resulted in empty train/test split.")
    return df.iloc[:split_idx], df.iloc[split_idx:]


def grid_search_arimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float, SARIMAX]:
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
                except Exception as exc:
                    print(
                        f"⚠️  ARIMAX grid search failed for order {order} seasonal {seasonal_order}: {exc}"
                    )

    if best_model is None:
        raise RuntimeError("ARIMAX grid search failed for all candidate orders.")

    return best_order, best_seasonal_order, best_aic, best_model


def save_residual_plots(resid: pd.Series) -> None:
    ensure_output_dir(OUTPUT_DIR)
    resid = resid.dropna()

    plt.figure(figsize=(10, 4))
    plt.plot(resid, color="steelblue", linewidth=0.7)
    plt.title("ARIMAX Residuals (time series)")
    plt.ylabel("Residual")
    plt.xlabel("Observation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_timeseries.png", dpi=150)
    plt.close()

    fitted = resid * 0  # placeholder to align scatter plot axes

    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, resid, s=10, color="darkorange", alpha=0.6)
    plt.axhline(0, color="gray", linewidth=1, linestyle="--")
    plt.title("Residuals scatter (zero baseline)")
    plt.xlabel("Fitted (not available in SARIMAX resid)")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_scatter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=40, color="slateblue", edgecolor="white", alpha=0.85)
    plt.title("Residual histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_hist.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(resid, lags=24, ax=ax[0])
    ax[0].set_title("Residual ACF (lags 1–24)")
    plot_pacf(resid, lags=24, ax=ax[1], method="ywm")
    ax[1].set_title("Residual PACF (lags 1–24)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals_acf_pacf.png", dpi=150)
    plt.close(fig)


def ljung_box_table(resid: pd.Series, lags: List[int]) -> pd.DataFrame:
    lb = acorr_ljungbox(resid.dropna(), lags=lags, return_df=True)
    lb.index.name = "lag"
    lb.reset_index(inplace=True)
    lb.rename(columns={"lb_stat": "LB stat", "lb_pvalue": "p-value"}, inplace=True)
    return lb


def main():
    ensure_output_dir(OUTPUT_DIR)
    df = load_dataset(DATA_FILE)
    exog_cols = select_exog(df, EXOG_COLUMNS)

    train_df, test_df = train_test_split(df, TEST_FRAC)
    y_train = train_df["cl1_ret"]
    y_test = test_df["cl1_ret"]

    exog_train = train_df[exog_cols]
    exog_test = test_df[exog_cols]
    train_means = exog_train.mean()
    train_stds = exog_train.std(ddof=0).replace(0, 1)
    exog_train = (exog_train - train_means) / train_stds
    exog_test = (exog_test.fillna(train_means) - train_means) / train_stds

    print("=" * 72)
    print("ARIMAX MODEL SETUP")
    print("=" * 72)
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples : {len(test_df):,}")
    print(f"Exogenous vars ({len(exog_cols)}): {exog_cols}")
    print()

    best_order, best_seasonal_order, best_aic, best_model = grid_search_arimax(
        y_train, exog_train
    )

    print(f"Best ARIMA order (p,d,q): {best_order}")
    if best_seasonal_order is not None:
        print(f"Seasonal order (P,D,Q,s): {best_seasonal_order}")
    print(f"Best AIC: {best_aic:.2f}")
    print(best_model.summary())

    # Forecast on the test set
    forecasts = best_model.get_forecast(steps=len(test_df), exog=exog_test)
    y_pred = forecasts.predicted_mean

    # Metrics
    valid_idx = y_pred.index.intersection(y_test.index)
    metric_values = metrics(y_test.loc[valid_idx].values, y_pred.loc[valid_idx].values)
    pnl_values = toy_pnl(y_test.loc[valid_idx], y_pred.loc[valid_idx])

    print("\n=== Test-set performance ===")
    for k, v in {**metric_values, **pnl_values}.items():
        print(f"{k:15s}: {v: .4f}")
    print(f"Number of test observations: {len(valid_idx)}")

    # Residuals diagnostics on the fitted (train) sample
    resid = best_model.resid
    save_residual_plots(resid)
    lb_tbl = ljung_box_table(resid, lags=[4, 8, 12, 24])

    print("\nLjung-Box test on training residuals:")
    print(lb_tbl.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\nDiagnostic plots saved to: {OUTPUT_DIR.resolve()}")

    # Save forecasts for reference
    out_forecast = pd.DataFrame(
        {
            "y_true": y_test.loc[valid_idx],
            "y_pred": y_pred.loc[valid_idx],
        }
    )
    out_forecast.to_csv(OUTPUT_DIR / "test_forecasts.csv")
    print(f"Saved test forecasts to: {(OUTPUT_DIR / 'test_forecasts.csv').resolve()}")


if __name__ == "__main__":
    main()


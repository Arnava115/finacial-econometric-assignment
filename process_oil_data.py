"""
End-to-end preprocessing for the Oil coursework dataset.

Inputs:
  - Oil Data.xlsx (sheets: Industry_D, Industry_W, Industry_M, Macro_M, StockPrices_M)

Outputs:
  - Oil_Clean_Weekly.csv          (merged, harmonized weekly panel incl. cl1_ret)
  - Oil_Clean_Weekly_Stn.csv      (optional 'stationarity-ready' predictors + cl1_ret)
  - Oil_Model_Weekly_Core.csv     (compact modeling table for forecasting cl1_ret)

Frequency rules:
  - Daily -> resample to weekly (Friday) using last observation
  - Weekly -> keep as-is (assumed to align to weeks)
  - Monthly -> forward fill to weekly (Friday)

Transform rules:
  - Returns: log returns for price/level series where applicable
  - Stationarity-ready:
      * Keep return-like series as-is (then z-score)
      * For level-like series: log-diff if strictly positive else first diff
  - Model-ready:
      * Target: cl1_ret
      * AR lags: cl1_ret_l1..l4
      * Cross-asset returns: brent_ret, cl2_ret, bdiy_index_ret, spx_ret, dxy_returns (if present)
      * Weekly industry diffs: prod, rigs, Δstocks, commercials, OI
      * Macro diffs: 3m_yield, 10y_yield, term_spread, vix
      * Oil-equity returns

Run:
  python process_oil_data_full.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------- CONFIG --------------------
DATA_XLSX = Path("Oil Data.xlsx")
OUT_CLEAN = Path("Oil_Clean_Weekly.csv")
OUT_STN   = Path("Oil_Clean_Weekly_Stn.csv")
OUT_MODEL = Path("Oil_Model_Weekly_Core.csv")

STATIONARITY_MAX_MISS = 0.10  # drop predictors with >10% NA post-transform
Z_SCORE_STN = True            # z-score stationarity-ready predictors
MODEL_LAGS = 4                # cl1_ret AR lags
STANDARDIZE_MODEL_PRED = True # z-score predictors in model dataset
# ------------------------------------------------


# --------------- Utilities ----------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, trim, replace special chars with underscores."""
    return df.rename(columns=lambda c: "_".join(
        str(c).strip().lower()
        .replace("%", "pct")
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .split()
    ))

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Find a date column (prefer one named like 'date'), set as index."""
    d = df.copy()
    date_col = None
    for c in d.columns:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = d.columns[0]
    d[date_col] = pd.to_datetime(d[date_col])
    return d.set_index(date_col).sort_index()

def resample_weekly_last(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly Friday close/last tick."""
    return df.resample("W-FRI").last()

def resample_weekly_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly Friday, forward-fill within month."""
    return df.resample("W-FRI").ffill()

def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """Natural log return (safe where positive)."""
    return np.log(series / series.shift(periods))

def is_return_like(name: str) -> bool:
    n = name.lower()
    return ("_ret" in n) or (n.endswith("returns")) or (n == "returns") or (n == "dxy_returns")

def transform_levels_to_stationary(s: pd.Series) -> pd.Series:
    """Log-diff if strictly positive; else first difference."""
    s = s.astype(float)
    if np.isfinite(s).sum() < 3:
        return pd.Series(index=s.index, dtype=float)
    if (s > 0).all():
        return np.log(s).diff()
    else:
        return s.diff()

# --------------- Load & Normalize ---------------
def load_all_sheets(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path)

    # 1) Industry_D (Daily -> Weekly last)
    ind_d = pd.read_excel(xls, sheet_name="Industry_D")
    ind_d = standardize_columns(ind_d)
    ind_d = ensure_datetime_index(ind_d)
    ind_d_w = resample_weekly_last(ind_d)

    # Compute weekly returns for common price-like series if present
    for col in ["cl1", "brent", "cl2", "crkss12_index", "bdiy_index"]:
        if col in ind_d_w.columns:
            ind_d_w[f"{col}_ret"] = log_return(ind_d_w[col])

    # 2) Industry_W (Weekly)
    ind_w = pd.read_excel(xls, sheet_name="Industry_W")
    ind_w = standardize_columns(ind_w)
    ind_w = ensure_datetime_index(ind_w)
    # Assume already aligned weekly; if not, uncomment:
    # ind_w = resample_weekly_last(ind_w)

    # 3) Industry_M (Monthly -> Weekly ffill)
    ind_m = pd.read_excel(xls, sheet_name="Industry_M")
    ind_m = standardize_columns(ind_m)
    ind_m = ensure_datetime_index(ind_m)
    ind_m_w = resample_weekly_ffill(ind_m)
    # Drop columns we already have from daily side to avoid duplicates
    for c in ["cl1", "cl2", "brent", "crks321c_index", "baltic_dry_index"]:
        if c in ind_m_w.columns:
            ind_m_w = ind_m_w.drop(columns=c)

    # 4) Macro_M (Monthly -> Weekly ffill)
    macro_m = pd.read_excel(xls, sheet_name="Macro_M")
    macro_m = standardize_columns(macro_m)
    macro_m = ensure_datetime_index(macro_m)
    macro_w = resample_weekly_ffill(macro_m)
    # Build term spread if yields exist
    if "3m_yield" in macro_w.columns and "10y_yield" in macro_w.columns:
        macro_w["term_spread"] = macro_w["10y_yield"] - macro_w["3m_yield"]
    # If SPX level exists, compute returns
    if "spx" in macro_w.columns and "spx_ret" not in macro_w.columns:
        macro_w["spx_ret"] = log_return(macro_w["spx"])

    # 5) StockPrices_M (Monthly -> Weekly ffill + returns)
    stocks_m = pd.read_excel(xls, sheet_name="StockPrices_M")
    stocks_m = standardize_columns(stocks_m)
    stocks_m = ensure_datetime_index(stocks_m)
    stocks_w = resample_weekly_ffill(stocks_m)
    for col in stocks_w.columns:
        if stocks_w[col].dtype.kind in "biufc" and not col.endswith("_ret"):
            stocks_w[f"{col}_ret"] = log_return(stocks_w[col])

    return ind_d_w, ind_w, ind_m_w, macro_w, stocks_w

# --------------- Build Clean Weekly Panel ---------------
def build_clean_weekly(ind_d_w, ind_w, ind_m_w, macro_w, stocks_w) -> pd.DataFrame:
    # Start from daily->weekly base which has CL1, returns, etc.
    master = ind_d_w.copy()

    # Join numeric cols from each other table
    for df_to_merge in [ind_w, ind_m_w, macro_w, stocks_w]:
        num_cols = df_to_merge.select_dtypes(include=[np.number]).columns.tolist()
        master = master.join(df_to_merge[num_cols], how="left")

    # Ensure cl1_ret target is present
    if "cl1_ret" not in master.columns and "cl1" in master.columns:
        master["cl1_ret"] = log_return(master["cl1"])

    # Drop the initial NaN caused by return shift
    master = master.dropna(subset=["cl1_ret"])

    # Light missing handling:
    miss_ratio = master.isna().mean()

    # Fill tiny gaps (<5%)
    fill_cols = miss_ratio[(miss_ratio > 0) & (miss_ratio < 0.05)].index.tolist()
    if fill_cols:
        master[fill_cols] = master[fill_cols].ffill().bfill()

    # Drop columns with heavy missingness (>30%)
    drop_cols = master.columns[master.isna().mean() > 0.30].tolist()
    if drop_cols:
        master = master.drop(columns=drop_cols)

    # Drop remaining NA rows for a dense panel
    master_clean = master.dropna()

    return master_clean, drop_cols

# --------------- Build Stationarity-Ready Panel ---------------
def build_stationarity_ready(clean_df: pd.DataFrame) -> pd.DataFrame:
    if "cl1_ret" not in clean_df.columns:
        raise ValueError("cl1_ret missing in clean weekly panel.")

    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    target = "cl1_ret"
    predictors = [c for c in numeric_cols if c != target]

    out = pd.DataFrame(index=clean_df.index)
    out[target] = clean_df[target]

    kept, dropped = [], []

    for col in predictors:
        s = clean_df[col]
        if is_return_like(col):
            t = s.copy()
        else:
            t = transform_levels_to_stationary(s)

        # Filter by missingness
        if t.isna().mean() > STATIONARITY_MAX_MISS:
            dropped.append(col)
            continue

        # Z-score if enabled, skip near-constant
        if Z_SCORE_STN:
            sd = t.std(ddof=0)
            if np.isfinite(sd) and sd > 0:
                t = (t - t.mean()) / sd
            else:
                dropped.append(col)
                continue

        out[col] = t
        kept.append(col)

    out = out.dropna()
    return out

# --------------- Build Model-Ready (Core) ---------------
def build_model_core(clean_df: pd.DataFrame) -> pd.DataFrame:
    if "cl1_ret" not in clean_df.columns:
        raise ValueError("cl1_ret missing in clean weekly panel.")

    model = pd.DataFrame(index=clean_df.index)
    # Target
    model["cl1_ret"] = clean_df["cl1_ret"]

    # AR lags
    for L in range(1, MODEL_LAGS + 1):
        model[f"cl1_ret_l{L}"] = clean_df["cl1_ret"].shift(L)

    # Cross-asset returns if present
    cross_ret = ["brent_ret", "cl2_ret", "bdiy_index_ret", "spx_ret", "dxy_returns"]
    for col in cross_ret:
        if col in clean_df.columns:
            model[col] = clean_df[col]

    # Weekly industry levels -> first differences
    weekly_levels = [
        "weekly_prod_crude", "weekly_rig_count", "weekly_change_in_crude_stock",
        "weekly_commercial_long", "weekly_commercial_short", "weekly_total_open_interest"
    ]
    for col in weekly_levels:
        if col in clean_df.columns:
            model[col + "_diff"] = clean_df[col].diff()

    # Macro levels -> differences
    macro_levels = ["3m_yield", "10y_yield", "term_spread", "vix"]
    for col in macro_levels:
        if col in clean_df.columns:
            model[col + "_diff"] = clean_df[col].diff()

    # Oil equity sentiment (returns already)
    oil_eq_ret = [
        "dow_jones_us_oilandgas_index_ret", "exxonmobil_ret", "conocophilips_ret",
        "chevron_ret", "bp_ret", "totalenergies_ret"
    ]
    for col in oil_eq_ret:
        if col in clean_df.columns:
            model[col] = clean_df[col]

    # Drop NA from lags/diffs
    model = model.dropna()

    # Standardize predictors (not the target)
    if STANDARDIZE_MODEL_PRED:
        pred_cols = [c for c in model.columns if c != "cl1_ret"]
        model[pred_cols] = (model[pred_cols] - model[pred_cols].mean()) / model[pred_cols].std(ddof=0)

    return model

# ----------------------------- MAIN -----------------------------
def main():
    # Load and normalize all sheets
    ind_d_w, ind_w, ind_m_w, macro_w, stocks_w = load_all_sheets(DATA_XLSX)

    # Build clean weekly panel
    clean_df, dropped_cols = build_clean_weekly(ind_d_w, ind_w, ind_m_w, macro_w, stocks_w)
    clean_df.to_csv(OUT_CLEAN)

    print("Clean weekly panel:")
    print(f"  Rows: {len(clean_df):,}  Cols: {clean_df.shape[1]}")
    print(f"  Dates: {clean_df.index.min().date()} -> {clean_df.index.max().date()}")
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} columns for high missingness (>30%): {dropped_cols[:10]}{' ...' if len(dropped_cols)>10 else ''}")
    print(f"  Saved: {OUT_CLEAN.resolve()}")

    # Build stationarity-ready (optional but useful for EDA/regression)
    stn_df = build_stationarity_ready(clean_df)
    stn_df.to_csv(OUT_STN)
    print("\nStationarity-Ready panel:")
    print(f"  Rows: {len(stn_df):,}  Cols: {stn_df.shape[1]} (incl. target)")
    print(f"  Dates: {stn_df.index.min().date()} -> {stn_df.index.max().date()}")
    print(f"  Saved: {OUT_STN.resolve()}")

    # Build model-ready compact dataset
    model_df = build_model_core(clean_df)
    model_df.to_csv(OUT_MODEL)
    print("\nModel-ready weekly dataset (core):")
    print(f"  Rows: {len(model_df):,}  Cols: {model_df.shape[1]} (incl. target)")
    print(f"  Dates: {model_df.index.min().date()} -> {model_df.index.max().date()}")
    print(f"  Saved: {OUT_MODEL.resolve()}")

if __name__ == "__main__":
    main()

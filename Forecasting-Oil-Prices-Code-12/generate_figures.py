"""
Generate visualization figures for the Hybrid Model Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
forecasts_df = pd.read_csv("Results/all_forecasts.csv", index_col=0, parse_dates=True)
results_df = pd.read_csv("Results/benchmark_results.csv", index_col=0)

print("Generating figures for Hybrid Model Report...")

# ==================== Figure 1: Performance Comparison ====================
print("1. Performance comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

models = ['Naive', 'AR(1)', 'OLS(Improved)', 'ARIMAX']
colors = ['gray', 'orange', 'blue', 'green']
model_labels = ['Naive', 'AR(1)', 'OLS', 'ARIMAX']  # Display labels

# RMSE
axes[0, 0].bar(model_labels, results_df.loc[models, 'RMSE'], color=colors, alpha=0.7)
axes[0, 0].set_title('RMSE (Lower is Better)', fontweight='bold')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df.loc[models, 'RMSE']):
    axes[0, 0].text(i, v + 0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# Hit Rate
axes[0, 1].bar(model_labels, results_df.loc[models, 'HitRate'] * 100, color=colors, alpha=0.7)
axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
axes[0, 1].set_title('Hit Rate (Higher is Better)', fontweight='bold')
axes[0, 1].set_ylabel('Hit Rate (%)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend()
for i, v in enumerate(results_df.loc[models, 'HitRate'] * 100):
    axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

# Sharpe Ratio
axes[1, 0].bar(model_labels, results_df.loc[models, 'Sharpe(approx)'], color=colors, alpha=0.7)
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Sharpe Ratio (Higher is Better)', fontweight='bold')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df.loc[models, 'Sharpe(approx)']):
    if not np.isnan(v):
        axes[1, 0].text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', 
                       ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

# Annual Return
axes[1, 1].bar(model_labels, results_df.loc[models, 'AnnRet(approx)'] * 100, color=colors, alpha=0.7)
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Annual Return (Higher is Better)', fontweight='bold')
axes[1, 1].set_ylabel('Annual Return (%)')
axes[1, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df.loc[models, 'AnnRet(approx)'] * 100):
    axes[1, 1].text(i, v + 1 if v >= 0 else v - 2, f'{v:.1f}%', 
                   ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'performance_comparison.png'}")

# ==================== Figure 2: Forecasts vs Actual ====================
print("2. Forecasts vs actual...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# OLS Forecasts
axes[0].plot(forecasts_df.index, forecasts_df['y_true'], 'k-', label='Actual Returns', linewidth=1.5, alpha=0.7)
axes[0].plot(forecasts_df.index, forecasts_df['ols'], 'b-', label='OLS Forecast', linewidth=1.2, alpha=0.8)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[0].set_title('OLS Model: Forecasts vs Actual Returns', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Weekly Return')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# ARIMAX Forecasts
axes[1].plot(forecasts_df.index, forecasts_df['y_true'], 'k-', label='Actual Returns', linewidth=1.5, alpha=0.7)
axes[1].plot(forecasts_df.index, forecasts_df['arimax'], 'g-', label='ARIMAX Forecast', linewidth=1.2, alpha=0.8)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[1].set_title('ARIMAX Model: Forecasts vs Actual Returns', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Weekly Return')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "forecasts_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'forecasts_vs_actual.png'}")

# ==================== Figure 3: Trading Signals ====================
print("3. Trading signals...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top panel: Forecasts and signals
axes[0].fill_between(forecasts_df.index, 0, forecasts_df['arimax'], 
                     where=(forecasts_df['arimax'] > 0), 
                     color='green', alpha=0.3, label='Long Signal')
axes[0].fill_between(forecasts_df.index, 0, forecasts_df['arimax'], 
                     where=(forecasts_df['arimax'] <= 0), 
                     color='red', alpha=0.3, label='Short/Flat Signal')
axes[0].plot(forecasts_df.index, forecasts_df['arimax'], 'k-', linewidth=1.5, alpha=0.8, label='ARIMAX Forecast')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_title('Trading Signals Over Time', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Forecast')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Bottom panel: Actual returns colored by signal correctness
correct = (np.sign(forecasts_df['y_true']) == np.sign(forecasts_df['arimax']))
colors_bottom = ['green' if c else 'red' for c in correct]
axes[1].bar(forecasts_df.index, forecasts_df['y_true'], color=colors_bottom, alpha=0.6, width=5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_title('Actual Returns (Green = Correct Prediction, Red = Incorrect)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Actual Return')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "trading_signals.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'trading_signals.png'}")

# ==================== Figure 4: Forecast Scatter ====================
print("4. Forecast scatter plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# OLS scatter
axes[0].scatter(forecasts_df['y_true'], forecasts_df['ols'], alpha=0.5, s=20)
xlim = axes[0].get_xlim()
ylim = axes[0].get_ylim()
lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Return')
axes[0].set_ylabel('Forecast')
axes[0].set_title('OLS: Forecast vs Actual', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal', adjustable='box')

# ARIMAX scatter
axes[1].scatter(forecasts_df['y_true'], forecasts_df['arimax'], alpha=0.5, s=20, color='green')
xlim = axes[1].get_xlim()
ylim = axes[1].get_ylim()
lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
axes[1].plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Return')
axes[1].set_ylabel('Forecast')
axes[1].set_title('ARIMAX: Forecast vs Actual', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(output_dir / "forecast_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'forecast_scatter.png'}")

# ==================== Figure 5: Error Distribution ====================
print("5. Error distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# OLS errors
ols_errors = forecasts_df['y_true'] - forecasts_df['ols']
axes[0].hist(ols_errors.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0].axvline(x=ols_errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {ols_errors.mean():.4f}')
axes[0].set_xlabel('Forecast Error')
axes[0].set_ylabel('Frequency')
axes[0].set_title('OLS: Distribution of Forecast Errors', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ARIMAX errors
arimax_errors = forecasts_df['y_true'] - forecasts_df['arimax']
axes[1].hist(arimax_errors.dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1].axvline(x=arimax_errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {arimax_errors.mean():.4f}')
axes[1].set_xlabel('Forecast Error')
axes[1].set_ylabel('Frequency')
axes[1].set_title('ARIMAX: Distribution of Forecast Errors', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'error_distribution.png'}")

# ==================== Figure 6: Cumulative Returns ====================
print("6. Cumulative returns...")
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate cumulative returns for each model
for model_name, model_col in [('Naive', 'naive'), ('AR(1)', 'ar1'), ('OLS', 'ols'), ('ARIMAX', 'arimax')]:
    # Long when forecast > 0
    positions = (forecasts_df[model_col] > 0).astype(float)
    returns = positions * forecasts_df['y_true']
    cumulative = returns.cumsum()
    ax.plot(forecasts_df.index, cumulative, label=model_name, linewidth=2, alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title('Cumulative Returns from Long-Only Trading Strategy', fontweight='bold', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "cumulative_returns.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'cumulative_returns.png'}")

print("\nAll figures generated successfully!")
print(f"Figures saved to: {output_dir.resolve()}")


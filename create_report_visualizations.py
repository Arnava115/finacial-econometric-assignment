"""
Create visualizations and tables for the oil forecasting report.
Generates trader-friendly charts and tables to support the written report.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional-looking plots
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = Path("model_benchmarks")
FIGURES_DIR = Path("report_figures")
FIGURES_DIR.mkdir(exist_ok=True)

def load_data():
    """Load forecast data and benchmark results."""
    forecasts_df = pd.read_csv(OUTPUT_DIR / "all_forecasts.csv", index_col=0, parse_dates=True)
    benchmark_df = pd.read_csv(OUTPUT_DIR / "benchmark_results.csv", index_col=0)
    return forecasts_df, benchmark_df

def create_performance_comparison_chart(benchmark_df):
    """Create a bar chart comparing model performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = benchmark_df.index
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # RMSE (lower is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, benchmark_df['RMSE'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('Forecast Error (RMSE) - Lower is Better', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, benchmark_df['RMSE'].max() * 1.1)
    for i, (bar, val) in enumerate(zip(bars1, benchmark_df['RMSE'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Hit Rate (higher is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, benchmark_df['HitRate'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax2.set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Directional Accuracy - Higher is Better', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 60)
    ax2.legend()
    for i, (bar, val) in enumerate(zip(bars2, benchmark_df['HitRate'] * 100)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Sharpe Ratio (higher is better)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, benchmark_df['Sharpe(approx)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero (no edge)')
    ax3.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Risk-Adjusted Returns - Higher is Better', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend()
    for i, (bar, val) in enumerate(zip(bars3, benchmark_df['Sharpe(approx)'])):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 if val > 0 else bar.get_height() - 0.1,
                    f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=10)
    
    # Annual Return
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, benchmark_df['AnnRet(approx)'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero return')
    ax4.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Annualized Returns (Approx.)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    for i, (bar, val) in enumerate(zip(bars4, benchmark_df['AnnRet(approx)'] * 100)):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1 if val > 0 else bar.get_height() - 2,
                    f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'performance_comparison.png'}")

def create_forecast_vs_actual_plot(forecasts_df):
    """Create time series plot showing forecasts vs actual returns."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: All models forecasts vs actual
    ax1 = axes[0]
    ax1.plot(forecasts_df.index, forecasts_df['y_true'], 'k-', linewidth=2, label='Actual Returns', alpha=0.7)
    ax1.plot(forecasts_df.index, forecasts_df['arimax'], 'b-', linewidth=1.5, label='ARIMAX Forecast', alpha=0.8)
    ax1.plot(forecasts_df.index, forecasts_df['ols'], 'g--', linewidth=1.5, label='OLS Forecast', alpha=0.7)
    ax1.plot(forecasts_df.index, forecasts_df['ar1'], 'r:', linewidth=1.5, label='AR(1) Forecast', alpha=0.6)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Weekly Return', fontsize=12, fontweight='bold')
    ax1.set_title('Model Forecasts vs Actual Returns (Out-of-Sample)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ARIMAX forecast vs actual (zoomed)
    ax2 = axes[1]
    ax2.plot(forecasts_df.index, forecasts_df['y_true'], 'ko-', markersize=3, label='Actual Returns', alpha=0.6)
    ax2.plot(forecasts_df.index, forecasts_df['arimax'], 'b^-', markersize=3, label='ARIMAX Forecast', alpha=0.8)
    ax2.fill_between(forecasts_df.index, forecasts_df['y_true'], forecasts_df['arimax'], 
                     where=(forecasts_df['y_true'] >= forecasts_df['arimax']), 
                     alpha=0.3, color='green', label='Over-prediction')
    ax2.fill_between(forecasts_df.index, forecasts_df['y_true'], forecasts_df['arimax'], 
                     where=(forecasts_df['y_true'] < forecasts_df['arimax']), 
                     alpha=0.3, color='red', label='Under-prediction')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weekly Return', fontsize=12, fontweight='bold')
    ax2.set_title('ARIMAX Forecast vs Actual (Best Model)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecasts_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'forecasts_vs_actual.png'}")

def create_cumulative_returns_plot(forecasts_df):
    """Create cumulative returns plot for trading strategy."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate cumulative returns for each model (simple long-only strategy)
    strategies = {}
    for model in ['naive', 'ar1', 'ols', 'arimax']:
        signals = (forecasts_df[model] > 0).astype(float)
        returns = signals * forecasts_df['y_true']
        strategies[model] = (1 + returns).cumprod()
    
    # Plot cumulative returns
    ax.plot(forecasts_df.index, strategies['naive'], 'k-', linewidth=2, label='Naive (Flat)', alpha=0.7)
    ax.plot(forecasts_df.index, strategies['ar1'], 'r-', linewidth=1.5, label='AR(1)', alpha=0.7)
    ax.plot(forecasts_df.index, strategies['ols'], 'g-', linewidth=1.5, label='OLS', alpha=0.7)
    ax.plot(forecasts_df.index, strategies['arimax'], 'b-', linewidth=2, label='ARIMAX (Best)', alpha=0.9)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Returns: Long-Only Strategy Based on Model Forecasts', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add final values as text
    for model, cumret in strategies.items():
        final_val = cumret.iloc[-1]
        ax.text(forecasts_df.index[-1], final_val, f'  {model.upper()}: {final_val:.3f}', 
               fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cumulative_returns.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'cumulative_returns.png'}")

def create_scatter_forecast_vs_actual(forecasts_df):
    """Create scatter plot of forecasts vs actuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # OLS scatter
    ax1 = axes[0]
    ax1.scatter(forecasts_df['ols'], forecasts_df['y_true'], alpha=0.5, s=30, color='green')
    # Perfect prediction line
    min_val = min(forecasts_df['ols'].min(), forecasts_df['y_true'].min())
    max_val = max(forecasts_df['ols'].max(), forecasts_df['y_true'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('OLS Forecast', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Return', fontsize=12, fontweight='bold')
    ax1.set_title('OLS: Forecast vs Actual', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ARIMAX scatter
    ax2 = axes[1]
    ax2.scatter(forecasts_df['arimax'], forecasts_df['y_true'], alpha=0.5, s=30, color='blue')
    min_val = min(forecasts_df['arimax'].min(), forecasts_df['y_true'].min())
    max_val = max(forecasts_df['arimax'].max(), forecasts_df['y_true'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('ARIMAX Forecast', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual Return', fontsize=12, fontweight='bold')
    ax2.set_title('ARIMAX: Forecast vs Actual (Best Model)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecast_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'forecast_scatter.png'}")

def create_performance_table(benchmark_df):
    """Create a nicely formatted performance table."""
    # Create a formatted table
    table_df = benchmark_df.copy()
    table_df['HitRate'] = (table_df['HitRate'] * 100).round(1).astype(str) + '%'
    table_df['AnnRet(approx)'] = (table_df['AnnRet(approx)'] * 100).round(1).astype(str) + '%'
    table_df['AnnVol(approx)'] = (table_df['AnnVol(approx)'] * 100).round(1).astype(str) + '%'
    table_df['RMSE'] = table_df['RMSE'].round(4)
    table_df['MAE'] = table_df['MAE'].round(4)
    table_df['Sharpe(approx)'] = table_df['Sharpe(approx)'].round(2)
    table_df['TheilU'] = table_df['TheilU'].round(4)
    
    # Rename columns for better readability
    table_df.columns = ['RMSE', 'MAE', 'Hit Rate', 'Theil U', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Cum. Return', 'N']
    
    # Save as CSV
    table_df.to_csv(FIGURES_DIR / "performance_table.csv")
    
    # Also create a LaTeX-style table
    latex_table = table_df.to_latex(float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))
    with open(FIGURES_DIR / "performance_table.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"Saved: {FIGURES_DIR / 'performance_table.csv'}")
    print(f"Saved: {FIGURES_DIR / 'performance_table.tex'}")
    
    return table_df

def create_trading_signals_visualization(forecasts_df):
    """Create a visualization showing trading signals over time."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Forecasts and signals
    ax1 = axes[0]
    ax1.plot(forecasts_df.index, forecasts_df['arimax'], 'b-', linewidth=1.5, label='ARIMAX Forecast', alpha=0.8)
    ax1.fill_between(forecasts_df.index, 0, forecasts_df['arimax'], 
                     where=(forecasts_df['arimax'] > 0), 
                     alpha=0.3, color='green', label='Long Signal')
    ax1.fill_between(forecasts_df.index, 0, forecasts_df['arimax'], 
                     where=(forecasts_df['arimax'] < 0), 
                     alpha=0.3, color='red', label='Short/Flat Signal')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Forecast Return', fontsize=12, fontweight='bold')
    ax1.set_title('ARIMAX Trading Signals Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual returns colored by signal correctness
    ax2 = axes[1]
    correct_signals = (np.sign(forecasts_df['arimax']) == np.sign(forecasts_df['y_true']))
    ax2.scatter(forecasts_df.index[correct_signals], forecasts_df['y_true'][correct_signals], 
               c='green', s=30, alpha=0.6, label='Correct Signal')
    ax2.scatter(forecasts_df.index[~correct_signals], forecasts_df['y_true'][~correct_signals], 
               c='red', s=30, alpha=0.6, label='Incorrect Signal')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual Return', fontsize=12, fontweight='bold')
    ax2.set_title('Actual Returns: Green = Correct Signal, Red = Incorrect Signal', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trading_signals.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'trading_signals.png'}")

def create_error_distribution_plot(forecasts_df):
    """Create histogram of forecast errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # OLS errors
    ols_errors = forecasts_df['y_true'] - forecasts_df['ols']
    ax1 = axes[0]
    ax1.hist(ols_errors, bins=40, color='green', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=ols_errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {ols_errors.mean():.4f}')
    ax1.set_xlabel('Forecast Error', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('OLS Forecast Error Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ARIMAX errors
    arimax_errors = forecasts_df['y_true'] - forecasts_df['arimax']
    ax2 = axes[1]
    ax2.hist(arimax_errors, bins=40, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=arimax_errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {arimax_errors.mean():.4f}')
    ax2.set_xlabel('Forecast Error', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('ARIMAX Forecast Error Distribution (Best Model)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'error_distribution.png'}")

def main():
    print("=" * 80)
    print("Creating Report Visualizations")
    print("=" * 80)
    
    # Load data
    forecasts_df, benchmark_df = load_data()
    print(f"\nLoaded {len(forecasts_df)} forecast observations")
    print(f"Loaded {len(benchmark_df)} model results\n")
    
    # Create visualizations
    print("Creating visualizations...")
    create_performance_comparison_chart(benchmark_df)
    create_forecast_vs_actual_plot(forecasts_df)
    create_cumulative_returns_plot(forecasts_df)
    create_scatter_forecast_vs_actual(forecasts_df)
    create_trading_signals_visualization(forecasts_df)
    create_error_distribution_plot(forecasts_df)
    
    # Create tables
    print("\nCreating tables...")
    table_df = create_performance_table(benchmark_df)
    
    print("\n" + "=" * 80)
    print("All visualizations and tables created successfully!")
    print(f"Output directory: {FIGURES_DIR.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()


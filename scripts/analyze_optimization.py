#!/usr/bin/env python3
"""
Analyze Hyperparameter Optimization Results
============================================

Quickly visualize and analyze optimization results.

Usage:
    python scripts/analyze_optimization.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results/optimization")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def find_latest_results():
    """Find the most recent optimization results file."""
    csv_files = list(RESULTS_DIR.glob("optimization_results_*.csv"))
    if not csv_files:
        print("❌ No optimization results found!")
        print(f"   Expected location: {RESULTS_DIR}/optimization_results_*.csv")
        return None
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"✓ Found results: {latest.name}")
    return latest


def load_and_display_results(csv_path):
    """Load and display optimization results."""
    df = pd.read_csv(csv_path)

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)

    # Top 5 models
    print("\nTop 5 Models (sorted by MAE Test):")
    print("-" * 80)
    top5 = df.head(5)[['Rank', 'Model', 'MAE_Test', 'R2_Test', 'Overfitting_Ratio', 'N_Params']]
    print(top5.to_string(index=False))

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Number of models optimized: {len(df)}")
    print(f"Best MAE Test: {df['MAE_Test'].min():.3f} years ({df.loc[df['MAE_Test'].idxmin(), 'Model']})")
    print(f"Worst MAE Test: {df['MAE_Test'].max():.3f} years ({df.loc[df['MAE_Test'].idxmax(), 'Model']})")
    print(f"Mean MAE Test: {df['MAE_Test'].mean():.3f} years")
    print(f"Best R² Test: {df['R2_Test'].max():.4f} ({df.loc[df['R2_Test'].idxmax(), 'Model']})")
    print(f"Total optimization time: {df['Optimization_Time_Min'].sum():.1f} minutes ({df['Optimization_Time_Min'].sum()/60:.1f} hours)")

    # Overfitting analysis
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)
    print(f"Models with overfitting < 2.0x: {(df['Overfitting_Ratio'] < 2.0).sum()}/{len(df)}")
    print(f"Models with overfitting < 3.0x: {(df['Overfitting_Ratio'] < 3.0).sum()}/{len(df)}")
    print(f"Models with overfitting > 5.0x: {(df['Overfitting_Ratio'] > 5.0).sum()}/{len(df)}")

    # Best by category
    print("\n" + "="*80)
    print("BEST BY CATEGORY")
    print("="*80)

    linear_models = df[df['Model'].isin(['Ridge', 'Lasso', 'ElasticNet'])]
    if len(linear_models) > 0:
        best_linear = linear_models.loc[linear_models['MAE_Test'].idxmin()]
        print(f"Best Linear Model: {best_linear['Model']} (MAE: {best_linear['MAE_Test']:.3f})")

    ensemble_models = df[df['Model'].isin(['RandomForest', 'GradientBoosting'])]
    if len(ensemble_models) > 0:
        best_ensemble = ensemble_models.loc[ensemble_models['MAE_Test'].idxmin()]
        print(f"Best Classic Ensemble: {best_ensemble['Model']} (MAE: {best_ensemble['MAE_Test']:.3f})")

    boosting_models = df[df['Model'].isin(['XGBoost', 'LightGBM', 'CatBoost'])]
    if len(boosting_models) > 0:
        best_boosting = boosting_models.loc[boosting_models['MAE_Test'].idxmin()]
        print(f"Best Gradient Boosting: {best_boosting['Model']} (MAE: {best_boosting['MAE_Test']:.3f})")

    return df


def create_visualizations(df, output_dir):
    """Create visualization plots."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. MAE Test comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: MAE Test
    ax = axes[0, 0]
    df_sorted = df.sort_values('MAE_Test')
    colors = ['#2ecc71' if r < 2.0 else '#f39c12' if r < 3.0 else '#e74c3c' for r in df_sorted['Overfitting_Ratio']]
    ax.barh(df_sorted['Model'], df_sorted['MAE_Test'], color=colors)
    ax.set_xlabel('MAE Test (years)')
    ax.set_title('Model Performance (MAE Test)\nGreen: Overfitting <2x, Orange: <3x, Red: ≥3x')
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: R² Test
    ax = axes[0, 1]
    df_sorted = df.sort_values('R2_Test', ascending=False)
    ax.barh(df_sorted['Model'], df_sorted['R2_Test'], color='#3498db')
    ax.set_xlabel('R² Test')
    ax.set_title('Model Performance (R² Test)')
    ax.grid(axis='x', alpha=0.3)

    # Plot 3: Overfitting Ratio
    ax = axes[1, 0]
    df_sorted = df.sort_values('Overfitting_Ratio')
    colors = ['#2ecc71' if r < 2.0 else '#f39c12' if r < 3.0 else '#e74c3c' for r in df_sorted['Overfitting_Ratio']]
    ax.barh(df_sorted['Model'], df_sorted['Overfitting_Ratio'], color=colors)
    ax.axvline(2.0, color='green', linestyle='--', linewidth=2, label='Excellent (<2x)')
    ax.axvline(3.0, color='orange', linestyle='--', linewidth=2, label='Good (<3x)')
    ax.axvline(5.0, color='red', linestyle='--', linewidth=2, label='Limit (5x)')
    ax.set_xlabel('Overfitting Ratio (MAE_Test / MAE_Train)')
    ax.set_title('Overfitting Analysis')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Plot 4: Optimization time vs Performance
    ax = axes[1, 1]
    scatter = ax.scatter(
        df['Optimization_Time_Min'],
        df['MAE_Test'],
        s=df['N_Params']/10,
        c=df['R2_Test'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    for idx, row in df.iterrows():
        ax.annotate(
            row['Model'],
            (row['Optimization_Time_Min'], row['MAE_Test']),
            fontsize=8,
            ha='center'
        )
    ax.set_xlabel('Optimization Time (minutes)')
    ax.set_ylabel('MAE Test (years)')
    ax.set_title('Optimization Time vs Performance\n(Size = N_Params, Color = R² Test)')
    plt.colorbar(scatter, ax=ax, label='R² Test')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "optimization_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")

    # 2. Detailed comparison table plot
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')

    table_data = df[['Rank', 'Model', 'MAE_Test', 'R2_Test', 'Overfitting_Ratio', 'N_Params', 'Optimization_Time_Min']].copy()
    table_data['MAE_Test'] = table_data['MAE_Test'].apply(lambda x: f"{x:.3f}")
    table_data['R2_Test'] = table_data['R2_Test'].apply(lambda x: f"{x:.4f}")
    table_data['Overfitting_Ratio'] = table_data['Overfitting_Ratio'].apply(lambda x: f"{x:.2f}x")
    table_data['Optimization_Time_Min'] = table_data['Optimization_Time_Min'].apply(lambda x: f"{x:.1f}")

    table = ax.table(
        cellText=table_data.values,
        colLabels=['Rank', 'Model', 'MAE Test', 'R² Test', 'Overfit', 'Params', 'Time (min)'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color rows
    for i in range(len(df)):
        for j in range(7):
            cell = table[(i+1, j)]
            if i == 0:  # Best model
                cell.set_facecolor('#d4edda')
            elif i < 3:  # Top 3
                cell.set_facecolor('#fff3cd')

    # Bold headers
    for j in range(7):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(weight='bold', color='white')

    plt.title('Complete Optimization Results', fontsize=14, weight='bold', pad=20)
    table_path = output_dir / "optimization_table.png"
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved table: {table_path}")

    plt.show()


def load_hyperparameters():
    """Load and display best hyperparameters."""
    hyperparam_files = list(RESULTS_DIR.glob("best_hyperparameters_*.csv"))
    if not hyperparam_files:
        print("\n⚠️  No hyperparameters file found")
        return

    latest = max(hyperparam_files, key=lambda p: p.stat().st_mtime)
    df_params = pd.read_csv(latest)

    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS")
    print("="*80)

    for model in df_params['Model'].unique():
        print(f"\n{model}:")
        print("-" * 40)
        model_params = df_params[df_params['Model'] == model]
        for _, row in model_params.iterrows():
            print(f"  {row['Parameter']}: {row['Value']}")


def main():
    """Main analysis workflow."""
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION - RESULTS ANALYSIS")
    print("="*80)

    # Find latest results
    csv_path = find_latest_results()
    if csv_path is None:
        return

    # Load and display
    df = load_and_display_results(csv_path)

    # Create visualizations
    create_visualizations(df, RESULTS_DIR)

    # Load hyperparameters
    load_hyperparameters()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  - {RESULTS_DIR}/optimization_analysis.png")
    print(f"  - {RESULTS_DIR}/optimization_table.png")


if __name__ == "__main__":
    main()

"""
Script to regenerate all plots from existing results.csv files.
This script does not rerun any experiments - it only recreates the plots with integer k-axis values.
You can delete this script after running it.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_project_root():
    """Returns the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def regenerate_individual_plots(results_path, df, dataset_name, reg_folder):
    """
    Regenerate individual plots (non-scaled) for a single results.csv file.
    """
    k_values = df['k_value'].dropna().astype(int)
    k_min, k_max = int(k_values.min()), int(k_values.max())
    
    # Noise Robustness Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    for i, nl in enumerate([0.1, 0.2, 0.3]):
        if f'noise_{nl}' in df.columns:
            plt.plot(df['k_value'], df[f'noise_{nl}'], 'o-', color=colors[i], linewidth=2, label=f'Noise level: {nl}')
    if 'baseline_accuracy' in df.columns:
        plt.plot(df['k_value'], df['baseline_accuracy'], 'k--', linewidth=2, label='Baseline (no noise)')
    plt.xlabel('k-additivity')
    plt.ylabel('Accuracy')
    plt.title(f'Noise Robustness vs k-additivity ({dataset_name}, {reg_folder})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(k_min, k_max + 1))  # Ensure integer k values on x-axis
    plt.savefig(os.path.join(results_path, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bootstrap Stability Plot
    if 'bootstrap_mean' in df.columns and 'bootstrap_std' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['k_value'], df['bootstrap_mean'], yerr=df['bootstrap_std'], fmt='o-', capsize=5)
        plt.xlabel('k-additivity')
        plt.ylabel('Bootstrap Accuracy (mean ± std)')
        plt.title(f'Bootstrap Stability vs k-additivity ({dataset_name}, {reg_folder})')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(k_min, k_max + 1))  # Ensure integer k values on x-axis
        plt.savefig(os.path.join(results_path, 'bootstrap_stability.png'), dpi=300, bbox_inches='tight')
        plt.close()


def regenerate_scaled_plots(all_dfs, benchmark_root, y_lims):
    """
    Regenerate scaled plots with shared y-axis across all datasets.
    """
    for key, df in all_dfs.items():
        parts = key.rsplit('_', 1)  # Split from right: dataset_reg
        dataset = parts[0]
        reg_folder = parts[1]
        
        results_path = os.path.join(benchmark_root, dataset, reg_folder)
        
        k_values = df['k_value'].dropna().astype(int)
        k_min, k_max = int(k_values.min()), int(k_values.max())
        
        # Noise plot (scaled)
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, 3))
        for i, nl in enumerate([0.1, 0.2, 0.3]):
            if f'noise_{nl}' in df.columns:
                plt.plot(df['k_value'], df[f'noise_{nl}'], 'o-', color=colors[i], linewidth=2, label=f'Noise level: {nl}')
        if 'baseline_accuracy' in df.columns:
            plt.plot(df['k_value'], df['baseline_accuracy'], 'k--', linewidth=2, label='Baseline (no noise)')
        plt.xlabel('k-additivity')
        plt.ylabel('Accuracy')
        plt.title(f'Noise Robustness vs k-additivity ({dataset}, {reg_folder})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(y_lims)
        plt.xticks(range(k_min, k_max + 1))  # Ensure integer k values on x-axis
        out_path = os.path.join(results_path, 'noise_robustness_scaled.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bootstrap plot (scaled)
        if 'bootstrap_mean' in df.columns and 'bootstrap_std' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.errorbar(df['k_value'], df['bootstrap_mean'], yerr=df['bootstrap_std'], fmt='o-', capsize=5)
            plt.xlabel('k-additivity')
            plt.ylabel('Bootstrap Accuracy (mean ± std)')
            plt.title(f'Bootstrap Stability vs k-additivity ({dataset}, {reg_folder})')
            plt.grid(True, alpha=0.3)
            plt.ylim(y_lims)
            plt.xticks(range(k_min, k_max + 1))  # Ensure integer k values on x-axis
            out_path = os.path.join(results_path, 'bootstrap_stability_scaled.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()


def main():
    root_dir = get_project_root()
    benchmark_root = os.path.join(root_dir, 'results', 'benchmark')
    
    if not os.path.exists(benchmark_root):
        print(f"No results found in {benchmark_root}")
        return
    
    # Find all datasets (top-level folders in benchmark, excluding 'bounds')
    datasets = [d for d in os.listdir(benchmark_root) 
                if os.path.isdir(os.path.join(benchmark_root, d)) and d != 'bounds']
    
    if not datasets:
        print("No dataset folders found.")
        return
    
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    # Collect all data to find global y-limits across all regularizations
    all_dfs = {}
    for dataset in datasets:
        dataset_path = os.path.join(benchmark_root, dataset)
        
        # Check all regularization subfolders
        for reg_folder in ['None', 'l1', 'l2']:
            reg_path = os.path.join(dataset_path, reg_folder)
            csv_path = os.path.join(reg_path, 'results.csv')
            if os.path.exists(csv_path):
                key = f"{dataset}_{reg_folder}"
                all_dfs[key] = pd.read_csv(csv_path, index_col=0)
                print(f"  Loaded: {csv_path}")
    
    if not all_dfs:
        print("No results CSV files found.")
        return
    
    print(f"\nLoaded {len(all_dfs)} result files.")
    
    # Calculate global y-limits for scaled plots
    min_acc, max_acc = 1.0, 0.0
    for df in all_dfs.values():
        cols = ['baseline_accuracy', 'noise_0.1', 'noise_0.2', 'noise_0.3', 'bootstrap_mean']
        for col in cols:
            if col in df.columns:
                if col == 'bootstrap_mean':
                    std_col = 'bootstrap_std'
                    if std_col in df.columns:
                        min_val = (df[col] - df[std_col]).min()
                        max_val = (df[col] + df[std_col]).max()
                    else:
                        min_val = df[col].min()
                        max_val = df[col].max()
                else:
                    min_val = df[col].min()
                    max_val = df[col].max()
                
                if not np.isnan(min_val):
                    min_acc = min(min_acc, min_val)
                if not np.isnan(max_val):
                    max_acc = max(max_acc, max_val)
    
    y_margin = (max_acc - min_acc) * 0.05
    y_lims = (max(0, min_acc - y_margin), min(1, max_acc + y_margin))
    
    print(f"Shared y-axis limits: [{y_lims[0]:.3f}, {y_lims[1]:.3f}]")
    
    # Regenerate individual plots for each result set
    print("\nRegenerating individual plots...")
    for key, df in all_dfs.items():
        parts = key.rsplit('_', 1)
        dataset = parts[0]
        reg_folder = parts[1]
        results_path = os.path.join(benchmark_root, dataset, reg_folder)
        
        regenerate_individual_plots(results_path, df, dataset, reg_folder)
        print(f"  Regenerated plots for: {dataset}/{reg_folder}")
    
    # Regenerate scaled plots
    print("\nRegenerating scaled plots...")
    regenerate_scaled_plots(all_dfs, benchmark_root, y_lims)
    print(f"  Scaled plots saved for all {len(all_dfs)} result sets")
    
    print("\n" + "=" * 60)
    print("All plots regenerated successfully!")
    print("You can now delete this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()

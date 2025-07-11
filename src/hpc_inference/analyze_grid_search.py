import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_grid_search_result(grid_search_dir):
    
    results = []
    for subdir in os.listdir(grid_search_dir):
        profile_dir = os.path.join(grid_search_dir, subdir)
        specs_path = os.path.join(profile_dir, "computing_specs.json")
        if os.path.isfile(specs_path):
            with open(specs_path, "r") as f:
                specs = json.load(f)
                batch_size = specs.get("batch_size")
                num_workers = specs.get("num_workers")
                throughput = specs.get("throughput")
                if isinstance(throughput, str):
                    throughput_val = float(throughput.split()[0])
                else:
                    throughput_val = throughput
                results.append({
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "throughput": throughput_val
                })
    return pd.DataFrame(results)

def save_throughput_heatmap(df, grid_search_dir):
    pivot = df.pivot(index="batch_size", columns="num_workers", values="throughput")
    plt.figure(figsize=(8,6))
    plt.title("Throughput (images/sec) by Batch Size and Num Workers")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.ylabel("Batch Size")
    plt.xlabel("Num Workers")
    plt.tight_layout()
    plt.savefig(os.path.join(grid_search_dir, "grid_search_throughput_heatmap.png"))
    plt.close()

def save_gradient_plot(df, grid_search_dir):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: Fix batch size, vary num_workers
    for batch_size in sorted(df['batch_size'].unique()):
        subset = df[df['batch_size'] == batch_size]
        axes[0].plot(subset['num_workers'], subset['throughput'], marker='o', label=f'Batch {batch_size}')
    axes[0].set_xlabel("Num Workers")
    axes[0].set_ylabel("Throughput (images/sec)")
    axes[0].set_title("Throughput vs Num Workers\n(fixed Batch Size)")
    axes[0].legend()

    # Right: Fix num_workers, vary batch size
    for num_workers in sorted(df['num_workers'].unique()):
        subset = df[df['num_workers'] == num_workers]
        axes[1].plot(subset['batch_size'], subset['throughput'], marker='o', label=f'Workers {num_workers}')
    axes[1].set_xlabel("Batch Size")
    axes[1].set_title("Throughput vs Batch Size\n(fixed Num Workers)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(grid_search_dir, "grid_search_gradient_plot.png"))
    plt.close()

def save_3d_surface_plot(df, grid_search_dir):
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata
    import numpy as np

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    x = df['num_workers']
    y = df['batch_size']
    z = df['throughput']

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Num Workers')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Throughput (images/sec)')
    ax.set_title('Throughput Surface Plot')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(os.path.join(grid_search_dir, "grid_search_3d_surface_plot.png"))
    plt.close()

def main(grid_search_dir):
    df = parse_grid_search_result(grid_search_dir)
    if df.empty:
        print("No valid profiling data found.")
        return
    df.to_csv(os.path.join(grid_search_dir, "grid_search_summary.csv"), index=False)
    save_throughput_heatmap(df, grid_search_dir)
    save_gradient_plot(df, grid_search_dir)
    save_3d_surface_plot(df, grid_search_dir)
    print("Grid search analysis complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Grid Search Results")
    parser.add_argument("grid_search_dir", type=str, help="Directory containing grid search profiling results")
    args = parser.parse_args()

    main(args.grid_search_dir)
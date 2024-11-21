import os
import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils import time_function
from random import sample

@time_function
def evaluate_assignments(assignments_dict, vehicles, log_lines, location_dict, output_dir="output/plots"):
    """
    Evaluate all assignment methods and generate comparison plots.

    Parameters:
        assignments_dict (dict): Dictionary of assignments with method names as keys.
        vehicles (pd.DataFrame): Vehicle details.
        log_lines (list): Parsed log data (not used in this implementation but retained for extensibility).
        location_dict (dict): Preprocessed location data.
        output_dir (str): Directory to save the plots and evaluation results.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    evaluation_results = {}

    # Calculate metrics for each method
    for method_name, assignments in assignments_dict.items():
        if 'sc_total_distance' not in assignments.columns:
            assignments['sc_total_distance'] = assignments.groupby('sc_id')['sc_total_distance'].transform('sum')
        total_distances = assignments.groupby('sc_id')['sc_total_distance'].sum()
        combined_total_distance = total_distances.sum()

        evaluation_results[method_name] = {
            "total_distances": total_distances,
            "combined_total_distance": combined_total_distance,
        }

    # Plot Total SC Distance Traveled (Per SC)
    plt.figure(figsize=(12, 6))
    for idx, (method_name, results) in enumerate(evaluation_results.items()):
        results["total_distances"].plot(
            kind="bar",
            alpha=0.7,
            label=method_name,
            color=plt.cm.tab10(idx),  # Unique color per method
            width=0.4,  # Narrow bars for grouped display
            position=idx
        )
    plt.title("Total SC Distance Traveled (Per SC)")
    plt.xlabel("Straddle Carrier ID")
    plt.ylabel("Total Distance (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_distance_per_sc_updated.png"))
    plt.close()

    # Plot Total SC Distance Traveled (Combined Sum)
    plt.figure(figsize=(8, 6))
    combined_distances = {
        method_name: results["combined_total_distance"] for method_name, results in evaluation_results.items()
    }
    pd.Series(combined_distances).plot(kind="bar", alpha=0.7, color=plt.cm.Set3.colors[:len(combined_distances)])
    plt.title("Total SC Distance Traveled (Combined Sum)")
    plt.xlabel("Assignment Method")
    plt.ylabel("Total Distance (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_distance_combined_updated.png"))
    plt.close()

    # Random sample of 10 containers' start and end times
    method_names = list(assignments_dict.keys())
    sample_task_ids = sample(list(assignments_dict[method_names[0]]['task_id']), min(10, len(assignments_dict[method_names[0]])))
    sample_comparison = []

    for method_name, assignments in assignments_dict.items():
        sample_data = assignments[assignments['task_id'].isin(sample_task_ids)]
        sample_data = sample_data[['task_id', 'container_start_time', 'finish_time']].set_index('task_id')
        sample_data.rename(columns={
            'container_start_time': f'{method_name}_start',
            'finish_time': f'{method_name}_finish'
        }, inplace=True)
        sample_comparison.append(sample_data)

    # Merge sample data from all methods
    sample_comparison_df = pd.concat(sample_comparison, axis=1)

    # Save sample data to CSV
    sample_file = os.path.join(output_dir, "sample_containers_comparison_updated.csv")
    sample_comparison_df.to_csv(sample_file)
    print(f"Saved 10 random sample containers comparison: {sample_file}")

    # Plot Start and Finish Times for 10 Random Containers
    plt.figure(figsize=(12, 6))
    for method_name in assignments_dict.keys():
        plt.plot(
            sample_task_ids,
            pd.to_datetime(sample_comparison_df[f"{method_name}_start"]),
            label=f"{method_name} Start",
            linestyle="--",
            marker="o",
        )
        plt.plot(
            sample_task_ids,
            pd.to_datetime(sample_comparison_df[f"{method_name}_finish"]),
            label=f"{method_name} Finish",
            linestyle="-",
            marker="s",
        )
    plt.title("Start and Finish Times for 10 Random Containers (Comparison)")
    plt.xlabel("Random Container Task ID")
    plt.ylabel("Time (YYYY-MM-DD HH:MM:SS)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_containers_start_finish_comparison_updated.png"))
    plt.close()

    print(f"Evaluation plots saved to {output_dir}")

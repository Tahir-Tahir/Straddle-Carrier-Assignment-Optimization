import os
from matplotlib import pyplot as plt
import pandas as pd
import re
from tqdm import tqdm
from scripts.utils import time_function
import seaborn as sns

@time_function
def parse_logs(log_lines):
    """
    Parse the log file to extract travel details, SC schedules, and location usage.

    Parameters:
        log_lines (list): Lines from the log file.

    Returns:
        pd.DataFrame, pd.DataFrame: Parsed travel details and location usage.
    """
    travel_details = []
    location_usage = []
    sc_schedules = []

    for line in tqdm(log_lines, desc="Parsing logs"):
        # Extract travel details
        travel_match = re.search(
            r"INFO (SC\d+) .*driving to (\w+); (\d+) s; (\d+) mm", line
        )
        if travel_match:
            travel_details.append({
                "sc_id": travel_match.group(1),
                "destination": travel_match.group(2),
                "travel_time_s": int(travel_match.group(3)),
                "travel_distance_mm": int(travel_match.group(4)),
            })

        # Extract location usage
        location_match = re.search(
            r"DEBUG location (\w+): (using|freeing) lane (\d+) for CO (\w+)", line
        )
        if location_match:
            location_usage.append({
                "location": location_match.group(1),
                "action": location_match.group(2),
                "lane": int(location_match.group(3)),
                "container_id": location_match.group(4),
            })

        # Extract SC schedules
        schedule_match = re.search(
            r"INFO (SC\d+) schedule (.*)", line
        )
        if schedule_match:
            sc_schedules.append({
                "sc_id": schedule_match.group(1),
                "schedule": schedule_match.group(2),
            })

    travel_df = pd.DataFrame(travel_details)
    location_usage_df = pd.DataFrame(location_usage)
    schedules_df = pd.DataFrame(sc_schedules)

    print(f"Parsed {len(travel_df)} travel details.")
    print(f"Parsed {len(location_usage_df)} location usage details.")
    print(f"Parsed {len(schedules_df)} SC schedules.")

    return travel_df, location_usage_df, schedules_df

@time_function
def visualize_logs(travel_df, location_usage_df, schedules_df, output_dir="output/plots_logs"):
    """
    Generate visualizations from parsed log details.

    Parameters:
        travel_df (pd.DataFrame): Parsed travel details.
        location_usage_df (pd.DataFrame): Parsed location usage details.
        schedules_df (pd.DataFrame): Parsed SC schedules.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Total Travel Time and Distance per SC
    if not travel_df.empty:
        travel_summary = travel_df.groupby("sc_id").agg({
            "travel_time_s": "sum",
            "travel_distance_mm": "sum"
        }).reset_index()

        plt.figure(figsize=(12, 6))
        travel_summary.plot(
            x="sc_id",
            kind="bar",
            stacked=True,
            title="Total Travel Time and Distance per SC",
            xlabel="Straddle Carrier ID",
            ylabel="Value",
            figsize=(12, 6)
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "travel_summary.png"))
        plt.close()

    # Location Usage Heatmap
    if not location_usage_df.empty:
        location_usage_counts = location_usage_df.groupby("location")["action"].count().reset_index()
        location_usage_counts = location_usage_counts.rename(columns={"action": "usage_count"})

        plt.figure(figsize=(10, 8))
        sns.barplot(x="location", y="usage_count", data=location_usage_counts)
        plt.title("Location Usage Count")
        plt.xlabel("Location")
        plt.ylabel("Usage Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "location_usage.png"))
        plt.close()

    # SC Schedules Overview
    if not schedules_df.empty:
        schedules_df["task_count"] = schedules_df["schedule"].apply(lambda x: len(x.split(",")))

        plt.figure(figsize=(12, 6))
        sns.barplot(x="sc_id", y="task_count", data=schedules_df)
        plt.title("Number of Tasks per SC")
        plt.xlabel("Straddle Carrier ID")
        plt.ylabel("Number of Tasks")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sc_schedules.png"))
        plt.close()

    print(f"Log visualizations saved to {output_dir}")

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_travel_summary_comparison(travel_df, proximity_assignments, output_dir="output/plots"):
    """
    Plot travel summary from log data alongside proximity assignment results.

    Parameters:
        travel_df (pd.DataFrame): Parsed travel details from logs.
        proximity_assignments (pd.DataFrame): Assignments from proximity-based optimization.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Summarize travel data from logs
    if not travel_df.empty:
        log_summary = travel_df.groupby("sc_id").agg({
            "travel_time_s": "sum",
            "travel_distance_mm": "sum"
        }).reset_index()
        log_summary.rename(columns={
            "travel_time_s": "log_travel_time_s",
            "travel_distance_mm": "log_travel_distance_mm"
        }, inplace=True)
    else:
        raise ValueError("Travel DataFrame is empty. Cannot create comparison plot.")

    # Summarize travel data from proximity assignments
    if not proximity_assignments.empty:
        proximity_summary = proximity_assignments.groupby("sc_id").agg({
            "sc_travel_time": "sum",
            "sc_total_distance": "sum"
        }).reset_index()
        proximity_summary.rename(columns={
            "sc_travel_time": "proximity_travel_time_s",
            "sc_total_distance": "proximity_travel_distance_mm"
        }, inplace=True)
    else:
        raise ValueError("Proximity assignments DataFrame is empty. Cannot create comparison plot.")

    # Merge the summaries for comparison
    comparison_df = pd.merge(log_summary, proximity_summary, on="sc_id", how="outer").fillna(0)

    # Plot comparison: Travel Distance
    plt.figure(figsize=(12, 6))
    comparison_df.plot(
        x="sc_id",
        y=["log_travel_distance_mm", "proximity_travel_distance_mm"],
        kind="bar",
        alpha=0.7,
        width=0.8,
        figsize=(12, 6),
        title="Total Travel Distance: Logs vs Proximity Assignments",
        xlabel="Straddle Carrier ID",
        ylabel="Travel Distance (mm)"
    )
    plt.legend(["Log Data", "Proximity Assignments"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "travel_distance_comparison.png"))
    plt.close()

    # Plot comparison: Travel Time
    plt.figure(figsize=(12, 6))
    comparison_df.plot(
        x="sc_id",
        y=["log_travel_time_s", "proximity_travel_time_s"],
        kind="bar",
        alpha=0.7,
        width=0.8,
        figsize=(12, 6),
        title="Total Travel Time: Logs vs Proximity Assignments",
        xlabel="Straddle Carrier ID",
        ylabel="Travel Time (s)"
    )
    plt.legend(["Log Data", "Proximity Assignments"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "travel_time_comparison.png"))
    plt.close()

    print(f"Travel summary comparison plots saved to {output_dir}")

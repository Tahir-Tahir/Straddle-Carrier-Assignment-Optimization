import pandas as pd
import time

from scripts.utils import time_function

@time_function
def calculate_kpis(assignments, vehicles, parsed_logs, location_dict):
    """
    Calculate KPIs (distance, idle time, penalty time) for each SC.

    Parameters:
        assignments (pd.DataFrame): Task assignments for SCs (DataFrame format).
        vehicles (pd.DataFrame): Vehicle details.
        parsed_logs (dict): Parsed log data (if applicable).
        location_dict (dict): Preprocessed location data.

    Returns:
        pd.DataFrame: KPI summary for each SC.
    """
    kpi_results = []

    for _, vehicle in vehicles.iterrows():
        sc_id = vehicle['ID']
        log_on = pd.to_datetime(vehicle['LogOn'])
        log_off = pd.to_datetime(vehicle['LogOff'])
        total_logged_in_time = (log_off - log_on).total_seconds()

        # Filter assignments for this SC
        sc_assignments = assignments[assignments['sc_id'] == sc_id]

        # Total distance
        total_distance = sc_assignments['travel_time'].sum() * 5555  # Convert travel time to mm

        # Active time (travel and task completion)
        total_active_time = sc_assignments['travel_time'].sum()

        # Idle time
        total_idle_time = max(0, total_logged_in_time - total_active_time)

        # Append KPI results for this SC
        kpi_results.append({
            "sc_id": sc_id,
            "total_distance": total_distance,
            "total_active_time": total_active_time,
            "total_idle_time": total_idle_time,
            "tasks": len(sc_assignments)
        })

    # Convert KPI results to DataFrame
    kpi_df = pd.DataFrame(kpi_results)
    return kpi_df

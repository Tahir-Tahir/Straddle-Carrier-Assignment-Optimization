import pandas as pd
import time
from tqdm import tqdm
import re

from scripts.utils import time_function

@time_function
def clean_dataframe(df):
    """Strip leading/trailing spaces from all string columns in a DataFrame."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df

@time_function
def load_data(file_path_excel, log_file_path):
    """Load Excel and log files and clean all data."""
    locations = pd.read_excel(file_path_excel, sheet_name='Locations')
    vehicles = pd.read_excel(file_path_excel, sheet_name='Vehicles')
    container_orders = pd.read_excel(file_path_excel, sheet_name='ContainerOrders')

    # Clean all DataFrames
    locations = clean_dataframe(locations)
    vehicles = clean_dataframe(vehicles)
    container_orders = clean_dataframe(container_orders)

    # Read log file
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    return locations, vehicles, container_orders, log_lines

@time_function
def preprocess_data(locations):
    """Preprocess locations data for quick lookup."""
    locations['Capacity limitation (# SC)'] = pd.to_numeric(locations['Capacity limitation (# SC)'], errors='coerce').fillna(-1)
    location_dict = locations.set_index('Location Name').to_dict('index')
    return location_dict


@time_function
def parse_logs(log_lines):
    """Parse the log file to extract useful insights."""
    travel_details = []

    for line in log_lines:
        # Extract travel details
        if "driving to" in line:
            match = re.search(r"INFO (SC\d+).*travel (\d+ s); (\d+ mm)", line)
            if match:
                travel_details.append({
                    "sc_id": match.group(1),
                    "travel_time_s": int(match.group(2).split()[0]),
                    "distance_mm": int(match.group(3).split()[0])
                })

    return {"travel_details": travel_details}
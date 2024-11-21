import os
from scripts.data_loader import load_data, parse_logs, preprocess_data
from scripts.assignment_methods import *
from scripts.evaluation import evaluate_assignments
from scripts.utils import time_function

# --- Main Script ---
if __name__ == "__main__":
    # File paths
    file_path_excel = 'data/VOSimu-InputInformation.xlsx'
    log_file_path = 'data/logger_all.log'
    OUTPUT_DIR = "output/"

    # Load Data
    locations, vehicles, container_orders, log_lines = load_data(file_path_excel, log_file_path)

    # Preprocess Data
    location_dict = preprocess_data(locations)

    # Parse Logs
    parsed_logs = parse_logs(log_lines)

    # Generate assignments
    print("Generating Random Assignments...")
    random_assignments = assign_jobs_randomly(container_orders, vehicles, location_dict, output_file=os.path.join(OUTPUT_DIR, "assignments_random.csv"))

    print("Generating Proximity + Prioritization Assignments...")
    proximity_assignments = assign_jobs_proximity_with_prioritization(container_orders, vehicles, location_dict, output_file=os.path.join(OUTPUT_DIR, "assignments_proximity_prioritization.csv"))

     # MIP Assignment
    print("MIP Assignment...")
    mip_assignments = assign_jobs_mip_min_distance(
        container_orders, vehicles, location_dict, 
        output_file=os.path.join(OUTPUT_DIR, "assignments_mip.csv")
    )

    print("Generating Genetic Algorithm Assignments...")
    genetic_assignments = assign_jobs_genetic(container_orders, vehicles, location_dict, output_file=os.path.join(OUTPUT_DIR, "assignments_genetic.csv"))


     # Evaluate Random vs Proximity + Prioritization
    print("Evaluating Random vs Proximity + Prioritization...")
    assignments_dict_step1 = {
        "Random": random_assignments,
        "Proximity + Prioritization": proximity_assignments,
    }
    evaluate_assignments(
        assignments_dict_step1, vehicles, log_lines, location_dict, 
        output_dir=os.path.join(OUTPUT_DIR, "plots_step1")
    )

    # Evaluate Proximity + Prioritization vs MIP
    print("Evaluating Proximity + Prioritization vs MIP...")
    assignments_dict_step2 = {
        "Proximity + Prioritization": proximity_assignments,
        "MIP": mip_assignments,
    }
    evaluate_assignments(
        assignments_dict_step2, vehicles, log_lines, location_dict, 
        output_dir=os.path.join(OUTPUT_DIR, "plots_step2")
    )

    # Evaluate all assignments (MIP vs Genetic)
    print("Evaluating assignments...")
    assignments_dict = {
        "MIP": mip_assignments,
        "Genetic": genetic_assignments,
    }
    evaluate_assignments(
        assignments_dict, vehicles, log_lines, location_dict,
          output_dir=os.path.join(OUTPUT_DIR, "plots_step3"))



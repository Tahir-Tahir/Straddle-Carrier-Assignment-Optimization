import os
import random
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import time
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from scripts.utils import calculate_manhattan_distance, time_function


@time_function
def assign_jobs_proximity(container_orders, vehicles, location_dict):
    """Assign tasks to the closest SC based on proximity."""
    assignments = []
    sc_locations = {vehicle['ID']: location_dict.get(vehicle['StartLocation']) for _, vehicle in vehicles.iterrows()}

    for _, order in tqdm(container_orders.iterrows(), total=len(container_orders), desc="Proximity Assignment"):
        origin = location_dict.get(order['OriginLocation'])
        destination = location_dict.get(order['DestinationLocation'])
        
        if origin is None or destination is None:
            continue  # Skip if origin or destination is missing
        
        origin_coords = (origin['X-Coordinate [mm]'], origin['Y-Coordinate [mm]'])
        dest_coords = (destination['X-Coordinate [mm]'], destination['Y-Coordinate [mm]'])

        # Find the closest SC
        closest_sc = None
        min_distance = float('inf')
        for sc_id, sc_location in sc_locations.items():
            if sc_location is None:
                continue
            sc_coords = (sc_location['X-Coordinate [mm]'], sc_location['Y-Coordinate [mm]'])
            distance = calculate_manhattan_distance(sc_coords, origin_coords)
            if distance < min_distance:
                closest_sc = sc_id
                min_distance = distance

        # Assign the task to the closest SC
        if closest_sc:
            travel_distance = min_distance + calculate_manhattan_distance(origin_coords, dest_coords)
            assignments.append({
                'task_id': order['ContainerOrderId'],
                'sc_id': closest_sc,
                'travel_distance': travel_distance,
                'idle_time': 0,
                'penalty_time': 0,  # Placeholder for penalty
            })

            # Update the SC's location to the container's destination
            sc_locations[closest_sc] = destination

    return assignments

@time_function
def assign_jobs_proximity_with_prioritization(container_orders, vehicles, location_dict, output_file="output/assignments_proximity_prioritization.csv", speed_mm_per_sec=5555):
    """
    Assign tasks to the closest SC based on proximity and prioritize by `time_first_known`.

    Parameters:
        container_orders (pd.DataFrame): Container order details.
        vehicles (pd.DataFrame): Vehicle details.
        location_dict (dict): Preprocessed location data.
        output_file (str): Path to save the assignments CSV.
        speed_mm_per_sec (int): Speed of the SC in mm/s (default: 5555 mm/s).

    Returns:
        pd.DataFrame: Assignments with time details and KPIs.
    """
    assignments = []

    # Map SCs to their start locations
    sc_current_locations = {
        vehicle['ID']: location_dict.get(vehicle['StartLocation'])
        for _, vehicle in vehicles.iterrows()
    }
    sc_next_available_time = {vehicle['ID']: pd.to_datetime(vehicle['LogOn']) for _, vehicle in vehicles.iterrows()}

    # Sort container orders by `time_first_known`
    container_orders = container_orders.sort_values(by="Time first known")

    for _, order in tqdm(container_orders.iterrows(), total=len(container_orders), desc="Proximity + Prioritization"):
        origin = location_dict.get(order['OriginLocation'])
        destination = location_dict.get(order['DestinationLocation'])
        time_first_known = pd.to_datetime(order['Time first known'])

        if origin is None or destination is None:
            print(f"Skipping task {order['ContainerOrderId']}: Missing origin/destination")
            continue  # Skip if origin or destination is missing

        origin_coords = (origin['X-Coordinate [mm]'], origin['Y-Coordinate [mm]'])
        dest_coords = (destination['X-Coordinate [mm]'], destination['Y-Coordinate [mm]'])

        # Find the closest available SC
        best_sc = None
        min_distance = float('inf')

        for sc_id, sc_location in sc_current_locations.items():
            if sc_location is None:
                continue  # Skip SCs with missing start location

            sc_coords = (sc_location['X-Coordinate [mm]'], sc_location['Y-Coordinate [mm]'])
            distance_to_container = calculate_manhattan_distance(sc_coords, origin_coords)

            # Check if SC is available to take the task
            sc_available_time = sc_next_available_time[sc_id]
            if sc_available_time <= time_first_known and distance_to_container < min_distance:
                best_sc = sc_id
                min_distance = distance_to_container

        if best_sc is None:
            print(f"No available SC for task {order['ContainerOrderId']}")
            continue

        # Calculate distances and times
        distance_to_container = min_distance
        travel_to_dest = calculate_manhattan_distance(origin_coords, dest_coords)
        total_sc_distance = distance_to_container + travel_to_dest
        container_distance = travel_to_dest

        time_to_container = distance_to_container / speed_mm_per_sec
        container_travel_time = container_distance / speed_mm_per_sec

        # Determine the start times
        sc_start_time = max(time_first_known - pd.Timedelta(seconds=time_to_container), sc_next_available_time[best_sc])
        container_start_time = sc_start_time + pd.Timedelta(seconds=time_to_container)
        finish_time = container_start_time + pd.Timedelta(seconds=container_travel_time)

        # Standardize timestamps
        sc_start_time = pd.to_datetime(sc_start_time)
        container_start_time = pd.to_datetime(container_start_time)
        finish_time = pd.to_datetime(finish_time)

        # Calculate idle time
        idle_time = max(0, (sc_start_time - sc_next_available_time[best_sc]).total_seconds())

        # Save the assignment details
        assignments.append({
            'task_id': order['ContainerOrderId'],
            'sc_id': best_sc,
            'time_first_known': time_first_known,
            'sc_start_time': sc_start_time,
            'container_start_time': container_start_time,
            'finish_time': finish_time,
            'distance_to_container': distance_to_container,
            'time_to_container': time_to_container,
            'travel_time': total_sc_distance / speed_mm_per_sec,
            'idle_time': idle_time,
            'penalty_time': 0,  # Placeholder for penalty
            'origin_location': order['OriginLocation'],
            'destination_location': order['DestinationLocation'],
            'sc_total_distance': total_sc_distance,
            'container_distance': container_distance,
            'sc_travel_time': total_sc_distance / speed_mm_per_sec,
            'container_travel_time': container_travel_time
        })

        # Update SC's next available time and location
        sc_next_available_time[best_sc] = finish_time
        sc_current_locations[best_sc] = destination

    # Convert assignments to DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Convert all timestamp columns to consistent format
    for col in ['time_first_known', 'sc_start_time', 'container_start_time', 'finish_time']:
        assignments_df[col] = pd.to_datetime(assignments_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    if assignments_df.empty:
        raise ValueError("No assignments were generated. Check input data or task assignment logic.")

    print(f"Number of Assignments Generated: {len(assignments_df)}")

    # Save assignments to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    assignments_df.to_csv(output_file, index=False)
    print(f"Assignments with KPIs saved to {output_file}")

    return assignments_df

from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import pandas as pd
from tqdm import tqdm
from scripts.utils import time_function, calculate_manhattan_distance


@time_function
def assign_jobs_mip_min_distance(container_orders, vehicles, location_dict, output_file="output/assignments_mip.csv", speed_mm_per_sec=5555):
    """
    Assign tasks to SCs using Mixed-Integer Programming to minimize total combined distance.

    Parameters:
        container_orders (pd.DataFrame): Container order details.
        vehicles (pd.DataFrame): Vehicle details.
        location_dict (dict): Preprocessed location data.
        output_file (str): Path to save the assignments CSV.
        speed_mm_per_sec (int): Speed of the SC in mm/s (default: 5555 mm/s).

    Returns:
        pd.DataFrame: Assignments with distance and time details.
    """
    # Data setup
    sc_ids = vehicles['ID'].tolist()
    task_ids = container_orders['ContainerOrderId'].tolist()

    # Map SC start locations and container origins/destinations
    sc_start_coords = {
        sc: (location_dict[vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0]]['X-Coordinate [mm]'],
             location_dict[vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0]]['Y-Coordinate [mm]'])
        for sc in sc_ids if vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0] in location_dict
    }

    origin_coords = {
        row['ContainerOrderId']: (location_dict[row['OriginLocation']]['X-Coordinate [mm]'], 
                                  location_dict[row['OriginLocation']]['Y-Coordinate [mm]'])
        for _, row in container_orders.iterrows() if row['OriginLocation'] in location_dict
    }

    dest_coords = {
        row['ContainerOrderId']: (location_dict[row['DestinationLocation']]['X-Coordinate [mm]'], 
                                  location_dict[row['DestinationLocation']]['Y-Coordinate [mm]'])
        for _, row in container_orders.iterrows() if row['DestinationLocation'] in location_dict
    }

    # Calculate distances between SCs and tasks
    distances = {}
    for sc_id in sc_ids:
        for task_id in task_ids:
            if sc_id in sc_start_coords and task_id in origin_coords and task_id in dest_coords:
                sc_to_origin = calculate_manhattan_distance(sc_start_coords[sc_id], origin_coords[task_id])
                origin_to_dest = calculate_manhattan_distance(origin_coords[task_id], dest_coords[task_id])
                total_distance = sc_to_origin + origin_to_dest
                distances[(sc_id, task_id)] = (sc_to_origin, origin_to_dest, total_distance)

    # Define the MIP problem
    prob = LpProblem("SC_Assignment_Min_Distance", LpMinimize)

    # Define decision variables
    x = LpVariable.dicts("assign", [(sc, task) for sc in sc_ids for task in task_ids], 0, 1, cat="Binary")

    # Objective function: Minimize total distance
    prob += lpSum([x[(sc, task)] * distances[(sc, task)][2] for sc, task in distances])

    # Constraints
    # 1. Each task is assigned to exactly one SC
    for task in task_ids:
        prob += lpSum([x[(sc, task)] for sc in sc_ids if (sc, task) in distances]) == 1

    # Solve the problem
    prob.solve()

    # Extract results
    assignments = []
    for sc, task in x:
        if x[(sc, task)].value() == 1:
            sc_to_origin, origin_to_dest, total_distance = distances[(sc, task)]
            time_to_container = sc_to_origin / speed_mm_per_sec
            container_travel_time = origin_to_dest / speed_mm_per_sec

            # Retrieve task details
            task_data = container_orders[container_orders['ContainerOrderId'] == task].iloc[0]
            time_first_known = pd.to_datetime(task_data['Time first known'])

            # Calculate times
            sc_start_time = time_first_known - pd.Timedelta(seconds=time_to_container)
            container_start_time = sc_start_time + pd.Timedelta(seconds=time_to_container)
            finish_time = container_start_time + pd.Timedelta(seconds=container_travel_time)

            # Save the assignment
            assignments.append({
                'task_id': task,
                'sc_id': sc,
                'time_first_known': time_first_known,
                'sc_start_time': sc_start_time,
                'container_start_time': container_start_time,
                'finish_time': finish_time,
                'distance_to_container': sc_to_origin,
                'time_to_container': time_to_container,
                'travel_time': total_distance / speed_mm_per_sec,
                'idle_time': 0,  # Placeholder, to be updated if necessary
                'origin_location': task_data['OriginLocation'],
                'destination_location': task_data['DestinationLocation'],
                'sc_total_distance': total_distance,
                'container_distance': origin_to_dest,
            })

    # Convert assignments to DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Format time columns
    for col in ['time_first_known', 'sc_start_time', 'container_start_time', 'finish_time']:
        assignments_df[col] = pd.to_datetime(assignments_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    assignments_df.to_csv(output_file, index=False)
    print(f"Assignments saved to {output_file}")

    return assignments_df

@time_function
def assign_jobs_mip_min_makespan(container_orders, vehicles, location_dict, output_file="output/assignments_mip_makespan.csv", speed_mm_per_sec=5555):
    """
    Assign tasks to SCs using Mixed-Integer Programming to minimize the makespan and optimize SC utilization.

    Parameters:
        container_orders (pd.DataFrame): Container order details.
        vehicles (pd.DataFrame): Vehicle details.
        location_dict (dict): Preprocessed location data.
        output_file (str): Path to save the assignments CSV.
        speed_mm_per_sec (int): Speed of the SC in mm/s (default: 5555 mm/s).

    Returns:
        pd.DataFrame: Assignments with time and distance details.
    """
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, LpBinary

    # Data setup
    sc_ids = vehicles['ID'].tolist()
    task_ids = container_orders['ContainerOrderId'].tolist()

    # Map SC start locations and container origins/destinations
    sc_start_coords = {
        sc: (location_dict[vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0]]['X-Coordinate [mm]'],
             location_dict[vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0]]['Y-Coordinate [mm]'])
        for sc in sc_ids if vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0] in location_dict
    }

    origin_coords = {
        row['ContainerOrderId']: (location_dict[row['OriginLocation']]['X-Coordinate [mm]'], 
                                  location_dict[row['OriginLocation']]['Y-Coordinate [mm]'])
        for _, row in container_orders.iterrows() if row['OriginLocation'] in location_dict
    }

    dest_coords = {
        row['ContainerOrderId']: (location_dict[row['DestinationLocation']]['X-Coordinate [mm]'], 
                                  location_dict[row['DestinationLocation']]['Y-Coordinate [mm]'])
        for _, row in container_orders.iterrows() if row['DestinationLocation'] in location_dict
    }

    time_first_known = {
        row['ContainerOrderId']: pd.to_datetime(row['Time first known'])
        for _, row in container_orders.iterrows()
    }

    # Define the MIP problem
    prob = LpProblem("SC_Assignment_Min_Makespan", LpMinimize)

    # Decision variables
    x = LpVariable.dicts("assign", [(sc, task) for sc in sc_ids for task in task_ids], 0, 1, cat=LpBinary)
    start_times = LpVariable.dicts("start", task_ids, lowBound=0, cat=LpContinuous)
    finish_times = LpVariable.dicts("finish", task_ids, lowBound=0, cat=LpContinuous)
    makespan = LpVariable("makespan", lowBound=0, cat=LpContinuous)

    # Distances and travel times
    distances = {}
    for sc_id in sc_ids:
        for task_id in task_ids:
            if sc_id in sc_start_coords and task_id in origin_coords and task_id in dest_coords:
                sc_to_origin = calculate_manhattan_distance(sc_start_coords[sc_id], origin_coords[task_id])
                origin_to_dest = calculate_manhattan_distance(origin_coords[task_id], dest_coords[task_id])
                total_distance = sc_to_origin + origin_to_dest
                distances[(sc_id, task_id)] = {
                    'sc_to_origin': sc_to_origin,
                    'origin_to_dest': origin_to_dest,
                    'total_distance': total_distance,
                    'time_to_origin': sc_to_origin / speed_mm_per_sec,
                    'travel_time': total_distance / speed_mm_per_sec
                }

    # Objective: Minimize the makespan
    prob += makespan

    # Constraints
    for task_id in task_ids:
        # Each task must be assigned to exactly one SC
        prob += lpSum([x[(sc, task_id)] for sc in sc_ids if (sc, task_id) in distances]) == 1

        # Task finish time must account for travel and service times
        for sc_id in sc_ids:
            if (sc_id, task_id) in distances:
                travel_time = distances[(sc_id, task_id)]['travel_time']
                prob += finish_times[task_id] >= start_times[task_id] + travel_time

        # Start time must not be earlier than the task's `time_first_known`
        prob += start_times[task_id] >= (time_first_known[task_id] - pd.Timestamp("1970-01-01")).total_seconds()

    for sc_id in sc_ids:
        # Non-overlapping tasks for the same SC
        for t1 in task_ids:
            for t2 in task_ids:
                if t1 != t2 and (sc_id, t1) in distances and (sc_id, t2) in distances:
                    prob += start_times[t2] >= finish_times[t1] - (1 - x[(sc_id, t1)]) * 1e6
                    prob += start_times[t1] >= finish_times[t2] - (1 - x[(sc_id, t2)]) * 1e6

    # Makespan constraint: Finish time of all tasks must be less than or equal to makespan
    for task_id in task_ids:
        prob += finish_times[task_id] <= makespan

    # Solve the problem
    prob.solve()

    # Extract results
    assignments = []
    for sc, task in x:
        if x[(sc, task)].value() == 1:
            sc_to_origin = distances[(sc, task)]['sc_to_origin']
            origin_to_dest = distances[(sc, task)]['origin_to_dest']
            total_distance = distances[(sc, task)]['total_distance']
            time_to_origin = distances[(sc, task)]['time_to_origin']
            travel_time = distances[(sc, task)]['travel_time']

            start_time = start_times[task].value()
            finish_time = finish_times[task].value()
            sc_start_time = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=start_time)
            container_start_time = sc_start_time + pd.Timedelta(seconds=time_to_origin)
            finish_time = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=finish_time)

            task_data = container_orders[container_orders['ContainerOrderId'] == task].iloc[0]

            assignments.append({
                'task_id': task,
                'sc_id': sc,
                'time_first_known': time_first_known[task],
                'sc_start_time': sc_start_time,
                'container_start_time': container_start_time,
                'finish_time': finish_time,
                'distance_to_container': sc_to_origin,
                'time_to_container': time_to_origin,
                'travel_time': travel_time,
                'idle_time': 0,  # Placeholder for idle time
                'penalty_time': 0,  # Placeholder for penalty
                'origin_location': task_data['OriginLocation'],
                'destination_location': task_data['DestinationLocation'],
                'sc_total_distance': total_distance,
                'container_distance': origin_to_dest,
                'sc_travel_time': travel_time,
                'container_travel_time': origin_to_dest / speed_mm_per_sec
            })

    # Convert to DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    assignments_df.to_csv(output_file, index=False)
    print(f"Assignments saved to {output_file}")

    return assignments_df


@time_function
def assign_jobs_genetic(container_orders, vehicles, location_dict, output_file="output/assignments_genetic.csv", 
                        population_size=50, generations=100, speed_mm_per_sec=5555):
    """
    Optimize SC assignments using Genetic Algorithm to minimize total combined distance.

    Parameters:
        container_orders (pd.DataFrame): Container order details.
        vehicles (pd.DataFrame): Vehicle details.
        location_dict (dict): Preprocessed location data.
        output_file (str): Path to save the assignments CSV.
        population_size (int): Population size for the Genetic Algorithm.
        generations (int): Number of generations for the Genetic Algorithm.
        speed_mm_per_sec (int): Speed of the SC in mm/s (default: 5555 mm/s).

    Returns:
        pd.DataFrame: Assignments with distance and time details.
    """
    # Problem setup
    sc_ids = vehicles['ID'].tolist()
    task_ids = container_orders['ContainerOrderId'].tolist()

    # Map SC start locations and container origins/destinations
    sc_start_coords = {
        sc: (location_dict[vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0]]['X-Coordinate [mm]'],
             location_dict[vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0]]['Y-Coordinate [mm]'])
        for sc in sc_ids if vehicles.loc[vehicles['ID'] == sc, 'StartLocation'].values[0] in location_dict
    }

    origin_coords = {
        row['ContainerOrderId']: (location_dict[row['OriginLocation']]['X-Coordinate [mm]'], 
                                  location_dict[row['OriginLocation']]['Y-Coordinate [mm]'])
        for _, row in container_orders.iterrows() if row['OriginLocation'] in location_dict
    }

    dest_coords = {
        row['ContainerOrderId']: (location_dict[row['DestinationLocation']]['X-Coordinate [mm]'], 
                                  location_dict[row['DestinationLocation']]['Y-Coordinate [mm]'])
        for _, row in container_orders.iterrows() if row['DestinationLocation'] in location_dict
    }

    # Define evaluation function
    def evaluate(individual):
        total_distance = 0
        for task, sc in zip(task_ids, individual):
            if sc in sc_start_coords and task in origin_coords and task in dest_coords:
                sc_to_origin = calculate_manhattan_distance(sc_start_coords[sc], origin_coords[task])
                origin_to_dest = calculate_manhattan_distance(origin_coords[task], dest_coords[task])
                total_distance += sc_to_origin + origin_to_dest
        return total_distance,

    # Setup Genetic Algorithm
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_sc", random.choice, sc_ids)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_sc, n=len(task_ids))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Run Genetic Algorithm
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    # Extract the best solution
    best_individual = tools.selBest(population, k=1)[0]
    assignments = []

    for task, sc in zip(task_ids, best_individual):
        if sc in sc_start_coords and task in origin_coords and task in dest_coords:
            sc_to_origin = calculate_manhattan_distance(sc_start_coords[sc], origin_coords[task])
            origin_to_dest = calculate_manhattan_distance(origin_coords[task], dest_coords[task])
            total_distance = sc_to_origin + origin_to_dest
            time_to_origin = sc_to_origin / speed_mm_per_sec
            container_distance = origin_to_dest
            travel_time = total_distance / speed_mm_per_sec

            # Time calculations
            task_data = container_orders[container_orders['ContainerOrderId'] == task].iloc[0]
            time_first_known = pd.to_datetime(task_data['Time first known'])
            sc_start_time = time_first_known - pd.Timedelta(seconds=time_to_origin)
            container_start_time = sc_start_time + pd.Timedelta(seconds=time_to_origin)
            finish_time = container_start_time + pd.Timedelta(seconds=travel_time - time_to_origin)

            # Save assignment details
            assignments.append({
                'task_id': task,
                'sc_id': sc,
                'time_first_known': time_first_known,
                'sc_start_time': sc_start_time,
                'container_start_time': container_start_time,
                'finish_time': finish_time,
                'distance_to_container': sc_to_origin,
                'time_to_container': time_to_origin,
                'travel_time': travel_time,  # Total travel time
                'idle_time': 0,  # Placeholder
                'penalty_time': 0,  # Placeholder
                'origin_location': task_data['OriginLocation'],
                'destination_location': task_data['DestinationLocation'],
                'sc_total_distance': total_distance,
                'container_distance': container_distance,
                'sc_travel_time': travel_time,
                'container_travel_time': container_distance / speed_mm_per_sec
            })

    # Convert assignments to DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Format time columns
    for col in ['time_first_known', 'sc_start_time', 'container_start_time', 'finish_time']:
        assignments_df[col] = pd.to_datetime(assignments_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    assignments_df.to_csv(output_file, index=False)
    print(f"Assignments saved to {output_file}")

    return assignments_df


@time_function
def assign_jobs_randomly(container_orders, vehicles, location_dict, output_file="output/assignments_random.csv", speed_mm_per_sec=5555):
    """
    Assign tasks randomly to available SCs, calculate KPIs, and save the assignments.

    Parameters:
        container_orders (pd.DataFrame): Container order details.
        vehicles (pd.DataFrame): Vehicle details.
        location_dict (dict): Preprocessed location data.
        output_file (str): Path to save the assignments CSV.
        speed_mm_per_sec (int): Speed of the SC in mm/s (default: 5555 mm/s).

    Returns:
        pd.DataFrame: Assignments with time details and KPIs.
    """
    assignments = []

    # Map SCs to their start locations
    sc_start_locations = {
        vehicle['ID']: location_dict.get(vehicle['StartLocation'])
        for _, vehicle in vehicles.iterrows()
    }
    sc_next_available_time = {vehicle['ID']: pd.to_datetime(vehicle['LogOn']) for _, vehicle in vehicles.iterrows()}

    for _, order in tqdm(container_orders.iterrows(), total=len(container_orders), desc="Random Assignment"):
        origin = location_dict.get(order['OriginLocation'])
        destination = location_dict.get(order['DestinationLocation'])
        time_first_known = pd.to_datetime(order['Time first known'])

        if origin is None or destination is None:
            print(f"Skipping task {order['ContainerOrderId']}: Missing origin/destination")
            continue  # Skip if origin or destination is missing

        origin_coords = (origin['X-Coordinate [mm]'], origin['Y-Coordinate [mm]'])
        dest_coords = (destination['X-Coordinate [mm]'], destination['Y-Coordinate [mm]'])

        # Randomly select an SC
        sc_id = random.choice(list(sc_start_locations.keys()))
        sc_location = sc_start_locations.get(sc_id)

        if sc_location is None:
            print(f"Skipping SC {sc_id}: Missing start location")
            continue  # Skip if SC's start location is missing

        sc_coords = (sc_location['X-Coordinate [mm]'], sc_location['Y-Coordinate [mm]'])

        # Calculate distances
        distance_to_container = calculate_manhattan_distance(sc_coords, origin_coords)
        travel_to_dest = calculate_manhattan_distance(origin_coords, dest_coords)
        total_sc_distance = distance_to_container + travel_to_dest
        container_distance = travel_to_dest

        # Calculate times
        time_to_container = distance_to_container / speed_mm_per_sec
        container_travel_time = container_distance / speed_mm_per_sec

        # Determine the start times
        sc_start_time = max(time_first_known - pd.Timedelta(seconds=time_to_container), sc_next_available_time[sc_id])
        container_start_time = sc_start_time + pd.Timedelta(seconds=time_to_container)
        finish_time = container_start_time + pd.Timedelta(seconds=container_travel_time)

        # Standardize timestamps
        sc_start_time = pd.to_datetime(sc_start_time)
        container_start_time = pd.to_datetime(container_start_time)
        finish_time = pd.to_datetime(finish_time)

        # Calculate idle time
        idle_time = max(0, (sc_start_time - sc_next_available_time[sc_id]).total_seconds())

        # Save the assignment details
        assignments.append({
            'task_id': order['ContainerOrderId'],
            'sc_id': sc_id,  # Add the SC ID explicitly
            'time_first_known': time_first_known,
            'sc_start_time': sc_start_time,
            'container_start_time': container_start_time,
            'finish_time': finish_time,
            'distance_to_container': distance_to_container,
            'time_to_container': time_to_container,
            'travel_time': total_sc_distance / speed_mm_per_sec,  # Total travel time
            'idle_time': idle_time,
            'penalty_time': 0,  # Placeholder for penalty
            'origin_location': order['OriginLocation'],
            'destination_location': order['DestinationLocation'],
            'sc_total_distance': total_sc_distance,
            'container_distance': container_distance,
            'sc_travel_time': total_sc_distance / speed_mm_per_sec,
            'container_travel_time': container_travel_time
        })

        # Update SC's next available time and location
        sc_next_available_time[sc_id] = finish_time
        sc_start_locations[sc_id] = destination

    # Convert assignments to DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Convert all timestamp columns to consistent format
    for col in ['time_first_known', 'sc_start_time', 'container_start_time', 'finish_time']:
        assignments_df[col] = pd.to_datetime(assignments_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    if assignments_df.empty:
        raise ValueError("No assignments were generated. Check input data or task assignment logic.")

    print(f"Number of Assignments Generated: {len(assignments_df)}")

    # Save assignments to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    assignments_df.to_csv(output_file, index=False)
    print(f"Assignments with KPIs saved to {output_file}")

    return assignments_df

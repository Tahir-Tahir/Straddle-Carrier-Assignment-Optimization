import time

def time_function(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"\nRunning {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"{func.__name__} completed in {time.time() - start_time:.2f} seconds.\n")
        return result
    return wrapper

def calculate_manhattan_distance(coord1, coord2):
    """Calculate Manhattan Distance between two points."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

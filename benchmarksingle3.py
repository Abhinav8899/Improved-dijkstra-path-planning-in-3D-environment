import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Optional, Any
import importlib.util
import sys
import inspect

def calculate_path_length(path: np.ndarray) -> float:
    """Calculate the total length of a path."""
    if path is None or len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))

def calculate_path_clearance(path: np.ndarray, obstacles: List[Any]) -> float:
    """Calculate the minimum clearance from obstacles."""
    def distance_to_obstacle(point: np.ndarray, obstacle: np.ndarray) -> float:
        """Calculate the distance from a point to a building obstacle."""
        x, y, z = point
        ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obstacle
        dx = max(ox_min - x, 0, x - ox_max)
        dy = max(oy_min - y, 0, y - oy_max)
        dz = max(oz_min - z, 0, z - oz_max)
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    if path is None:
        return 0.0
        
    clearances = []
    for point in path:
        min_clearance = float('inf')
        for obs in obstacles:
            clearance = distance_to_obstacle(point, obs)
            min_clearance = min(min_clearance, clearance)
        clearances.append(min_clearance)
    return np.mean(clearances)

def calculate_path_smoothness(path: np.ndarray) -> float:
    """Calculate the smoothness of the path as a percentage."""
    if path is None or len(path) < 3:
        return 0.0
    
    # Calculate angles between consecutive segments
    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        # Normalize vectors
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        # Calculate angle in radians
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)
    
    # Convert angles to smoothness percentage
    # 0 degrees (parallel segments) = 100% smooth
    # 180 degrees (complete reversal) = 0% smooth
    angles = np.array(angles)
    smoothness = 100 * (1 - np.mean(angles) / np.pi)
    
    return smoothness

def calculate_cpu_usage(process: psutil.Process, start_time: float, end_time: float) -> float:
    """Calculate the average CPU usage percentage during execution."""
    try:
        # Get CPU times for the process
        cpu_times = process.cpu_times()
        total_cpu_time = cpu_times.user + cpu_times.system
        
        # Calculate actual CPU percentage based on total time and number of cores
        elapsed_time = end_time - start_time
        cpu_count = psutil.cpu_count()
        
        if elapsed_time > 0:
            cpu_percent = (total_cpu_time / elapsed_time / cpu_count) * 100
            return min(cpu_percent, 100.0)  # Cap at 100%
        return 0.0
    except Exception:
        return 0.0

def calculate_unsafe_points(path: np.ndarray, obstacles: List[Any], safety_threshold: float = 0.5) -> int:
    """Calculate number of points that are too close to obstacles."""
    if path is None:
        return 0
        
    unsafe_count = 0
    for point in path:
        min_clearance = float('inf')
        for obs in obstacles:
            x, y, z = point
            ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
            dx = max(ox_min - x, 0, x - ox_max)
            dy = max(oy_min - y, 0, y - oy_max)
            dz = max(oz_min - z, 0, z - oz_max)
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            min_clearance = min(min_clearance, dist)
        
        if min_clearance < safety_threshold:
            unsafe_count += 1
            
    return unsafe_count

def calculate_path_efficiency(path: np.ndarray) -> float:
    """Calculate path efficiency as percentage (direct distance / path length)"""
    if path is None or len(path) < 2:
        return 0.0
    
    # Calculate direct distance from start to goal
    direct_distance = np.linalg.norm(path[-1] - path[0])
    # Calculate actual path length
    path_length = calculate_path_length(path)

    if path_length == 0:
        return 0.0
        
    # Calculate efficiency as percentage
    efficiency = (direct_distance / path_length) * 100
    
    return min(efficiency, 100.0)  # Cap at 100%

def visualize_path(bounds: np.ndarray, obstacles: List[Any], path: np.ndarray, 
                  start: np.ndarray, goal: np.ndarray) -> None:
    """Visualize the path in a 3D environment."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set bounds
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    
    # Plot obstacles (buildings)
    for obs in obstacles:
        ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
        vertices = [
            # Bottom face
            [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
             (ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min)],
            # Top face
            [(ox_min, oy_min, oz_max), (ox_max, oy_min, oz_max),
             (ox_max, oy_max, oz_max), (ox_min, oy_max, oz_max)],
            # Side faces
            [(ox_min, oy_min, oz_min), (ox_max, oy_min, oz_min),
             (ox_max, oy_min, oz_max), (ox_min, oy_min, oz_max)],
            [(ox_max, oy_min, oz_min), (ox_max, oy_max, oz_min),
             (ox_max, oy_max, oz_max), (ox_max, oy_min, oz_max)],
            [(ox_max, oy_max, oz_min), (ox_min, oy_max, oz_min),
             (ox_min, oy_max, oz_max), (ox_max, oy_max, oz_max)],
            [(ox_min, oy_max, oz_min), (ox_min, oy_min, oz_min),
             (ox_min, oy_min, oz_max), (ox_min, oy_max, oz_max)]
        ]
        # Make buildings more visible
        ax.add_collection3d(Poly3DCollection(vertices, 
                                           facecolors='gray', 
                                           alpha=0.6,  # Increased opacity
                                           linewidth=1, 
                                           edgecolor='black'))
    
    # Plot path with increased visibility
    if path is not None and len(path) > 0:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', 
                linewidth=3, label='Path')  # Increased linewidth
    
    # Plot start and goal points with increased size
    ax.scatter(start[0], start[1], start[2], 
              c='green', s=200, label='Start')  # Increased size
    ax.scatter(goal[0], goal[1], goal[2], 
              c='red', s=200, label='Goal')     # Increased size
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('UAV Path Planning in Urban Environment')
    ax.legend()
    
    # Add grid for better depth perception
    ax.grid(True)
    
    plt.show()

def load_algorithm(file_path: str) -> Any:
    """Dynamically load the algorithm module and find the planner class."""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("algorithm", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["algorithm"] = module
        spec.loader.exec_module(module)
        
        # Find the first class that has plan_path method
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, 'plan_path'):
                return obj
        
        raise AttributeError("No suitable path planning class found in the module")
    except Exception as e:
        print(f"Error loading algorithm: {str(e)}")
        return None

if __name__ == "__main__":
    # Get algorithm file path
    algorithm_file = input("Enter the path to your algorithm file: ").strip()
    if not algorithm_file:
        print("No file path provided")
        exit(1)

    # Load the algorithm
    PlannerClass = load_algorithm(algorithm_file)
    if PlannerClass is None:
        print("Failed to load algorithm")
        exit(1)

    # Environment setup
    bounds = np.array([[-15, 15], [-15, 15], [0, 15]])
    obstacles = np.array([
        # Reduced number of buildings for a less dense environment
        [-8, -8, 0, -6, -6, 9],      # Building 1 (9 units)
        [4, 4, 0, 6, 6, 10],         # Building 2 (10 units)
        [-3, 3, 0, -1, 5, 8],        # Building 3 (8 units)
        [0, -5, 0, 2, -3, 9.5],      # Building 4 (9.5 units)
        [-4, -2, 0, -2, 0, 8.5],     # Building 5 (8.5 units)
        [8, 8, 0, 10, 10, 9.8],      # Building 6 (9.8 units)
        [-10, 5, 0, -8, 7, 8.7]      # Building 7 (8.7 units)
    ])
    
    # Start and goal positions
    start = np.array([-12., -12., 1.])
    goal = np.array([12., 12., 5.])

    # Record initial state
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    try:
        # Create planner instance using the loaded algorithm class
        planner = PlannerClass(
            start=start,
            goal=goal,
            bounds=bounds,
            obstacles=obstacles
        )
        
        # Run path planning
        path = planner.plan_path()
        success = path is not None and len(path) > 0
        
        # End timing
        end_time = time.time()
        computational_time = end_time - start_time
        
        # Calculate memory usage
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_usage = final_memory - initial_memory
        
        # Calculate metrics
        print("\nPath Planning Metrics:")
        if success:
            path_length = calculate_path_length(path)
            path_clearance = calculate_path_clearance(path, obstacles)
            path_smoothness = calculate_path_smoothness(path)
            path_efficiency = calculate_path_efficiency(path)
            cpu_usage = calculate_cpu_usage(process, start_time, end_time)
            print(f"Path Length (m): {path_length:.2f}")
            print(f"Path Clearance (m): {path_clearance:.2f}")
            print(f"Path Smoothness (%): {path_smoothness:.2f}")
            print(f"Path Efficiency (%): {path_efficiency:.2f}")
            print(f"CPU Usage (%): {cpu_usage:.2f}")
            print(f"Success Rate: Yes")
        else:
            print("Path Length (m): N/A")
            print("Path Clearance (m): N/A")
            print("Path Smoothness (%): N/A")
            print("Path Efficiency (%): N/A")
            print("CPU Usage (%): N/A")
            print("Success Rate: No")
            
        print(f"Computational Time (s): {computational_time:.3f}")
        print(f"Memory Usage (MB): {memory_usage:.2f}")
        
        if hasattr(planner, 'nodes'):
            print(f"Iterations: {len(planner.nodes)}")
        
        # Visualize results
        if success:
            visualize_path(bounds, obstacles, path, start, goal)
            
    except Exception as e:
        print(f"Error during path planning: {str(e)}")
        print("\nMetrics:")
        print("Path Length (m): N/A")
        print("Path Clearance (m): N/A")
        print("Path Smoothness (%): N/A")
        print("Path Efficiency (%): N/A")
        print("CPU Usage (%): N/A")
        print("Success Rate: No")


        #Must have a plan_path() method that returns the path
#Must accept start, goal, bounds, and obstacles in its constructor in algorithm
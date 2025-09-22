import numpy as np
from typing import List, Tuple, Optional, Union
import heapq

class DijkstraPlanner:
    def __init__(self, 
                 start: np.ndarray,
                 goal: np.ndarray,
                 bounds: np.ndarray,
                 obstacles: List[Union[np.ndarray, Tuple, dict]]):
        
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.resolution = 0.5  # Reduced resolution for better path finding
        self.nodes = []
        
    def check_collision(self, point: np.ndarray) -> bool:
        """Check if point collides with obstacles"""
        margin = 0.2  # Reduced margin for better path finding
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):  # Box obstacle
                x, y, z = point
                ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
                if (ox_min - margin <= x <= ox_max + margin and
                    oy_min - margin <= y <= oy_max + margin and
                    oz_min - margin <= z <= oz_max + margin):
                    return True
            else:  # Spherical obstacle
                if isinstance(obs, dict):
                    center = obs['position']
                    radius = obs['radius']
                else:
                    center, radius = obs
                if np.linalg.norm(point - np.array(center)) <= radius + margin:
                    return True
        return False

    def get_neighbors(self, point: np.ndarray) -> List[np.ndarray]:
        """Get valid neighboring points"""
        neighbors = []
        # Basic movements plus diagonals
        directions = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
        ]
        
        for direction in directions:
            neighbor = point + np.array(direction) * self.resolution
            
            # Check bounds with small margin
            margin = 0.1
            if not all(self.bounds[i][0] + margin <= neighbor[i] <= self.bounds[i][1] - margin for i in range(3)):
                continue
                
            # Check collision
            if self.check_collision(neighbor):
                continue
                
            neighbors.append(neighbor)
            
        return neighbors

    def heuristic(self, point: np.ndarray) -> float:
        """Calculate heuristic distance to goal"""
        return np.linalg.norm(point - self.goal)

    def plan_path(self) -> Optional[np.ndarray]:
        """Plan path using Dijkstra's algorithm"""
        start_tuple = tuple(self.start)
        open_set = [(0, start_tuple)]  # Priority queue (cost, point)
        g_costs = {start_tuple: 0}  # Actual cost from start
        f_costs = {start_tuple: self.heuristic(self.start)}  # Total estimated cost
        parents = {}  # Dictionary to store parents
        closed_set = set()  # Set of visited points
        
        while open_set and len(self.nodes) < 20000:
            # Get node with minimum cost
            current_cost, current = heapq.heappop(open_set)
            current_array = np.array(current)
            
            # Add to visited nodes for visualization
            if current not in closed_set:
                self.nodes.append(current_array)
            
            # Check if goal is reached
            if np.linalg.norm(current_array - self.goal) < self.resolution:
                path = [self.goal]
                while current in parents:
                    path.append(np.array(current))
                    current = parents[current]
                path.append(self.start)
                return np.array(path[::-1])
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Process neighbors
            for neighbor in self.get_neighbors(current_array):
                neighbor_tuple = tuple(neighbor)
                
                if neighbor_tuple in closed_set:
                    continue
                
                # Calculate new cost
                tentative_g_cost = g_costs[current] + np.linalg.norm(neighbor - current_array)
                
                if neighbor_tuple not in g_costs or tentative_g_cost < g_costs[neighbor_tuple]:
                    # Update path and costs
                    parents[neighbor_tuple] = current
                    g_costs[neighbor_tuple] = tentative_g_cost
                    f_costs[neighbor_tuple] = tentative_g_cost + self.heuristic(neighbor)
                    heapq.heappush(open_set, (f_costs[neighbor_tuple], neighbor_tuple))
        
        # If goal not reached but have nodes, return best partial path
        if self.nodes:
            closest_node = min(closed_set, key=lambda x: np.linalg.norm(np.array(x) - self.goal))
            path = [self.goal]
            current = closest_node
            while current in parents:
                path.append(np.array(current))
                current = parents[current]
            path.append(self.start)
            return np.array(path[::-1])
            
        return None
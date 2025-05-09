import numpy as np
import math

# Pre-compute all possible moves and their costs
DIRECTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
                      (-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonal
DIAGONAL_COST = 1.414  # sqrt(2)
STRAIGHT_COST = 1.0

# Pre-compute direction indices for faster access
CARDINAL_INDICES = [0, 1, 2, 3]
DIAGONAL_INDICES = [4, 5, 6, 7]

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(pos, world):
    # Ultra-optimized neighbor calculation
    x, y = pos
    rows, cols = world.shape
    neighbors = []
    
    # Check cardinal directions first (faster)
    if x > 0 and world[x-1, y] == 0:
        neighbors.append((x-1, y))
    if x < rows-1 and world[x+1, y] == 0:
        neighbors.append((x+1, y))
    if y > 0 and world[x, y-1] == 0:
        neighbors.append((x, y-1))
    if y < cols-1 and world[x, y+1] == 0:
        neighbors.append((x, y+1))
    
    # Check diagonals only if needed
    if len(neighbors) < 4:  # If we have less than 4 cardinal moves
        if x > 0 and y > 0 and world[x-1, y-1] == 0:
            neighbors.append((x-1, y-1))
        if x > 0 and y < cols-1 and world[x-1, y+1] == 0:
            neighbors.append((x-1, y+1))
        if x < rows-1 and y > 0 and world[x+1, y-1] == 0:
            neighbors.append((x+1, y-1))
        if x < rows-1 and y < cols-1 and world[x+1, y+1] == 0:
            neighbors.append((x+1, y+1))
    
    return neighbors

def calculate_heuristic(current, pursued, pursuer):
    """
    Ultra-optimized heuristic with minimal calculations
    """
    # Pre-computed constants
    SAFE_DISTANCE = 4
    DANGER_DISTANCE = 2
    
    # Calculate distances once
    dist_to_target = manhattan_distance(current, pursued)
    dist_from_pursuer = manhattan_distance(current, pursuer)
    
    # Simplified safety score
    safety_penalty = 10 if dist_from_pursuer < DANGER_DISTANCE else max(0, SAFE_DISTANCE - dist_from_pursuer) * 2
    
    # Quick strategic check
    strategic_score = 2 if (dist_from_pursuer > SAFE_DISTANCE and 
                          dist_to_target + dist_from_pursuer < manhattan_distance(pursued, pursuer)) else 0
    
    return (dist_to_target * 0.7 - 
            dist_from_pursuer * 0.8 + 
            safety_penalty * 1.2 + 
            strategic_score)

def a_star_search(world, start, pursued, pursuer, max_steps=20):
    """
    Ultra-optimized A* search with minimal overhead
    """
    start_node = (start, 0, calculate_heuristic(start, pursued, pursuer), None)
    open_list = [start_node]
    closed_set = {start}
    node_dict = {start: start_node}
    steps = 0
    
    while open_list and steps < max_steps:
        steps += 1
        # Find node with minimum f_cost
        current_idx = 0
        min_f = open_list[0][1] + open_list[0][2]
        for i in range(1, len(open_list)):
            f = open_list[i][1] + open_list[i][2]
            if f < min_f:
                min_f = f
                current_idx = i
        current = open_list.pop(current_idx)
        current_pos = current[0]
        
        if current_pos == pursued:
            # Reconstruct path
            path = []
            while current:
                path.append(current[0])
                current = current[3]
            return path[::-1]
        
        for neighbor_pos in get_neighbors(current_pos, world):
            if neighbor_pos in closed_set:
                continue
                
            # Calculate move cost
            is_diagonal = abs(neighbor_pos[0] - current_pos[0]) + abs(neighbor_pos[1] - current_pos[1]) == 2
            move_cost = DIAGONAL_COST if is_diagonal else STRAIGHT_COST
            
            g_cost = current[1] + move_cost
            h_cost = calculate_heuristic(neighbor_pos, pursued, pursuer)
            f_cost = g_cost + h_cost
            
            if neighbor_pos not in node_dict or node_dict[neighbor_pos][1] + node_dict[neighbor_pos][2] > f_cost:
                neighbor_node = (neighbor_pos, g_cost, h_cost, current)
                node_dict[neighbor_pos] = neighbor_node
                open_list.append(neighbor_node)
                closed_set.add(neighbor_pos)
    
    return None

class PlannerAgent:
    # Class variables for cycle detection
    last_positions = []
    cycle_count = 0
    max_cycle_count = 3
    
    def __init__(self):
        pass
    
    @staticmethod
    def plan_action(world, current, pursued, pursuer):
        """
        Ultra-optimized action planning
        """
        # Convert numpy arrays to tuples for the search
        current_pos = (int(current[0]), int(current[1]))
        pursued_pos = (int(pursued[0]), int(pursued[1]))
        pursuer_pos = (int(pursuer[0]), int(pursuer[1]))
        
        # Get the optimal path with limited steps
        path = a_star_search(world, current_pos, pursued_pos, pursuer_pos, max_steps=20)
        
        if path and len(path) > 1:
            next_pos = path[1]
            action = np.array([next_pos[0] - current_pos[0], 
                             next_pos[1] - current_pos[1]])
            
            # Simplified cycle detection
            if len(PlannerAgent.last_positions) >= 4:
                if (PlannerAgent.last_positions[-1] == PlannerAgent.last_positions[-3] and 
                    PlannerAgent.last_positions[-2] == PlannerAgent.last_positions[-4]):
                    PlannerAgent.cycle_count += 1
                    if PlannerAgent.cycle_count >= PlannerAgent.max_cycle_count:
                        # Quick random move
                        valid_moves = []
                        for direction in DIRECTIONS:
                            new_pos = current + direction
                            if (0 <= new_pos[0] < world.shape[0] and 
                                0 <= new_pos[1] < world.shape[1] and 
                                world[new_pos[0], new_pos[1]] == 0):
                                valid_moves.append(direction)
                        if valid_moves:
                            PlannerAgent.cycle_count = 0
                            return valid_moves[np.random.choice(len(valid_moves))]
                else:
                    PlannerAgent.cycle_count = 0
            
            # Update position history
            PlannerAgent.last_positions.append(current_pos)
            if len(PlannerAgent.last_positions) > 5:
                PlannerAgent.last_positions.pop(0)
            
            return action
        
        # Quick tactical move
        valid_moves = []
        scores = []
        for direction in DIRECTIONS:
            new_pos = current + direction
            if (0 <= new_pos[0] < world.shape[0] and 
                0 <= new_pos[1] < world.shape[1] and 
                world[new_pos[0], new_pos[1]] == 0):
                valid_moves.append(direction)
                # Simplified scoring
                score = -manhattan_distance(tuple(new_pos), pursuer_pos)
                scores.append(score)
        
        if valid_moves:
            # Choose the move with the best score
            best_idx = np.argmax(scores)
            return valid_moves[best_idx]
        
        # If no safe moves, stay still
        return np.array([0, 0])
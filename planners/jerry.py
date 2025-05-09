import numpy as np
from typing import List, Tuple, Optional
import math
import random
from collections import deque

# Initial guess for probabilistic transitions
# These will be refined through learning
P_LEFT = 0.2
P_STRAIGHT = 0.6
P_RIGHT = 0.2

# Direction vectors - optimized for speed
DIRECTIONS = np.array([
    (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions first for priority
    (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal directions
])

# Ultra-performance settings
MAX_PATH_DEPTH = 15  # Maximum search depth for pathfinding

# Strategic weights
CAPTURE_REWARD = 5000  # Extreme reward for capture
DISTANCE_WEIGHT = 200  # Weight for distance calculations
PURSUIT_WEIGHT = 10.0  # Extreme pursuit bias

def manhattan_distance(p1, p2):
    """Calculate Manhattan distance between two points"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rotate_action(action, direction):
    """Rotate an action by 90 degrees"""
    if direction == 0:
        return action
    elif direction == 1:
        return np.array([-action[1], action[0]])
    else:
        return np.array([action[1], -action[0]])

def apply_probabilistic_transition(action, p_left=P_LEFT, p_straight=P_STRAIGHT):
    """Apply probabilistic transition to an action using learned probabilities"""
    p_right = 1.0 - p_left - p_straight
    rand = np.random.random()
    if rand < p_left:
        return rotate_action(action, -1)
    elif rand < p_left + p_straight:
        return action
    else:
        return rotate_action(action, 1)

class PlannerAgent:
    def __init__(self):
        """Initialize the ultra-aggressive Jerry agent with learning capabilities"""
        self.path_cache = {}  # Cache successful paths
        self.last_positions = []  # Track for cycle detection
        
        # Tracking opponent movements for learning
        self.pursuer_history = []  # Track pursuer's positions
        self.pursuer_actions = []  # Track pursuer's actions
        self.target_history = []   # Track target's positions
        self.target_actions = []   # Track target's actions
        
        # Learned transition probabilities for opponents
        self.pursuer_p_left = 0.2
        self.pursuer_p_straight = 0.6
        self.target_p_left = 0.2
        self.target_p_straight = 0.6
        
        # History of observed intended vs actual movements
        self.observed_transitions = []
        self.learning_rate = 0.05  # Rate to update probability estimates
        
        # Initial position to detect first move
        self.initial_position = None
        
    def update_transition_probabilities(self, intended_direction, actual_direction):
        """Update transition probability estimates based on observed movements"""
        self.observed_transitions.append((intended_direction, actual_direction))
        
        # Only update after collecting enough observations
        if len(self.observed_transitions) < 10:
            return
            
        # Count occurrences of each transition type
        left_count = 0
        straight_count = 0
        right_count = 0
        total = 0
        
        for intended, actual in self.observed_transitions[-30:]:  # Use last 30 observations
            if not np.array_equal(intended, actual):
                # Calculate the rotation that happened
                if np.array_equal(actual, rotate_action(intended, -1)):
                    left_count += 1
                elif np.array_equal(actual, rotate_action(intended, 1)):
                    right_count += 1
                else:
                    straight_count += 1  # Should not happen, but handle it
                total += 1
                
        # Update probabilities if we have observations
        if total > 0:
            p_left_new = left_count / total
            p_straight_new = straight_count / total
            
            # Use sliding window to update probabilities
            global P_LEFT, P_STRAIGHT
            P_LEFT = P_LEFT * (1 - self.learning_rate) + p_left_new * self.learning_rate
            P_STRAIGHT = P_STRAIGHT * (1 - self.learning_rate) + p_straight_new * self.learning_rate
            
            # Ensure probabilities remain valid
            if P_LEFT + P_STRAIGHT > 0.99:
                P_STRAIGHT = 0.99 - P_LEFT
                
    def track_opponent_movement(self, current_pursuer, last_pursuer, current_target, last_target):
        """Track opponent movements to learn their patterns"""
        # Skip if we don't have history yet
        if last_pursuer is None or last_target is None:
            return
            
        # Record pursuer's action
        if not np.array_equal(current_pursuer, last_pursuer):
            action = current_pursuer - last_pursuer
            self.pursuer_actions.append(action)
            
            # Limit history length
            if len(self.pursuer_actions) > 50:
                self.pursuer_actions.pop(0)
                
        # Record target's action
        if not np.array_equal(current_target, last_target):
            action = current_target - last_target
            self.target_actions.append(action)
            
            # Limit history length
            if len(self.target_actions) > 50:
                self.target_actions.pop(0)
                
    def predict_next_position(self, current, history, actions, p_left, p_straight):
        """Predict the next position of a player based on movement patterns"""
        if not actions:
            return current
            
        # Find the most common action in recent history
        if len(actions) >= 3:
            recent_actions = actions[-3:]
            # Convert actions to tuples for counting
            action_tuples = [tuple(a) for a in recent_actions]
            # Find most frequent action
            most_common = max(set(action_tuples), key=action_tuples.count)
            intended_action = np.array(most_common)
        else:
            # Not enough history, use last action
            intended_action = actions[-1]
            
        # Apply probabilistic transitions
        p_right = 1.0 - p_left - p_straight
        next_positions = []
        probabilities = []
        
        # Action could be straight, left, or right rotation
        straight_action = intended_action
        left_action = rotate_action(intended_action, -1)
        right_action = rotate_action(intended_action, 1)
        
        next_positions.append(current + straight_action)
        probabilities.append(p_straight)
        
        next_positions.append(current + left_action)
        probabilities.append(p_left)
        
        next_positions.append(current + right_action)
        probabilities.append(p_right)
        
        return next_positions, probabilities
        
    def direct_pursuit(self, world, current, pursued, pursuer):
        """Find direct path to target using optimized BFS with predictive avoidance"""
        # Check if we can capture directly
        if manhattan_distance(current, pursued) == 1:
            return pursued - current
            
        # Get predicted positions for pursuer and target
        if len(self.pursuer_actions) > 0:
            pursuer_next_positions, pursuer_probs = self.predict_next_position(
                pursuer, self.pursuer_history, self.pursuer_actions, 
                self.pursuer_p_left, self.pursuer_p_straight
            )
        else:
            pursuer_next_positions = [pursuer]
            pursuer_probs = [1.0]
            
        # Check for cached path
        key = (tuple(current), tuple(pursued))
        if key in self.path_cache:
            cached_path = self.path_cache[key]
            # Quick validation of first step
            if cached_path and len(cached_path) > 1:
                next_pos = cached_path[1]
                if (0 <= next_pos[0] < world.shape[0] and 
                    0 <= next_pos[1] < world.shape[1] and 
                    world[next_pos[0], next_pos[1]] == 0):
                    
                    # Check if next position is safe from predicted pursuer positions
                    action = np.array([next_pos[0] - current[0], next_pos[1] - current[1]])
                    is_safe = True
                    for p_pos, p_prob in zip(pursuer_next_positions, pursuer_probs):
                        if p_prob > 0.3 and manhattan_distance(next_pos, p_pos) <= 1:
                            is_safe = False
                            break
                            
                    if is_safe:
                        return action
        
        # Initialize BFS
        queue = [(current, [current])]
        visited = {tuple(current)}
        closest_dist = manhattan_distance(current, pursued)
        closest_path = None
        
        while queue and len(queue) < 1000:  # Limit search size for performance
            pos, path = queue.pop(0)
            
            # Check if reached target
            if np.array_equal(pos, pursued):
                if len(path) > 1:
                    self.path_cache[key] = path
                    return np.array([path[1][0] - current[0], path[1][1] - current[1]])
            
            # Check path length to prevent excessive searching
            if len(path) > MAX_PATH_DEPTH:
                continue
                
            # Track closest path for fallback
            dist = manhattan_distance(pos, pursued)
            if dist < closest_dist:
                closest_dist = dist
                closest_path = path
            
            # Try cardinal directions first (more efficient paths)
            for idx, direction in enumerate(DIRECTIONS):
                next_pos = (pos[0] + direction[0], pos[1] + direction[1])
                
                if (0 <= next_pos[0] < world.shape[0] and 
                    0 <= next_pos[1] < world.shape[1] and 
                    world[next_pos[0], next_pos[1]] == 0 and
                    next_pos not in visited):
                    
                    # Skip if this position is likely to be occupied by pursuer in next move
                    if len(path) == 1:  # Only check for first step
                        is_risky = False
                        for p_pos, p_prob in zip(pursuer_next_positions, pursuer_probs):
                            if p_prob > 0.3 and manhattan_distance(next_pos, p_pos) <= 1:
                                if not np.array_equal(next_pos, pursued):  # Allow risk if it's a capture
                                    is_risky = True
                                    break
                        if is_risky:
                            continue
                    
                    next_path = path + [next_pos]
                    visited.add(next_pos)
                    
                    # Immediate capture has highest priority
                    if np.array_equal(next_pos, pursued):
                        self.path_cache[key] = next_path
                        return np.array([next_path[1][0] - current[0], next_path[1][1] - current[1]])
                    
                    # Prioritize cardinal directions in queue
                    if idx < 4:
                        queue.insert(0, (next_pos, next_path))
                    else:
                        queue.append((next_pos, next_path))
        
        # If no path found, use closest approach
        if closest_path and len(closest_path) > 1:
            self.path_cache[key] = closest_path
            return np.array([closest_path[1][0] - current[0], closest_path[1][1] - current[1]])
            
        # Last resort: move in direction of target
        dx = pursued[0] - current[0]
        dy = pursued[1] - current[1]
        
        # Try to move in x direction first
        if dx != 0:
            next_pos = (current[0] + np.sign(dx), current[1])
            if (0 <= next_pos[0] < world.shape[0] and 
                0 <= next_pos[1] < world.shape[1] and 
                world[next_pos[0], next_pos[1]] == 0):
                return np.array([np.sign(dx), 0])
                
        # If not possible, try y direction
        if dy != 0:
            next_pos = (current[0], current[1] + np.sign(dy))
            if (0 <= next_pos[0] < world.shape[0] and 
                0 <= next_pos[1] < world.shape[1] and 
                world[next_pos[0], next_pos[1]] == 0):
                return np.array([0, np.sign(dy)])
        
        # If all fails, find any valid move
        for direction in DIRECTIONS:
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= next_pos[0] < world.shape[0] and 
                0 <= next_pos[1] < world.shape[1] and 
                world[next_pos[0], next_pos[1]] == 0):
                return direction
                
        return np.array([0, 0])  # Stay in place if no valid moves
    
    def emergency_evade(self, world, current, pursuer, predicted_pursuer_positions=None):
        """Perform emergency evasion when pursuer is adjacent"""
        best_move = None
        best_score = float('-inf')
        
        for direction in DIRECTIONS:
            next_pos = current + direction
            
            # Ensure move is valid
            if not (0 <= next_pos[0] < world.shape[0] and 
                    0 <= next_pos[1] < world.shape[1] and 
                    world[next_pos[0], next_pos[1]] == 0):
                continue
            
            # Calculate base score based on distance from pursuer
            score = manhattan_distance(next_pos, pursuer) * 10
            
            # Consider predicted pursuer positions if available
            if predicted_pursuer_positions:
                # Reduce score for positions at risk of being caught
                for p_pos, p_prob in predicted_pursuer_positions:
                    if manhattan_distance(next_pos, p_pos) <= 1:
                        score -= 50 * p_prob
            
            # Check if this position was recently visited (avoid cycles)
            position_penalty = 15 if tuple(next_pos) in self.last_positions else 0
            score -= position_penalty
            
            # Only consider moves that increase distance from pursuer
            if manhattan_distance(next_pos, pursuer) > manhattan_distance(current, pursuer) and score > best_score:
                best_score = score
                best_move = direction
                
        return best_move
    
    def evaluate_move(self, world, current, next_pos, pursued, pursuer, predicted_pursuer=None):
        """Evaluate a potential move with advanced strategic considerations"""
        # Immediate capture has highest value
        if np.array_equal(next_pos, pursued):
            return CAPTURE_REWARD
            
        # Calculate distances
        current_to_target = manhattan_distance(current, pursued)
        next_to_target = manhattan_distance(next_pos, pursued)
        next_to_pursuer = manhattan_distance(next_pos, pursuer)
        
        # Calculate base score
        score = (current_to_target - next_to_target) * DISTANCE_WEIGHT
        
        # Add bonus for moves that maintain safe distance from pursuer
        if next_to_pursuer > 1:
            score += next_to_pursuer * 20
        
        # Consider predicted pursuer positions if available
        if predicted_pursuer:
            for p_pos, p_prob in predicted_pursuer:
                if manhattan_distance(next_pos, p_pos) <= 1:
                    score -= 100 * p_prob
        
        # Penalty for recently visited positions (avoid cycles)
        if tuple(next_pos) in self.last_positions:
            score -= 150
            
        # Bonus for positions that limit target's escape options
        target_escape_count = 0
        for direction in DIRECTIONS:
            escape_pos = pursued + direction
            if (0 <= escape_pos[0] < world.shape[0] and 
                0 <= escape_pos[1] < world.shape[1] and 
                world[escape_pos[0], escape_pos[1]] == 0):
                target_escape_count += 1
                
        if next_to_target <= 2 and target_escape_count <= 3:
            score += 300  # Big bonus for cutting off escape routes
            
        # Bonus for moves that lead to corridors where target can be trapped
        if next_to_target <= 3 and target_escape_count <= 2:
            score += 400  # Extreme bonus for trapping potential
            
        return score
    
    def find_best_move(self, world, current, pursued, pursuer, predicted_pursuer=None):
        """Find the best move using evaluation of all possible moves"""
        best_move = None
        best_score = float('-inf')
        
        for direction in DIRECTIONS:
            next_pos = current + direction
            
            # Ensure move is valid
            if not (0 <= next_pos[0] < world.shape[0] and 
                    0 <= next_pos[1] < world.shape[1] and 
                    world[next_pos[0], next_pos[1]] == 0):
                continue
                
            # Skip moves that put us adjacent to pursuer unless it's a capture move
            if manhattan_distance(next_pos, pursuer) <= 1 and not np.array_equal(next_pos, pursued):
                # Check if this is a predicted position
                if not predicted_pursuer:
                    continue
                
                # Allow risky moves if probability of capture is low
                risky = False
                for p_pos, p_prob in predicted_pursuer:
                    if manhattan_distance(next_pos, p_pos) <= 1 and p_prob > 0.4:
                        risky = True
                        break
                
                if risky:
                    continue
                
            # Evaluate the move
            score = self.evaluate_move(world, current, next_pos, pursued, pursuer, predicted_pursuer)
            
            if score > best_score:
                best_score = score
                best_move = direction
                
        return best_move
    
    def plan_action(self, world: np.ndarray, current: Tuple[int, int], 
                   pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:
        """Plan the next action using an ultra-aggressive approach with learning and prediction"""
        # Convert inputs to numpy arrays
        current = np.array(current)
        pursued = np.array(pursued)
        pursuer = np.array(pursuer)
        
        # Initialize position on first call
        if self.initial_position is None:
            self.initial_position = current
            self.last_pursuer_pos = pursuer
            self.last_target_pos = pursued
        else:
            # Track opponent movements
            self.track_opponent_movement(pursuer, self.last_pursuer_pos, pursued, self.last_target_pos)
            self.last_pursuer_pos = pursuer
            self.last_target_pos = pursued
            
            # Learn from our last action if we took one
            if len(self.last_positions) >= 2:
                last_pos = np.array(self.last_positions[-2])
                intended_action = current - last_pos
                if np.any(intended_action != 0):  # If we moved
                    # Find the direction we intended vs what happened
                    intended_direction = None
                    for direction in DIRECTIONS:
                        if np.array_equal(direction, intended_action):
                            intended_direction = direction
                            break
                    
                    if intended_direction is not None:
                        # Update transition probabilities
                        self.update_transition_probabilities(intended_direction, intended_action)
        
        # Update position history (for cycle detection)
        self.last_positions.append(tuple(current))
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)
        
        # IMMEDIATE CAPTURE: If adjacent to target, capture immediately
        if manhattan_distance(current, pursued) == 1:
            return pursued - current
        
        # Get predicted positions for pursuer
        predicted_pursuer = None
        if len(self.pursuer_actions) > 0:
            pursuer_next_positions, pursuer_probs = self.predict_next_position(
                pursuer, self.pursuer_history, self.pursuer_actions, 
                self.pursuer_p_left, self.pursuer_p_straight
            )
            predicted_pursuer = list(zip(pursuer_next_positions, pursuer_probs))
        
        # CRITICAL EVASION: If pursuer is adjacent, prioritize evasion
        if manhattan_distance(current, pursuer) <= 1:
            evasion_move = self.emergency_evade(world, current, pursuer, predicted_pursuer)
            if evasion_move is not None:
                return apply_probabilistic_transition(evasion_move, P_LEFT, P_STRAIGHT)
        
        # AGGRESSIVE PURSUIT: Find direct path to target with pursuer avoidance
        pursuit_move = self.direct_pursuit(world, current, pursued, pursuer)
        
        # Check if pursuit move would put us in danger
        next_pos = current + pursuit_move
        if (0 <= next_pos[0] < world.shape[0] and 
            0 <= next_pos[1] < world.shape[1]):
            
            # Check against actual pursuer
            direct_danger = manhattan_distance(next_pos, pursuer) <= 1
            
            # Check against predicted pursuer positions
            predicted_danger = False
            if predicted_pursuer:
                for p_pos, p_prob in predicted_pursuer:
                    if p_prob > 0.4 and manhattan_distance(next_pos, p_pos) <= 1:
                        predicted_danger = True
                        break
            
            # If not in danger, pursue directly
            if not direct_danger and not predicted_danger:
                return apply_probabilistic_transition(pursuit_move, P_LEFT, P_STRAIGHT)
        
        # If direct pursuit is dangerous, find best alternative move
        best_move = self.find_best_move(world, current, pursued, pursuer, predicted_pursuer)
        if best_move is not None:
            return apply_probabilistic_transition(best_move, P_LEFT, P_STRAIGHT)
            
        # If no good move found, try direct pursuit anyway
        return apply_probabilistic_transition(pursuit_move, P_LEFT, P_STRAIGHT)



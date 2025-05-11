import numpy as np
import math
import random
from typing import List, Tuple, Optional, Dict
from collections import deque, defaultdict

class PlannerAgent:
    def __init__(self):
        # Movement directions: [stay, up, down, left, right, up-left, up-right, down-left, down-right]
        self.directions = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1], [-1,-1], [-1,1], [1,-1], [1,1]])
        
        # Maps each action to its possible rotations
        self.actions_map = {0:[0,0,0], 1:[3,1,4], 2:[4,2,3], 3:[2,3,1], 4:[1,4,2], 5:[7,5,6], 6:[5,6,8], 7:[2,7,5], 8:[6,8,2]}
        
        self.learning_rate = 0.15
        self.discount_factor = 0.95
        self.epsilon = 0.2
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.05
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32
        self.replay_start_size = 100
        self.q_table = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.last_state = None
        self.last_action = None
        self.step_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.best_reward = float('-inf')
        self.action_history = deque(maxlen=15)
        self.prob_counts = np.array([1, 1, 1])  # Counts for [left, straight, right]
        self.uncertainty = 1.0  # Track uncertainty in probability estimation
        self.direction_indices = np.arange(len(self.directions))
        self.rotation_lookup = {action: {r: i for i, r in enumerate(rotations)} 
                              for action, rotations in self.actions_map.items()}        
        self.distance_cache = {}
    
    def update_probability_distribution(self, current, last_pos, last_action):
        if last_pos is not None and last_action is not None:
            expected = last_pos + last_action
            outcomes = np.array([
                last_pos + np.array([-last_action[1], last_action[0]]),  # Left rotation
                expected,                                                 # Expected
                last_pos + np.array([last_action[1], -last_action[0]])   # Right rotation
            ])
            
            # Find which outcome matches current position
            diff = np.abs(outcomes - current)
            matches = np.sum(diff, axis=1) == 0
            if np.any(matches):
                result = np.where(matches)[0][0]
                self.action_history.append(result)
                self.prob_counts[result] += 1
                
                # Update uncertainty based on history length and consistency
                if len(self.action_history) >= 5:
                    total = np.sum(self.prob_counts)
                    if total > 0:
                        probs = self.prob_counts / total
                        entropy = -np.sum(probs * np.log2(probs + 1e-10))
                        self.uncertainty = 1.0 - (entropy / np.log2(3))
    
    def get_state_key(self, current, pursued, pursuer):
        def discretize(pos, ref):
            dx = (pos[0] - ref[0]) // 3
            dy = (pos[1] - ref[1]) // 3
            return (dx, dy)
        
        pursued_rel = discretize(pursued, current)
        pursuer_rel = discretize(pursuer, current)
        dist_to_pursued = abs(current[0] - pursued[0]) + abs(current[1] - pursued[1])
        dist_to_pursuer = abs(current[0] - pursuer[0]) + abs(current[1] - pursuer[1])
        
        # Discretize distances
        dist_pursued_bin = min(dist_to_pursued // 5, 3)  # 0-3 bins
        dist_pursuer_bin = min(dist_to_pursuer // 5, 3)  # 0-3 bins
        
        uncertainty_bin = min(int(self.uncertainty * 3), 2)  # 0-2 bins
        
        return (pursued_rel, pursuer_rel, dist_pursued_bin, dist_pursuer_bin, uncertainty_bin)
    
    def get_legal_actions(self, world, pos):
        new_positions = pos + self.directions
        valid_mask = (
            (new_positions[:, 0] >= 0) & 
            (new_positions[:, 0] < world.shape[0]) & 
            (new_positions[:, 1] >= 0) & 
            (new_positions[:, 1] < world.shape[1]) & 
            (world[new_positions[:, 0], new_positions[:, 1]] == 0)
        )
        return self.direction_indices[valid_mask]
    
    def calculate_reward(self, current, pursued, pursuer):
        dist_to_pursued = abs(current[0] - pursued[0]) + abs(current[1] - pursued[1])
        dist_to_pursuer = abs(current[0] - pursuer[0]) + abs(current[1] - pursuer[1])
        
        # Immediate win/loss rewards
        if dist_to_pursued == 0:
            return 200.0
        if dist_to_pursuer == 0:
            return -200.0
        
        # Base reward for getting closer to target
        # Exponential decay to prioritize getting very close
        pursuit_reward = 2.0 * np.exp(-dist_to_pursued / 5.0)
        
        # Safety reward with multiple zones
        if dist_to_pursuer <= 1:
            safety_reward = -3.0  # Heavy penalty for being too close
        elif dist_to_pursuer <= 3:
            safety_reward = -1.0  # Moderate penalty for being close
        elif dist_to_pursuer <= 5:
            safety_reward = 0.5   # Small bonus for maintaining safe distance
        else:
            safety_reward = 0.0
        
        # If pursuer is closer to us than we are to pursued, prioritize safety
        if dist_to_pursuer < dist_to_pursued:
            pursuit_weight = 0.3
            safety_weight = 0.7
        else:
            pursuit_weight = 0.7
            safety_weight = 0.3
        
        reward = (pursuit_weight * pursuit_reward + 
                 safety_weight * safety_reward)
        if np.array_equal(current, self.last_state):
            reward -= 0.2
        reward *= (1.0 - self.uncertainty * 0.5)
        
        return reward
    
    def choose_action(self, state, legal_actions, epsilon):
        if random.random() < epsilon * (1.0 + self.uncertainty):  # More exploration when uncertain
            return random.choice(legal_actions)
        
        # Get Q-values for legal actions
        q_values = [self.q_table[state][action] for action in legal_actions]
        if not q_values:
            return 0
        
        # Return action with highest Q-value
        return legal_actions[np.argmax(q_values)]
    
    def update_q_value(self, state, action, reward, next_state, next_legal_actions):
        # Store experience in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, next_legal_actions))
        
        # Update Q-value for current experience
        if not next_legal_actions:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state][a] for a in next_legal_actions)
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        if len(self.replay_buffer) >= self.replay_start_size:
            self._experience_replay()
    
    def _experience_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for state, action, reward, next_state, next_legal_actions in batch:
            if not next_legal_actions:
                max_next_q = 0
            else:
                max_next_q = max(self.q_table[next_state][a] for a in next_legal_actions)
            
            current_q = self.q_table[state][action]
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state][action] = new_q
    
    def plan_action(self, world: np.ndarray, current: Tuple[int, int], pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:
        try:
            self.step_count += 1
            
            current = np.array(current)
            pursued = np.array(pursued)
            pursuer = np.array(pursuer)
            
            self.update_probability_distribution(current, self.last_state, self.last_action)
            
            current_state = self.get_state_key(current, pursued, pursuer)
            legal_actions = self.get_legal_actions(world, current)
            
            if not legal_actions:
                return self.directions[0]
            
            for action in legal_actions:
                new_pos = current + self.directions[action]
                if np.array_equal(new_pos, pursued):
                    return self.directions[action]
            
            dist_to_pursuer = abs(current[0] - pursuer[0]) + abs(current[1] - pursuer[1])
            if dist_to_pursuer <= 1:
                new_positions = current + self.directions[legal_actions]
                dists_to_pursuer = np.sum(np.abs(new_positions - pursuer), axis=1)
                return self.directions[legal_actions[np.argmax(dists_to_pursuer)]]
            
            # Update exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            action_idx = self.choose_action(current_state, legal_actions, self.epsilon)
            
            next_pos = current + self.directions[action_idx]
            next_legal_actions = self.get_legal_actions(world, next_pos)
            next_state = self.get_state_key(next_pos, pursued, pursuer)
            
            reward = self.calculate_reward(next_pos, pursued, pursuer)
            
            if self.last_state is not None and self.last_action is not None:
                self.update_q_value(self.last_state, self.last_action, reward, current_state, legal_actions)
            
            self.last_state = current_state
            self.last_action = action_idx
            
            return self.directions[action_idx]
            
        except Exception as e:
            return self.directions[0]
    
    def is_valid_action(self, world, pos, action):
        new_pos = pos + action
        return (0 <= new_pos[0] < world.shape[0] and 0 <= new_pos[1] < world.shape[1] and 
                world[new_pos[0], new_pos[1]] == 0)



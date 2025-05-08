# Pursue-Escape Planning Algorithm Development

## Overview

This project extends Project-I to a multi-agent pursue-escape setting involving three distinct agents: Tom, Jerry, and Spike. Each agent must make real-time decisions based on its current position, its pursued target, and its own pursuer:
- Tom is trying to capture Jerry.
- Jerry is trying to capture Spike.
- Spike is trying to capture Tom.
  
Each agent independently plans its next move on a shared grid-based environment with static obstacles. The challenge lies in designing effective planning strategies under adversarial, dynamic conditions.

## Project Structure
```plaintext
project_root/
│── data/                # Contains grid environments and initial positions for all three agents
│── main.py              # Loads environments, runs simulations, handles visualization (DO NOT modify)
│── planners/            # Contains planning code for all three agents (Tom, Jerry, Spike)
│   ├── planner_tom.py   # Planning logic for Tom
│   ├── planner_jerry.py # Planning logic for Jerry
│   └── planner_spike.py # Planning logic for Spike
│── devel.py             # Development script for testing (modifiable, NOT submitted)
│── README.md            # Project documentation
```

## Task Configurations

- The `data/` folder contains **100 different task configurations**, each represented as a grid file.
- Each file specifies a **grid representation**.

### Scoring Rules

Each task ends when one of the following occurs:
- An agent **successfully captures its pursued target** without being caught by its own pursuer and without colliding into obstacles.
- A **collision** or **stalemate** (e.g., no one can catch their target) occurs within a fixed number of steps.

Scoring for each task is as follows:
- **3 points**: Awarded to the **winning agent** — the first to successfully complete its pursuit objective without being caught or crashing into an obstacle.
- **0 points**: Awarded to the losing agents in a clear win/loss outcome.
- **1 point each**: Awarded to all agents if the task results in a **tie** (e.g., multiple captures in the same step or a timed-out draw).


## Implementation Details

### `main.py` (DO NOT MODIFY)
The `main.py` file provides the **execution routine** that sets the stage for the three-agent competition.
- The `run()` function in the Task class orchestrates the full simulation:
  - It initializes the environment, loads agent planners, and manages the turn-by-turn logic.
- At each timestep, the `step()` function is called to advance the simulation:
  - Each agent’s `plan_action()` method is called with its current position, pursued target, and pursuer.
  - The returned action is **checked for validity** (e.g., bounds).
  - Only valid actions are executed; otherwise, the agent remains in its current position.
- The file also provides visualization support and result logging.
- **This file must not be modified.**

### `planner.py` (MAIN DEVELOPMENT FILE)
Each agent has a separate planner file in the `planners/` directory. All planner files must follow a consistent structure, defined by the `PlannerAgent` class.

### Planner Interface

```python
class PlannerAgent:
    def __init__(self):
        pass

    def plan_action(
        world: np.ndarray,
        current: Tuple[int, int],
        pursued: Tuple[int, int],
        pursuer: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Determines the agent’s next move.

        Parameters:
        - world: 2D numpy array representing the environment
        - current: (row, col) position of the agent
        - pursued: (row, col) position of the target the agent is chasing
        - pursuer: (row, col) position of the agent chasing this agent

        Returns:
        - A numpy array [dr, dc] representing the direction of movement
        """
```

- This is where students will **implement their planning algorithm**.
- A **DFS (Depth-First Search) example** is provided in jerry.py and a **random sapmling example** is provided in tom.py and spike.py to illustrate the function structure.
- **The plan_action function name and input-output structure must remain unchanged**, but the internal implementation can be modified.
- Students can use any suitable adversarial searching and learning approach. 
- Students can only use the following packages:
  - The Python standard libraries
  - Numpy
  - Scipy
  - Matplotlib
  - Pandas
- Directly re-using the provided example is plagiarism. Using only heuristics without a formal searching algorithm is also plagiarism.

### `devel.py` (MODIFIABLE, BUT NOT SUBMITTED)
- Provides an example of solving **one task** using functions from `main.py` and planners.
- Students are **free to modify** this file for **development and testing purposes**.
- **This file will NOT be included in the final submission**.

## Submission Requirements

- **Only `planner.py` should be submitted** to Canvas (along with the project report in typeset pdf).
- The submitted `planner.py` will overwrite `tom.py`, `jerry.py`, or `spike.py` based on the competition setup. 
- The `PlannerAgent` class signature must remain unchanged.
- The implementation should be tested using `main.py` before submission.

## Getting Started

1. Read the program in `main.py` to understand the task structure and validity criterion.
2. Implement the path planning algorithm inside `planners`, the `.py` files can have identical or different designs of planners.
3. Use `devel.py` to test and debug the implementation.
4. Ensure the code is valid by executing
  ```bash
  python main.py
  ```
5. Submit only the `planner.py` file.
  - this file may have the name being `tom.py`, `jerry.py`, or `spike.py` based on your local setup, please rename the file to `planner.py`
  - you may have multiple versions of the planner, please submit one and only one version of your selection

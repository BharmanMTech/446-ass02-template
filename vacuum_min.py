"""
================================================================================
BFS FOR VACUUM CLEANER — QUICK START GUIDE
================================================================================

Description of Implementation

1. Define Classes:
   - Percept: Represents the agent's perception of the environment.
   - Action: Represents the action taken by the agent.
   - Sensor: Abstract base class for sensors.
   - FullSensor: A sensor that can see the entire environment.
   - Agent: Abstract base class for agents.
   - RandomAgent: An agent that chooses actions randomly.
   - SearchNode: Represents a node in the search tree.
   - VacuumSearchProblem: Defines the search problem.
   - BFSAgent: An agent that uses Breadth-First Search to plan actions.

2. Implement Methods:
   - Percept Class:
     - __init__: Initializes the percept with position, status, bump, and visible cells.
     - __repr__: Returns a string representation of the percept.
   - Action Class:
     - VALID_MOVES: A set of valid moves.
     - __init__: Initializes the action with a clean flag and move.
     - __repr__: Returns a string representation of the action.
   - Sensor Class:
     - Abstract method read that must be implemented by subclasses.
   - FullSensor Class:
     - Implements the read method to return a percept with the full state of the environment.
   - Agent Class:
     - Abstract method decide that must be implemented by subclasses.
   - RandomAgent Class:
     - Implements the decide method to choose a random action.
   - SearchNode Class:
     - __init__: Initializes the search node with state, parent, action, and path cost.
     - get_path: Returns the path from the root to the current node.
   - VacuumSearchProblem Class:
     - __init__: Initializes the search problem with initial position, dirty cells, and grid size.
     - is_goal: Checks if the current state is the goal state.
     - get_actions: Returns a list of possible actions from the current state.
     - transition_model: Returns the next state after taking an action.
   - BFSAgent Class:
     - Implements the _compute_plan method to compute the plan using BFS.
     - Implements the decide method to use the computed plan to decide the next action.

3. Test the Implementation:
   - Create an instance of VacuumSearchProblem with the initial state.
   - Use an instance of BFSAgent to compute the plan and verify that it correctly solves the problem.
   - Compare the results with RandomAgent to see the difference in performance.

Example Usage

# Create an instance of VacuumSearchProblem
initial_position = (0, 0)
dirty_cells = {(1, 1), (2, 2)}
problem = VacuumSearchProblem(initial_position, dirty_cells)

# Create an instance of BFSAgent
bfs_agent = BFSAgent()

# Compute the plan
plan = bfs_agent._compute_plan(problem)
print("Plan:", plan)

# Decide the next action
percept = FullSensor().read(problem)  # Assuming problem has a method to simulate the environment
action = bfs_agent.decide(percept)
print("Action:", action)

"""

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple, Set, Dict, Any, FrozenSet, Optional
import random

# Type aliases
Position = Tuple[int, int]
SearchState = Tuple[Position, FrozenSet[Position]]

# ==============================================================================
# 1. INTERFACE DEFINITIONS (DO NOT MODIFY)
# ==============================================================================

class Percept:
    def __init__(self, position: Position, status: str, bump: Optional[bool] = None, visible_cells: Optional[Dict[Position, str]] = None):
        self.position = position
        self.status = status
        self.bump = bump
        self.visible_cells = visible_cells

    def __repr__(self) -> str:
        return f"Percept(pos={self.position}, status={self.status}, dirty_count={sum(1 for s in self.visible_cells.values() if s == 'Dirty') if self.visible_cells else 0})"


class Action:
    VALID_MOVES = {"Up", "Down", "Left", "Right", "NoOp", "Suck"}

    def __init__(self, clean: bool, move: str):
        if move not in self.VALID_MOVES:
            raise ValueError(f"Invalid move: '{move}'")
        self.clean = clean
        self.move = move

    def __repr__(self) -> str:
        return f"Action(clean={self.clean}, move={self.move})"


class Sensor(ABC):
    @abstractmethod
    def read(self, env: 'Any') -> Percept:
        pass


class FullSensor(Sensor):
    def read(self, env: 'Any') -> Percept:
        return Percept(
            position=env.agent_position(),
            status=env.cell_status(env.agent_position()),
            visible_cells=env.all_cells()
        )


class Agent(ABC):
    def __init__(self, sensor: Sensor):
        self.sensor = sensor

    @abstractmethod
    def decide(self, percept: Percept) -> Action:
        pass


class RandomAgent(Agent):
    def __init__(self):
        super().__init__(sensor=FullSensor())

    def decide(self, percept: Percept) -> Action:
        clean = (percept.status == "Dirty")
        available_moves = Action.VALID_MOVES if clean else Action.VALID_MOVES - {"Suck"}
        move = random.choice(list(available_moves))
        return Action(clean=clean, move=move)


# ==============================================================================
# 2. STUDENT IMPLEMENTATION SECTION
# ==============================================================================

class SearchNode:
    """Represents a node in the search tree."""
    def __init__(self, state: SearchState, parent: Optional['SearchNode'] = None, action: Optional[str] = None, path_cost: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def get_path(self) -> List[str]:
        """Reconstructs the action path from the root to this node."""
        path = []
        node = self          
        while node.parent != None:            
            path.append(node.action)  
            node = node.parent   
        path.reverse()
        return path



class VacuumSearchProblem:
    """Defines the state space and rules for the Vacuum Cleaner environment."""
    def __init__(self, initial_position: Position, dirty_cells: Set[Position], grid_size: int = 5):
        self.grid_size = grid_size
        self.initial: SearchState = (initial_position, frozenset(dirty_cells))

    def is_goal(self, state: SearchState) -> bool:
        """Returns True if the given state satisfies the goal condition."""
        dirty = state[1]
        return len(dirty) == 0


    def get_actions(self, state: SearchState) -> List[str]:
        """Returns valid actions from the given state without stepping out of grid bounds."""
        position = state[0]
        dirty = state[1]
        x, y = position            
        actions = []

        if y > 0:                     
            actions.append("Up")
        if y < (self.grid_size - 1):                   
            actions.append("Down")
        if x > 0:                    
            actions.append("Left")
        if x < (self.grid_size - 1):                    
            actions.append("Right")
        if position in dirty:                   
            actions.append("Suck")

        return actions

    def transition_model(self, state: SearchState, action: str) -> SearchState:
        """Applies an action to a state and returns the resulting next state."""
        position = state[0]
        dirty = state[1]
        x, y = position

        if action == "Suck":
            return (position, dirty - {position})      
        if action == "Up":
            return ((x, y - 1), dirty)    
        if action == "Down":
            return ((x, y + 1), dirty)
        if action == "Left":
            return ((x - 1, y), dirty)
        if action == "Right":
            return ((x + 1, y), dirty)


class BFSAgent(Agent):
    """An agent that uses Breadth-First Search to find the optimal plan."""
    def __init__(self):
        super().__init__(sensor=FullSensor())

    def _compute_plan(self, problem: VacuumSearchProblem) -> List[str]:
        """Performs Breadth-First Search on the problem and returns a list of action strings."""
        root = SearchNode(problem.initial)                
        if problem.is_goal(root.state):              
            return []

        frontier = deque([root])
        reached = {problem.initial}

        while len(frontier) != 0:                            
            node = frontier.popleft()              

            for action in problem.get_actions(node.state):
                child_state = problem.transition_model(node.state, action)

                if child_state in reached:    
                    continue                 

                child = SearchNode(child_state, parent=node, action=action, path_cost=(node.path_cost + 1))   # 5a

                if problem.is_goal(child_state):      
                    return child.get_path()                

                reached.add(child_state)          
                frontier.append(child)         

        return [] 

    def decide(self, percept: Percept) -> Action:
        """Parses the percept, computes a search plan, and returns the next immediate Action."""
        dirty = set()
        for pos, status in percept.visible_cells.items():
            if status == "Dirty":                                      # 1: is this cell dirty?
                dirty.add(pos)

        grid_size = int(len(percept.visible_cells) ** 0.5)

        problem = VacuumSearchProblem(percept.position, dirty, grid_size)     # position, dirty cells, grid size
        plan = self._compute_plan(problem)

        if len(plan) == 0:                               # 4: nothing to do
            return Action(clean=False, move="NoOp")

        move = plan[0]                                  # 3: first action of the plan
        return Action(clean=(move == "Suck"), move=move)


# ==============================================================================
# 3. VERIFICATION & TESTING ENVIRONMENT
# ==============================================================================

# class MockEnvironment:
#     """Simple grid environment for testing."""
#     def __init__(self, agent_pos: Position, dirty_cells: Set[Position], grid_size: int = 5):
#         self.pos = agent_pos
#         self.dirty = set(dirty_cells)
#         self.size = grid_size
#
#     def agent_position(self) -> Position:
#         return self.pos
#
#     def cell_status(self, pos: Position) -> str:
#         return "Dirty" if pos in self.dirty else "Clean"
#
#     def all_cells(self) -> Dict[Position, str]:
#         return {(c, r): "Dirty" if (c, r) in self.dirty else "Clean"
#                 for c in range(self.size) for r in range(self.size)}


class VacuumEnvironment:
    """Grid world that applies agent actions and tracks performance."""
    def __init__(self, agent_pos: Position, dirty_cells: Set[Position], grid_size: int = 5):
        self.pos = agent_pos
        self.dirty = set(dirty_cells)
        self.size = grid_size
        self.steps = 0
        self.bumped = False

    def agent_position(self) -> Position:
        return self.pos

    def cell_status(self, pos: Position) -> str:
        return "Dirty" if pos in self.dirty else "Clean"

    def all_cells(self) -> Dict[Position, str]:
        return {(c, r): "Dirty" if (c, r) in self.dirty else "Clean"
                for c in range(self.size) for r in range(self.size)}
    
    def execute(self, action: Action) -> None:
        """Applies the agent's action to the world."""
        self.steps += 1                         
        self.bumped = False
        x, y = self.pos

        if action.move == "Suck":
            self.dirty.discard(self.pos)            
        elif action.move == "NoOp":
            pass                             
        else:
            if action.move == "Up":
                new_pos = (x, y - 1)
            elif action.move == "Down":
                new_pos = (x, y + 1)
            elif action.move == "Left":
                new_pos = (x - 1, y)
            else:                               
                new_pos = (x + 1, y)

            nx, ny = new_pos
            if (0 <= nx < self.size) and (0 <= ny < self.size):                            
                self.pos = (nx, ny)
            else:
                self.bumped = True         

    def is_clean(self) -> bool:
        return len(self.dirty) == 0


if __name__ == "__main__":
    initial_pos = (0, 0)
    dirty_set = {(1, 1), (2, 2)}
    max_steps = 1000

    random.seed(446)
    for agent in [BFSAgent(), RandomAgent()]:
        env = VacuumEnvironment(initial_pos, dirty_set)
        while not env.is_clean() and env.steps < max_steps:
            env.execute(agent.decide(agent.sensor.read(env)))
        print(f"{type(agent).__name__}: cleaned={env.is_clean()}, steps={env.steps}")

# Brock Harman
# CSCI 446 Fall 2026
# Programming Assignment #1
# I declare that I am the author of this work, take full responsibility for it, and have disclosed any material external assistance.
# I used Claude to walkthrough the assignment while I wrote, tested, and verified the submitted implementation myself.
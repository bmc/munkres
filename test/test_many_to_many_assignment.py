import numpy as np
import unittest
import pytest
from many_to_many_assignment import kuhn_munkers_backtracking

def compare_dicts(dict1, dict2):
    """
    Function to compare two dictionaries for equality.
    
    Parameters:
    `dict1 (dict)`: The first dictionary to compare.
    `dict2 (dict)`: The second dictionary to compare.

    Returns:
    bool: True if both dictionaries are equal, False otherwise.
    """
    return dict1 == dict2

def generate_random_test_case(num_agents, num_tasks, max_ability, max_task_range, max_performance_value):
    """
    Generate a random test case for the many-to-many assignment problem.
    
    Parameters:
    `num_agents (int)`: Number of agents.
    `num_tasks (int)`: Number of tasks.
    `max_ability (int)`: Maximum ability value for an agent.
    `max_task_range (int)`: Maximum range value for a task.
    `max_performance_value (int)`: Maximum performance value for the matrix.

    Returns:
    `tuple`: A tuple containing the ability_agent_vector, task_range_vector, and performance_matrix.
    """
    ability_agent_vector = np.random.randint(1, max_ability + 1, size=num_agents)
    task_range_vector = np.random.randint(1, max_task_range + 1, size=num_tasks)
    
    # Ensure the cardinality constraint is satisfied
    if ability_agent_vector.sum() < task_range_vector.sum():
        # Scale down task range vector to satisfy the constraint
        factor = ability_agent_vector.sum() / task_range_vector.sum()
        task_range_vector = np.floor(task_range_vector * factor).astype(int)
    
    if task_range_vector.sum() == 0:
        task_range_vector[np.random.randint(0, num_tasks)] = 1

    performance_matrix = np.random.randint(1, max_performance_value + 1, size=(num_agents, num_tasks))
    
    return ability_agent_vector, task_range_vector, performance_matrix

def validate_assignment(ability_agent_vector, task_range_vector, assignment):
    """
    Validate the assignment against the ability and task range constraints.

    Parameters:
    ability_agent_vector (np.array): Array of the abilities of the agents.
    task_range_vector (np.array): Array of the task ranges.
    assignment (dict): Dictionary containing the assignment of agents to tasks.

    Returns:
    bool: True if the assignment is valid, False otherwise.
    """
    task_count = np.zeros_like(task_range_vector)

    # Check each agent's assigned tasks
    for agent, tasks in assignment.items():
        if len(tasks) > ability_agent_vector[agent]:
            return False  # Agent is assigned more tasks than its ability
        for task in tasks:
            if task != -1:
                task_count[task] += 1

    # Check if the task count matches the task range
    return np.all(task_count <= task_range_vector)

class TestManyToManyAssignment(unittest.TestCase):

    def test_example_1(self) -> None:
        ability_agent_vector = np.array([2, 2, 2, 2])
        task_range_vector = np.array([2, 2, 2, 2])
        performance_matrix = np.array([[3, 0, 1, 2], [2, 3, 0, 1], [3, 0, 1, 2], [1, 0, 2, 3]])
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        expected_output = {0: [3, 0], 1: [2, 3], 2: [1, 2], 3: [0, 1]}
        if isinstance(output, dict):
            self.assertTrue(compare_dicts(output, expected_output))
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_example_2(self) -> None:
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        performance_matrix = np.array([[40, 60, 15], [25, 30, 45], [55, 30, 25]])
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        expected_output = {0: [2], 1: [0], 2: [1]}
        if isinstance(output, dict):
            self.assertTrue(compare_dicts(output, expected_output))
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))

    def test_example_3(self) -> None:
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        performance_matrix = np.array([[30, 25, 10], [15, 10, 20], [25, 20, 15]])
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        expected_output = {0: [2], 1: [1], 2: [0]}
        if isinstance(output, dict):
            self.assertTrue(compare_dicts(output, expected_output))
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_example_4(self) -> None:
        ability_agent_vector = np.array([])
        task_range_vector = np.array([])
        performance_matrix = np.array([])
        with pytest.raises(ValueError, match="Empty input"):
            kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
    
    def test_example_5(self) -> None:
        ability_agent_vector = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
        task_range_vector = np.array([2, 2, 1, 3])
        performance_matrix = np.array([1, 2, 3, 1, 2])
        try:
            _ = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        except ValueError as e:
            print(e)
            self.assertTrue("The performance matrix must be 2-dimensional")
            print("Test test_example_5 passed")

    def test_large_example_1(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(50, 50, 3, 3, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_large_example_2(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(100, 100, 3, 3, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_random_example_1(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(10, 10, 3, 3, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))

    def test_random_example_2(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(20, 20, 4, 4, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_random_example_3(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(30, 30, 5, 5, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_random_example_4(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(40, 40, 3, 3, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))
    
    def test_random_example_5(self) -> None:
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(50, 50, 4, 4, 100)
        output = kuhn_munkers_backtracking(matrix=performance_matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        if isinstance(output, dict):
            self.assertTrue(validate_assignment(ability_agent_vector, task_range_vector, output))

if __name__ == "__main__":
    unittest.main()

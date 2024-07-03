"""
	An implementation of the algorithms in:
 
	"Solving the Many to Many assignment problem by improving
    the Kuhn–Munkres algorithm with backtracking".

    Authors: Haibin Zhu, Dongning Liu, Siqin Zhang, Yu Zhu, Luyao Teng, and Shaohua Teng.

    https://www.sciencedirect.com/science/article/pii/S0304397516000037

    Programmers: Tom Shabalin and Dor Harizi.
    Date: 2024-05-16.
"""

import numpy as np
import logging

# 1. Create a logger instance
logger = logging.getLogger("many_to_many_assignment")

def kuhn_munkers_backtracking(matrix: np.asarray, agentVector: np.asarray, taskRangeVector: np.asarray) -> dict:
    """
    Solving the Many to Many assignment problem by improving the Kuhn–Munkres algorithm with backtracking.

    Parameters
    ----------
    `agentVector`: Vector of the abilities of the agents
    `taskRangeVector`: Vector of the task ranges
    `matrix`: Performance matrix of the agents and tasks

    Returns
    ----------
    Dictionary containing the assignment of agents to tasks.

    Example 1:
    ----------
    >>> matrix = np.array([[3, 0, 1, 2],[2, 3, 0, 1],[3, 0, 1, 2],[1, 0, 2, 3]])
    >>> ability_agent_vector = np.array([2,2,2,2])
    >>> task_range_vector = np.array([2,2,2,2])
    >>> kuhn_munkers_backtracking(matrix, ability_agent_vector, task_range_vector)
    {0: [3, 0], 1: [2, 3], 2: [1, 2], 3: [0, 1]}

    Example 2:
    ----------
    >>> matrix = np.array([[40, 60, 15],[25, 30, 45],[55, 30, 25]])
    >>> ability_agent_vector = np.array([1, 1, 1])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> kuhn_munkers_backtracking(matrix, ability_agent_vector, task_range_vector)
    {0: [2], 1: [0], 2: [1]}

    Example 3:
    ----------
    >>> matrix = np.array([[30, 25, 10],[15, 10, 20],[25, 20, 15]])
    >>> ability_agent_vector = np.array([1, 1, 1])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> kuhn_munkers_backtracking(matrix, ability_agent_vector, task_range_vector)
    {0: [2], 1: [1], 2: [0]}

    Example 4:
    ----------
    >>> matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
    >>> ability_agent_vector = np.array([2, 2, 1, 3])
    >>> task_range_vector = np.array([1, 2, 3, 1, 2])
    >>> try:
    ...     kuhn_munkers_backtracking(matrix, ability_agent_vector, task_range_vector)
    ... except ValueError as e:
    ...     print(e)
    The Cordinality Constraint is not satisfied, with agents summing to 8 and tasks summing to 9.
    """
    matrix = np.asarray(matrix)
    agentVector = np.asarray(agentVector)
    taskRangeVector = np.asarray(taskRangeVector)

    # Check for empty input
    if 0 in matrix.shape or len(agentVector) == 0 or len(taskRangeVector) == 0:
        raise ValueError("Empty input")
    
    if matrix.ndim != 2:
            raise ValueError("The performance matrix must be 2-dimensional")
        
    if agentVector.ndim != 1:
        raise ValueError("The ability agent vector must be 1-dimensional")
        
    if taskRangeVector.ndim != 1:
        raise ValueError("The task range vector must be 1-dimensional")

    next_state: ManyToManyAssignment = ManyToManyAssignment(matrix, taskRangeVector, agentVector)
    
    current_step = step_1_2_func

    while current_step is not None:
        current_step = current_step(next_state)

    marked = next_state.final_solution
    assignments = np.where(marked == 1)

    # Create the dictionary for assignments
    assignment_dict = {}
    for agent_index, task_index in zip(assignments[0], assignments[1]):
        original_agent = next_state.find_agent_in_row[agent_index]
        if task_index < taskRangeVector.sum():
            original_task = next_state.find_task_in_col[task_index]
        else:
            original_task = -1
        
        if original_agent in assignment_dict:
            assignment_dict[original_agent].append(original_task)
        else:
            assignment_dict[original_agent] = [original_task]
    
    return assignment_dict

def step_1_2_func(state):
    """
    Step 1: Reduce matrix M: for each row of M, find the smallest element and subtract it from every element in its row;
            after that, find the smallest element and subtract it from every element in its column.
    
    Step 2: Initial Stars: find a zero (Z) in M. If there is no starred zero in its row or column, star Z, and adjust zeros to be
            unavailable, which belong to the same agent but in different rows or columns.
    
    Example:
    ----------
    >>> matrix = np.array([[4, 8, 5], [7, 6, 9], [8, 7, 6]])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> agent_vector = np.array([1, 1, 1])
    >>> state = ManyToManyAssignment(matrix, task_range_vector, agent_vector)
    >>> next_step = step_1_2_func(state)
    >>> state.matrix
    array([[0, 4, 1],
           [1, 0, 3],
           [2, 1, 0]])
    >>> state.final_solution
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])
    >>> next_step == step_3_func
    True
    """
    assert isinstance(state, ManyToManyAssignment)
    logger.info(f"------------------Step 1------------------")
    # Subtract the minimum value of each row from all elements of that row
    for i in range(state.matrix.shape[0]):
        min_value = np.min(state.matrix[i])
        state.matrix[i] -= min_value
    # Subtract the minimum value of each column from all elements of that column
    for j in range(state.matrix.shape[1]):
        min_value = np.min(state.matrix[:, j])
        state.matrix[:, j] -= min_value

    logger.info(f'Matrix after step 1: \n{state.matrix}')
    logger.info(f"------------------Step 2------------------")
    # Combine the indicies of the rows and columns where there is a 0.
    for i, j in zip(*np.where(state.matrix == 0)):
        if state.uncolored_columns[j] and state.uncolored_rows[i] and state.available[i,j]:
            state.find_star_zero(row=i, column=j)
            state.uncolored_columns[j] = False
            state.uncolored_rows[i] = False
    logger.debug(f'Final Solution matrix after step 2: \n{state.final_solution}')
    state.uncolor_rows_columns()
    logger.debug(f'uncolored_rows: {state.uncolored_rows}')
    logger.debug(f'uncolored_columns: {state.uncolored_columns}')
    return step_3_func

def step_3_func(state):
    """
    Cover each column containing a starred zero.
    If all the columns are covered, go to Step 7; else go to Step 4.

    Example 1:
    ----------
    >>> matrix = np.array([[0, 2, 0], [0, 0, 0], [0, 1, 0]])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> agent_vector = np.array([1, 1, 1])
    >>> state = ManyToManyAssignment(matrix, task_range_vector, agent_vector)
    >>> state.final_solution = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    >>> next_step = step_3_func(state)
    >>> state.uncolored_columns
    array([False,  True, False])
    >>> next_step == step_4_func
    True

    Example 2:
    ----------
    >>> matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> agent_vector = np.array([1, 1, 1])
    >>> state = ManyToManyAssignment(matrix, task_range_vector, agent_vector)
    >>> state.final_solution = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> next_step = step_3_func(state)
    >>> state.uncolored_columns
    array([False, False, False])
    """
    assert isinstance(state, ManyToManyAssignment)
    logger.info(f"------------------Step 3------------------")
    # Identify the columns in the final solution that contain at least one starred zero
    covered_columns = np.any(state.final_solution == 1, axis=0)

    # Mark the identified columns as covered
    logger.debug(f"before: uncolored_columns: {state.uncolored_columns}")
    state.uncolored_columns[covered_columns] = False
    logger.debug(f"after: uncolored_columns: {state.uncolored_columns}")

    # If there are still uncovered columns, proceed to the next step
    if covered_columns.sum() < state.matrix.shape[1]:
        logger.debug(f'Covered columns sum: {covered_columns.sum()}.\nColumns: {state.matrix.shape[1]}')
        return step_4_func

def step_4_func(state):
    """
    Prime some uncovered zeros: find an uncovered zero and prime it, and adjust zeros to be unavailable, which
    belong to the same agent but in different rows or columns (starred zeros are excluded).

    - If there is no starred zero in the row containing this primed zero, go to Step 5; else, cover this row and
    uncover the column containing the starred zero.

    - Continue until there are no uncovered zeros left.

    - Save the smallest uncovered value and go to Step 6.

    Example:
    ----------
    >>> matrix = np.array([[0, 2, 0], [3, 0, 1], [1, 0, 0]])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> agent_vector = np.array([1, 1, 1])
    >>> state = ManyToManyAssignment(matrix, task_range_vector, agent_vector)
    >>> state.uncolored_rows = np.array([True, True, True])
    >>> state.uncolored_columns = np.array([True, True, True])
    >>> state.available = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    >>> state.final_solution = np.zeros_like(matrix)
    >>> next_step = step_4_func(state)
    >>> state.final_solution
    array([[2, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> next_step == step_5_func
    True
    """
    assert isinstance(state, ManyToManyAssignment)
    logger.info(f"------------------Step 4------------------")
    # If the element is 0, it assigns 1 to the corresponding element in matrix, indicating that the element is uncovered.
    # If the element is not 0 (i.e., it's nonzero), it assigns 0 to the corresponding element in matrix, indicating that the element is covered.
    matrix = np.where(state.matrix == 0, 1, 0)
    logger.info(f'Temp Matrix at Step 4:\n{matrix}')

    # Create a covered matrix
    prime_matrix = matrix * state.uncolored_rows[:, np.newaxis]
    prime_matrix *= np.asarray(state.uncolored_columns, dtype=int)
    prime_matrix *= state.available.astype(int)

    logger.debug(f'Prime Matrix at Step 4:\n{prime_matrix}')

    rows = state.matrix.shape[0]
    columns = state.matrix.shape[1]

    while True:
        # Find an uncovered, available zero
        row, col = np.unravel_index(np.argmax(prime_matrix), (rows, columns))
        logger.debug(f'Uncovered zero at row: {row}, column: {col}')
        
        # If no uncovered zero is found, go to Step 6
        if prime_matrix[row, col] == 0:
            logger.debug(f'Prime Matrix at ({row},{col}):\n{prime_matrix[row, col]}')
            logger.debug(f'No uncovered zero found. Proceeding to Step 6.')
            return step_6_func

        # Mark the row and column as prime (which will be equal to 2 in the final solution matrix)
        state.final_solution[row, col] = 2
        logger.debug(f'Available before marking as unavailable:\n{state.available}')
        state.set_as_unavailable(row, col)
        logger.debug(f'Available after marking as unavailable:\n{state.available}')
        
        # Find the first starred element in the row
        star_col = np.argmax(state.final_solution[row] == 1)
        logger.debug(f'The starred element in the row: {star_col}')
        
        # If no starred element is found, go to Step 5
        if state.final_solution[row, star_col] != 1:
            logger.debug(f'Final Solution matrix at ({row},{star_col}):\n{state.final_solution[row, star_col]}')
            logger.debug(f'No starred element found in the row. Proceeding to Step 5.')
            state.initial_primed_zero_row = row
            state.initial_primed_zero_column = col
            return step_5_func
        
        # Otherwise, cover this row and uncover the column containing the starred zero
        col = star_col
        state.uncolored_rows[row] = False
        state.uncolored_columns[col] = True
        prime_matrix[:, col] = matrix[:, col] * state.uncolored_rows.astype(int) * state.available[:, col].astype(int)
        prime_matrix[row] = 0

def step_5_func(state):
    """
    Construct a series of alternating primed and starred zeros as follows:
    - `Z0`: represent the uncovered primed zero found in Step 4.
    - `Z1`: denote the starred zero in the column of Z0 (if any).
    - `Z2`: denote the primed zero in the row of Z1 (there will always be one).
    - Continue until the series terminates at a primed zero that has no starred zero in its column. 
    - Unstar each starred zero of the series, star each primed zero of the series, erase all primes and uncover every line in the matrix. 
    - Return to Step 3

    Example:
    --------
    >>> matrix = np.array([[0, 2, 0], [3, 0, 1], [1, 0, 0]])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> agent_vector = np.array([1, 1, 1])
    >>> state = ManyToManyAssignment(matrix, task_range_vector, agent_vector)
    >>> state.uncolored_rows = np.array([True, True, True])
    >>> state.uncolored_columns = np.array([True, True, True])
    >>> state.available = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    >>> state.final_solution = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
    >>> state.initial_primed_zero_row = 2
    >>> state.initial_primed_zero_column = 2
    >>> state.path = np.full((100, 2), -1)
    >>> next_step = step_5_func(state)
    >>> state.final_solution
    array([[0, 1, 0],
           [0, 0, 0],
           [0, 0, 1]])
    >>> next_step == step_3_func
    True
    """
    assert isinstance(state, ManyToManyAssignment)
    logger.info(f"------------------Step 5------------------")
    count = 0
    path = state.path
    # Step 5.1: Initialize path with the uncovered primed zero found in Step 4
    path[count, 0] = state.initial_primed_zero_row
    path[count, 1] = state.initial_primed_zero_column
    logger.debug(f'Current path, before entering Step 5 while loop:\n{path}')

    while True:
        logger.debug(f'Step 5 while loop')
        # Step 5.2: Find the first starred zero in the column of the current path element
        row = np.argmax(state.final_solution[:, path[count, 1]] == 1)
        if state.final_solution[row, path[count, 1]] != 1:
            # No starred zero found in the column, end the series.
            break
        else:
            # Add the starred zero to the path
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Step 5.3: Find the first primed zero in the row of the current path element.
        col = np.argmax(state.final_solution[path[count, 0]] == 2)
        if state.final_solution[row, col] != 2:
            # No primed zero found in the row, mark column as -1 (invalid column)
            col = -1
        # Add the primed zero to the path
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col
        logger.debug(f'Current path:\n{path}')

    # Step 5.4: Convert the path, alternating between unstar and star
    for i in range(count + 1):
        if state.final_solution[path[i, 0], path[i, 1]] == 1:
            # Unstar the starred zero
            state.final_solution[path[i, 0], path[i, 1]] = 0
            state.set_as_available(path[i, 0], path[i, 1])
        else:
            # Star the primed zero
            state.find_star_zero(row=path[i, 0], column=path[i, 1])

    # Step 5.5: Uncover all rows and columns
    logger.debug(f'Before uncoloring rows and columns:\nUncolored rows: {state.uncolored_rows}, Uncolored columns: {state.uncolored_columns}')
    state.uncolor_rows_columns()
    logger.debug(f'After uncoloring rows and columns:\nUncolored rows: {state.uncolored_rows}, Uncolored columns: {state.uncolored_columns}')
    # Step 5.6: Erase all primes
    logger.debug(f'Before erasing primes:\n{state.final_solution}')
    state.final_solution[state.final_solution == 2] = 0
    logger.debug(f'After erasing primes:\n{state.final_solution}')
    logger.debug(f'End of Step 5')
    return step_3_func

def step_6_func(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.

    Example:
    --------
    >>> matrix = np.array([[0, 2, 0], [3, 0, 1], [1, 0, 0]])
    >>> task_range_vector = np.array([1, 1, 1])
    >>> agent_vector = np.array([1, 1, 1])
    >>> state = ManyToManyAssignment(matrix, task_range_vector, agent_vector)
    >>> state.uncolored_rows = np.array([True, False, True])
    >>> state.uncolored_columns = np.array([True, False, True])
    >>> state.available = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    >>> next_step = step_6_func(state)
    >>> state.matrix
    array([[0, 2, 0],
           [3, 0, 1],
           [1, 0, 0]])
    >>> next_step == step_4_func
    True
    """
    assert isinstance(state, ManyToManyAssignment)
    logger.info(f"------------------Step 6------------------")
    logger.info(f'Matrix at beginning of Step 6:\n{state.matrix}')
    # Check if there are any uncovered rows and columns
    if np.any(state.uncolored_rows) and np.any(state.uncolored_columns):
        # Copy the current state of the matrix
        temp_matrix = state.matrix.copy()

        # Assign the maximum value in the matrix to elements where the available matrix is 0
        max_value = np.max(temp_matrix)
        temp_matrix[np.where(state.available==0)] = max_value

        # Find the smallest uncovered value in the matrix
        minval = np.min(temp_matrix[state.uncolored_rows], axis=0)
        minval = np.min(minval[state.uncolored_columns])

        # Add the smallest uncovered value to each element of the covered rows
        for i, covered in enumerate(~state.uncolored_rows):
            if covered:
                state.matrix[i] += minval

        # Subtract the smallest uncovered value from each element of the uncovered columns
        state.matrix[:, state.uncolored_columns] -= minval
        logger.debug(f'Matrix at end of Step 6 after updating:\n{state.matrix}')
        # Ensure that there are no negative values in the matrix
        state.matrix[np.where(state.matrix < 0)] = 0

    return step_4_func

class ManyToManyAssignment:
        """
        Class to represent the Many to Many Assignment Problem.
        """
        def __init__(self, matrix: np.asarray, taskRangeVector: np.asarray, agentVector: np.asarray) -> None:
            self.matrix = matrix.copy()
            self.taskRangeVector = taskRangeVector.copy()
            self.agentVector = agentVector.copy()
            self.agents = self.matrix.shape[0]
            self.tasks = self.matrix.shape[1]
            self.preperation_stage()
            self.k = self.matrix.shape[0]
            
            # New matrix with size (kxk)
            self.available = np.ones((self.k,self.k), dtype=bool)
            # The uncovered rows vector
            self.uncolored_rows = np.ones(self.k, dtype=bool)
            # The uncovered columns vector
            self.uncolored_columns = np.ones(self.k, dtype=bool)
            self.final_solution = np.zeros((self.k, self.k), dtype=int)
            # Row index of the initial uncovered primed zero
            self.initial_primed_zero_row = 0
            # Column index of the initial uncovered primed zero
            self.initial_primed_zero_column = 0
            # Path to construct alternating series of primed and starred zeros
            self.path = np.zeros((2 * self.k, 2), dtype=int)

        def uncolor_rows_columns(self) -> None:
            """
            Uncolor the rows and columns of the matrix.
            """
            self.uncolored_rows[:] = True
            self.uncolored_columns[:] = True

        def preperation_stage(self):
            """
            Preperation stage of the Matrix.
            """
            # Check cordinality constraints
            agent_sum = sum(self.agentVector)
            task_sum = sum(self.taskRangeVector)
            if agent_sum < task_sum:
                    warning_message = f"The Cordinality Constraint is not satisfied, with agents summing to {agent_sum} and tasks summing to {task_sum}."
                    logger.error(warning_message)
                    raise ValueError(warning_message)

            if np.any(self.matrix < 0):
                raise ValueError("The performance matrix has values less than 0")
            # Duplicate Columns: Ensures each task can appear multiple times.
            self.matrix = np.repeat(self.matrix, self.taskRangeVector, axis=1)
            # Duplicate Rows: Ensures each agent can handle multiple tasks.
            self.matrix = np.repeat(self.matrix, self.agentVector, axis=0)
            # Create Additional Columns: Ensures the matrix is square and prevents selection of these columns by assigning high costs.
            zero_columns = np.ones((self.matrix.shape[0], self.matrix.shape[0] - self.matrix.shape[1]), dtype=int) * (np.max(self.matrix) * 2)
            # Combine Matrices: Forms the final expanded and square matrix suitable for the assignment algorithm.
            self.matrix = np.hstack((self.matrix, zero_columns))

            self.find_agent_in_row = range(self.tasks)
            self.find_task_in_col = range(self.agents)

            self.find_agent_in_row = np.repeat(self.find_agent_in_row, self.agentVector)
            self.find_task_in_col = np.repeat(self.find_task_in_col, self.taskRangeVector)

            self.task_columns = {}
            self.agent_rows = {}

            self.agent_rows = [np.where(self.find_agent_in_row == i) for i in range(len(self.agentVector))]
            self.task_columns = [np.where(self.find_task_in_col == j) for j in range(len(self.taskRangeVector))]

        def set_as_unavailable(self, row: int, col: int) -> None:
            """
            Mark the cells in the `available` matrix as unavailable, ensuring that the same agent cannot be assigned to the same task more than once.

            Parameters:
            ---------
            - `row` - row of agent
            - `col` - column of task
            """
            if col < len(self.find_task_in_col):
                # Get the agent and task indices
                agent = self.find_agent_in_row[row]
                task = self.find_task_in_col[col]
                # Get the related rows and columns, excluding the current row and columns
                related_rows = self.agent_rows[agent]
                related_columns = self.task_columns[task]
                # Remove the current row and column from the related rows and columns
                related_rows = np.delete(self.agent_rows[agent], np.where(self.agent_rows[agent] == row)[1])
                related_columns = np.delete(self.task_columns[task], np.where(self.task_columns[task] == col)[1])
                # Mark the corresponding cells as unavailable if they are not starred (1 represents a starred zero in the final solution matrix)
                for i in related_rows:
                    for j in related_columns:
                        if self.final_solution[i, j] != 1:
                            self.available[i,j] = False
            
        def set_as_available(self, row: int, col: int) -> None:
            """
            Update the available matrix to mark the cells as available.

            Parameters:
            ---------
            - `row`: The row index of the agent.
            - `col`: The column index of the task.
            """
            if col < len(self.find_task_in_col):
                # Get the agent and task indices
                agent = self.find_agent_in_row[row]
                task = self.find_task_in_col[col]
                # Get the rows and columns related to the agent and task
                related_rows = self.agent_rows[agent]
                related_cols = self.task_columns[task]
                # Mark the corresponding cells as available
                start_row, end_row = related_rows[0][0], related_rows[0][-1]
                start_col, end_col = related_cols[0][0], related_cols[0][-1]
                self.available[start_row: end_row + 1, start_col: end_col + 1] = True

        def find_star_zero(self, row: int, column: int) -> None:
            """
            Finds and marks the zero in the matrix as a 0*.

            Parameters:
            ---------
            - `row`: The row index of the agent.
            - `column`: The column index of the task.
            """
            self.final_solution[row, column] = 1
            self.set_as_unavailable(row, column)

if __name__ == "__main__":
    """
    In order to see all the log messages, change the level to logging.DEBUG.
    In order to skip the debug steps, change the level to logging.INFO.
    In order to skip the info steps, change the level to logging.WARNING.
    """
    # 2. Set the logging level for the logger
    logger.setLevel(logging.WARNING)
    # 3. Create a console handler
    console = logging.StreamHandler()
    # 4. Set the logging level for the console handler
    console.setLevel(logging.WARNING)
    # 5. Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: Line %(lineno)d: %(message)s')
    # 6. Set the formatter for the console handler
    console.setFormatter(formatter)
    # 7. Add the console handler to the logger
    logger.addHandler(console)

    """
    For disabling the doctests, comment out the following line.
    """
    import doctest
    doctest.testmod(verbose=True)

    # Example with agents and tasks capacities more then 1

    # matrix = np.array([[3, 0, 1, 2],[2, 3, 0, 1],[3, 0, 1, 2],[1, 0, 2, 3]])
    # ability_agent_vector = np.array([2,2,2,2])
    # task_range_vector = np.array([2,2,2,2])
    # kuhn_munkers_backtracking(matrix, ability_agent_vector, task_range_vector)
    # {0: [3, 0], 1: [2, 3], 2: [1, 2], 3: [0, 1]}
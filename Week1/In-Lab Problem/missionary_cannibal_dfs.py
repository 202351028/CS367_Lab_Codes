# Check if a state is valid
def is_valid(state):
    missionaries, cannibals, boat = state
    if missionaries < 0 or cannibals < 0 or missionaries > 3 or cannibals > 3:
        return False
    if missionaries > 0 and missionaries < cannibals:
        return False
    if 3 - missionaries > 0 and 3 - missionaries < 3 - cannibals:
        return False
    return True

# Generate all valid next states
def get_successors(state):
    successors = []
    missionaries, cannibals, boat = state
    moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
    if boat == 1:  # Boat on starting side
        for move in moves:
            new_state = (missionaries - move[0], cannibals - move[1], 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:  # Boat on goal side
        for move in moves:
            new_state = (missionaries + move[0], cannibals + move[1], 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

# DFS according to the pseudo code
def dfs(state, goal_state, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    # Step 2: Mark current state as visited
    visited.add(state)
    path.append(state)  # Add current state to path

    # Step 5-6: Check if goal state is reached
    if state == goal_state:
        return path

    # Step 3-8: Explore each valid successor
    for successor in get_successors(state):
        if successor not in visited:
            result = dfs(successor, goal_state, path.copy(), visited.copy())
            if result is not None:  # If solution found, return path
                return result

    # Step 9-10: Backtrack automatically happens due to recursion using path.copy()
    return None  # Step 12: Fail if no solution

# Initial and goal states
start_state = (3, 3, 1)
goal_state = (0, 0, 0)

# Run DFS
solution = dfs(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")

def display_configuration(config):
    """Display the current configuration of the puzzle."""
    return ' '.join('E' if x == 1 else 'W' if x == 0 else '_' for x in config)

def is_solved(config):
    """Check if the puzzle is solved."""
    return config == [0, 0, 0, 2, 1, 1, 1]

def generate_moves(config):
    """Generate all possible moves from the current configuration."""
    possible_moves = []
    for i, rabbit in enumerate(config):
        if rabbit == 1:  # Eastbound rabbit
            if i + 1 < len(config) and config[i + 1] == 2:
                possible_moves.append((i, i + 1))
            if i + 2 < len(config) and config[i + 1] == 0 and config[i + 2] == 2:
                possible_moves.append((i, i + 2))
        elif rabbit == 0:  # Westbound rabbit
            if i - 1 >= 0 and config[i - 1] == 2:
                possible_moves.append((i, i - 1))
            if i - 2 >= 0 and config[i - 1] == 1 and config[i - 2] == 2:
                possible_moves.append((i, i - 2))
    return possible_moves

def apply_move(config, move):
    """Apply a move to the configuration."""
    new_config = config[:]
    new_config[move[1]] = new_config[move[0]]
    new_config[move[0]] = 2
    return new_config

def dfs(config, history=None, seen=None):
    """Solve the Rabbit Hop puzzle using depth-first search."""
    if history is None:
        history = [config]
    if seen is None:
        seen = set()

    if tuple(config) in seen:
        return None
    seen.add(tuple(config))

    if is_solved(config):
        return history

    for move in generate_moves(config):
        next_config = apply_move(config, move)
        solution = dfs(next_config, history + [next_config], seen)
        if solution:
            return solution

    return None

# Initial configuration: 1, 1, 1 (Eastbound), 2 (Empty), 0, 0, 0 (Westbound)
start_config = [1, 1, 1, 2, 0, 0, 0]
solution = dfs(start_config)

if solution:
    print("Solution found:")
    for step, config in enumerate(solution):
        print(f"Step {step}: {display_configuration(config)}")
    print(f"Total moves: {len(solution) - 1}")
else:
    print("No solution exists.")

import random
from typing import List, Dict, Tuple, Set

class KSatSolver:

    def __init__(self, num_variables: int, clause_length: int, num_clauses: int):
        self.var_count = num_variables
        self.k_val = clause_length
        self.num_expressions = num_clauses
        self.problem = self._make_instance()
        self.current_assignment = self._get_initial_state()

    def _make_instance(self) -> List[List[Tuple[int, bool]]]:
        instance_clauses = []
        for _ in range(self.num_expressions):
            clause_literals = []
            vars_used: Set[int] = set()
            while len(vars_used) < self.k_val:
                # Variable indices are 1 to var_count, inclusive
                v_idx = random.randint(1, self.var_count)
                if v_idx not in vars_used:
                    required_value = random.choice([True, False])
                    clause_literals.append((v_idx, required_value))
                    vars_used.add(v_idx)
            instance_clauses.append(clause_literals)
        return instance_clauses

    def _get_initial_state(self) -> Dict[int, bool]:
        return {v_idx: random.choice([True, False]) for v_idx in range(1, self.var_count + 1)}

    def calculate_satisfied_count(self, assignment: Dict[int, bool]) -> int:
        satisfied_tally = 0
        for clause in self.problem:
            # A clause is satisfied if at least one literal is True
            if any(assignment[v] == polarity for v, polarity in clause):
                satisfied_tally += 1
        return satisfied_tally

    def perturb_state(self, current_assignment: Dict[int, bool], flip_count: int) -> Dict[int, bool]:
        next_assignment = current_assignment.copy()
        
        # Ensure we don't try to flip more variables than exist
        num_to_flip = min(flip_count, self.var_count)
        
        # Select variables randomly without replacement
        indices_to_change = random.sample(list(next_assignment.keys()), num_to_flip)
        
        for v_idx in indices_to_change:
            next_assignment[v_idx] = not next_assignment[v_idx]
            
        return next_assignment

    def variable_neighborhood_search(self, max_search_cycles: int = 1000) -> Dict[int, bool]:
        best_solution_found = self.current_assignment
        best_score_achieved = self.calculate_satisfied_count(best_solution_found)

        # The neighborhood size (k) ranges from 1 up to half the total variables
        max_k_neighborhood = self.var_count // 2

        for _ in range(max_search_cycles):
            # Neighborhood exploration (Shaking step)
            for k in range(1, max_k_neighborhood + 1):
                # Generate a candidate solution from the current best by flipping k variables
                candidate_solution = self.perturb_state(best_solution_found, k)
                candidate_score = self.calculate_satisfied_count(candidate_solution)

                # Local search (Improvement step) - only accepting a better score
                if candidate_score > best_score_achieved:
                    best_solution_found = candidate_solution
                    best_score_achieved = candidate_score

                # Termination condition: All clauses are satisfied
                if best_score_achieved == self.num_expressions:
                    return best_solution_found

        return best_solution_found

    def solve(self) -> Tuple[Dict[int, bool], int]:
        final_assignment = self.variable_neighborhood_search()
        final_score = self.calculate_satisfied_count(final_assignment)
        return final_assignment, final_score

# Example usage
if __name__ == "__main__":
    # The original example usage block is maintained for consistency
    solver = KSatSolver(num_variables=5, clause_length=3, num_clauses=10)
    solution, score = solver.solve()
    print(f"Best solution: {solution}")
    print(f"Satisfied clauses: {score} out of {solver.num_expressions}")
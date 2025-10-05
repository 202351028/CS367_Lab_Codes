import random
from typing import List, Tuple, Dict, Any

class KSatProblemGenerator:

    def __init__(self, k: int, m: int, n: int):
        if k > n:
            raise ValueError("Clause length (k) cannot be greater than the number of variables (n).")
            
        self.k = k
        self.m = m
        self.n = n
        self.problem = self._generate_problem()
        
    def _generate_problem(self) -> List[List[Tuple[int, bool]]]:
        problem = []
        # List of all variable indices from 1 to n
        all_variables = list(range(1, self.n + 1))
        
        for _ in range(self.m):
            clause = []
            
            # 1. Select k DISTINCT variables (indices) using random.sample
            vars_in_clause = random.sample(all_variables, self.k)
            
            # 2. Assign a random polarity (True/False) to each selected variable
            for var in vars_in_clause:
                # True means positive literal (x_i), False means negative (~x_i)
                polarity = random.choice([True, False])
                clause.append((var, polarity))
                
            problem.append(clause)
            
        return problem

    def get_problem(self) -> List[List[Tuple[int, bool]]]:
        return self.problem

    def display_problem(self):
        print(f"--- Uniform Random {self.k}-SAT Problem ---")
        print(f"Variables (n): {self.n}, Clauses (m): {self.m}, Clause Length (k): {self.k}\n")
        
        for i, clause in enumerate(self.problem):
            readable_clause = []
            for var, polarity in clause:
                literal = f"x{var}" if polarity else f"~x{var}"
                readable_clause.append(literal)
            print(f"Clause {i+1}: ({' OR '.join(readable_clause)})")
            
        print("\nEnd")

# Define the parameters
K = 3   # Clause length (k)
M = 15  # Number of clauses (m)
N = 10  # Number of variables (n)

if __name__ == "__main__":
    try:
        # Generate the problem instance
        generator = KSatProblemGenerator(k=K, m=M, n=N)
        
        # Display the problem
        generator.display_problem()

        # Access the raw formula (list of lists of tuples)
        raw_problem = generator.get_problem()
        print("\nRaw formula (First clause):", raw_problem[0])

    except ValueError as e:
        print(f"Error: {e}")

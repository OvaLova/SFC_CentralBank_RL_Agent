from sympy import nsolve, lambdify
from scipy.optimize import root
import numpy as np
from equation_parser import print_equations


def substitute_knowns(eq, knowns, endogenous_vars):
    safe_knowns = {k: v for k, v in knowns.items() if k not in endogenous_vars}
    eq_subs = eq.subs(safe_knowns)
    return eq_subs

def solve_period(equations, knowns, guess):
    # Variables to solve for
    vars_to_solve = [eq.lhs for eq in equations]
    var_names = [str(v) for v in vars_to_solve]
    
    # Prepare substituted RHS expressions
    endogenous_var_names = set(var_names)
    substituted_eqs = [substitute_knowns(eq, knowns, endogenous_var_names) for eq in equations]

    # Create a numerical function from symbolic equations
    f = lambdify(vars_to_solve, [eq.lhs - eq.rhs for eq in substituted_eqs], modules='numpy')

    # Define a wrapper to match scipy's format
    def func_to_minimize(x):
        return np.array(f(*x), dtype=np.float64)

    # Initial guess as numpy array
    guess_array = np.array(guess, dtype=np.float64)

    # Call solver
    result = root(func_to_minimize, guess_array, method='hybr', tol=1e-8)

    if not result.success:
        raise ValueError(f"Solver failed: {result.message}")

    return dict(zip(var_names, result.x))

def initialize_guess(state, equations):
    # Create mapping from base names to most recent available values
    value_map = {}
    
    # Process state dictionary to handle time subscripts
    for key, value in state.items():
        if "_-1" in key:
            base_name = key.split('_-1')[0]  # Strip any _-1, _-2 etc.
        elif "-1" in key:
            base_name = key.split('-1')[0]
        else:
            base_name = key
        
        # Keep the most recent value available for each base name
        if base_name not in value_map:
            value_map[base_name] = value
    
    # Get all endogenous variables (LHS of equations)
    endogenous_vars = [str(eq.lhs) for eq in equations]
    
    # Build initial guess list
    initial_guess = []
    for var in endogenous_vars:        
        if var in value_map:
            initial_guess.append(float(value_map[var]))
        else:
            # Default to 1 if no value found (shouldn't happen for proper models)
            # print(f"Warning: No initial value found for {var}, defaulting to 1")
            initial_guess.append(1.0)
    
    return initial_guess


from sympy import nsolve


def substitute_knowns(eq, knowns, endogenous_vars):
    # Only substitute values for knowns that are NOT endogenous
    safe_knowns = {k: v for k, v in knowns.items() if k not in endogenous_vars}
    return eq.subs(safe_knowns)

def solve_period(equations, knowns, guess):
    # Substitute lags and exogenous into RHS
    vars_to_solve = [eq.lhs for eq in equations]
    endogenous_var_names = {str(v) for v in vars_to_solve}
    substituted_eqs = [substitute_knowns(eq, knowns, endogenous_var_names) for eq in equations]
    
    # Solve numerically
    sol = nsolve(substituted_eqs, vars_to_solve, guess)
    
    return dict(zip([str(v) for v in vars_to_solve], sol))

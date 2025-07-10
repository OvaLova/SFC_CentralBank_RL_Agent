from sympy import nsolve


def substitute_knowns(eq, knowns):
    return eq.subs(knowns)

def solve_period(equations, knowns, guess):
    # Substitute lags and exogenous into RHS
    substituted_eqs = [substitute_knowns(eq, knowns) for eq in equations]
    vars_to_solve = [eq.lhs for eq in equations]
    
    # Solve numerically
    sol = nsolve(substituted_eqs, vars_to_solve, guess)
    
    return dict(zip([str(v) for v in vars_to_solve], sol))

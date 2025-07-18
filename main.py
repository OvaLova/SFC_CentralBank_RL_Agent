from equation_parser import parse_equations, print_variabels, print_equations
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from solve_model import solve_period, initialize_guess
import pandas as pd
import matplotlib.pyplot as plt
from model_matrices import balance_sheet_map, state, build_matrix, check_matrix_consistency


if __name__ == "__main__":
    with open("equations_zezza.txt") as f:
        lines = f.readlines()

    deps, eqs = parse_equations(lines)

    G = build_dependency_graph(deps)
    cond_graph, sccs = build_condensation_graph(G)
    visualize_dependency_graph(G, sccs, title="Variable Dependency Graph", filename="dependency_graph.pdf")
    visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf")

    ### Prints ###
    print_equations(eqs)
    print_variabels(eqs)

    T = 100
    history = pd.DataFrame()
    # Set initial values (includes lags and exogenous vars)
    state = state
    # Initial guess for nsolve
    initial_guess = initialize_guess(state, eqs)

    for t in range(T):
        balance_sheet = build_matrix(balance_sheet_map, state)
        print(balance_sheet)
        check_matrix_consistency(balance_sheet)
        # print(state)
        print(f"\n-> Solving for timestep {t}...")
        # Solve equations at time t
        solution = solve_period(eqs, state, initial_guess)
        print("Done!")
        # Store solution
        row = {"t": t}
        row.update(solution)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        # Update state: copy solution and prepare lagged versions
        state.update(solution)
        for var in solution:
            lagged_key = f"{var}_-1"
            if lagged_key in state:
                state[lagged_key] = solution[var]
            else:
                lagged_key = f"{var}-1"
                if lagged_key in state:
                    state[lagged_key] = solution[var]

        # Update guess for next period
        initial_guess = list(solution.values())

    # Save history to CSV
    history.to_csv("history.csv", index=False)


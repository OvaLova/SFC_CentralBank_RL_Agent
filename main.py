from equation_parser import parse_equations, initialize_exogenous
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from solve_model import solve_period
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("equations.txt") as f:
        lines = f.readlines()

    deps, eqs = parse_equations(lines)

    # G = build_dependency_graph(deps)
    # cond_graph, sccs = build_condensation_graph(G)
    # visualize_dependency_graph(G, sccs, title="Variable Dependency Graph", filename="dependency_graph.pdf")
    # visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf")

    ### Prints ###
    # print("Strongly Connected Components:")
    # for i, scc in enumerate(sccs):
    #     print(f"SCC {i}: {scc}")
    #
    # print("\nCondensation Graph Nodes:")
    # print(cond_graph.nodes())
    #
    # for i, eq in enumerate(eqs):
    #     print(f'{i+1} -> {eq}')

    T = 0
    history = pd.DataFrame()
    # Set initial values (includes lags and exogenous vars)
    state = initialize_exogenous(eqs, {}, None)
    # Initial guess for nsolve
    initial_guess = [1] * len(eqs)

    for t in range(T):
        print(f"-> Solving for timestep {t}...")
        # Solve equations at time t
        while True:
            try:
                solution = solve_period(eqs, state, initial_guess)
                break
            except ValueError as e:
                print(e)
                if t == 0:
                    state = initialize_exogenous(eqs, {}, None)
                else:
                    new_state = initialize_exogenous(eqs, {}, None)
                    for k, v in new_state.items():
                        if "_-1" not in k:  # Don't overwrite lags
                            state[k] = v
                print(f"-> Solving for timestep {t} again...")
                continue
        # Store solution
        row = {"t": t}
        row.update(solution)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        # Update state: copy solution and prepare lagged versions
        state.update(solution)
        for var in solution:
            state[f"{var}_-1"] = solution[var]
        # Update guess for next period
        initial_guess = list(solution.values())

    # Save history to CSV
    history.to_csv("history.csv", index=False)

    plt.plot(history["t"], history["y"], marker='o')
    plt.title("Real output over time")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

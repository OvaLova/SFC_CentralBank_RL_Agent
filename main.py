from equation_parser import parse_equations
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from solve_model import solve_period, substitute_knowns
import pandas as pd
import matplotlib.pyplot as plt
from sympy import solve, symbols, Eq

if __name__ == "__main__":
    with open("equations.txt") as f:
        lines = f.readlines()

    deps, eqs = parse_equations(lines)
    G = build_dependency_graph(deps)
    cond_graph, sccs = build_condensation_graph(G)

    visualize_dependency_graph(G, sccs, title="Variable Dependency Graph", filename="dependency_graph.pdf")
    visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf")

    # print("Strongly Connected Components:")
    # for i, scc in enumerate(sccs):
    #     print(f"SCC {i}: {scc}")
    #
    # print("\nCondensation Graph Nodes:")
    # print(cond_graph.nodes())
    #
    for i, eq in enumerate(eqs):
        print(f'{i+1} -> {eq}')
        if i+1 == 63:
            M_d = symbols("M_d")
            eqs[i] = Eq(M_d, solve(eq, M_d)[0])
            print(f'{i+1} -> {eq}')

    T = 20
    history = pd.DataFrame()
    # Set initial values (includes lags and exogenous vars)
    state = {
        "a": 10,
        "b": 100,
        "c": 50,
    }
    # Initial guess for nsolve
    initial_guess = [1] * len(eqs)

    for t in range(T):
        # Solve equations at time t
        solution = solve_period(eqs, state, initial_guess)
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

    plt.plot(history["t"], history["y"], marker='o')
    plt.title("Real output over time")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

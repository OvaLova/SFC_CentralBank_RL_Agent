from equation_parser import parse_equations, print_variabels, print_equations
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from solve_model import solve_period, initialize_guess
import pandas as pd
import matplotlib.pyplot as plt
from model_matrices import balance_sheet_map, state, build_matrix, check_matrix_consistency
import random as rd
from gym_environment import pi_target, threshold
import numpy as np


def _apply_shocks():
    # Apply random shocks to non-lagged variables before solving
    shockable_vars = [var for var in state 
                    if not var.endswith('_-1') and not var.endswith('-1') 
                    and var != 'r_b_' and not var.startswith('λ')]  
    shockable_vars = rd.sample(shockable_vars, 5)
    for var in shockable_vars:
        shock = np.random.normal(scale=0.01)
        print(f"the shock to {var} is: {shock*100}%")
        shocked_value = state[var] * (1 + shock)
        state[var] = np.clip(shocked_value, 0, 1)


if __name__ == "__main__":
    with open("equations_zezza.txt") as f:
        lines = f.readlines()

    deps, eqs = parse_equations(lines)

    # G = build_dependency_graph(deps)
    # cond_graph, sccs = build_condensation_graph(G)
    # visualize_dependency_graph(G, sccs, title="Variable Dependency Graph", filename="dependency_graph.pdf")
    # visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf")

    ### Prints ###
    # print_equations(eqs)
    # print_variabels(eqs)

    T = 200
    history = pd.DataFrame()
    # Set initial values (includes lags and exogenous vars)
    state = state
    print(f'Initial interest rate is: {state["r_b_"]}')
    # Initial guess for nsolve
    initial_guess = initialize_guess(state, eqs)

    for t in range(T):
        # balance_sheet = build_matrix(balance_sheet_map, state)
        # print(balance_sheet)
        # check_matrix_consistency(balance_sheet, verbose=True)
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
        _apply_shocks()
        for var in solution:
            lagged_key = f"{var}_-1"
            if lagged_key in state:
                state[lagged_key] = solution[var]
            else:
                lagged_key = f"{var}-1"
                if lagged_key in state:
                    state[lagged_key] = solution[var]
        # Policy shocks
        # if t == 50:
        #     new_rate = state["r_b_"] + 0.01     # increase
        #     # new_rate = state["r_b_"] - 0.01     # decrease
        #     state["r_b_"] = new_rate
        # Policy simulations
        # new_rate = min(0.08, state["r_b_"] + 0.005)     # increase max
        # new_rate = max(0.0, state["r_b_"] - 0.005)    # decrease max
        new_rate = rd.choice([state["r_b_"] + 0.005, state["r_b_"] - 0.005])    # random walk
        new_rate = np.clip(new_rate, 0.0, 0.15)
        state["r_b_"] = new_rate
        print(f'Agent selected r_b_ = {state["r_b_"]}')

        # Update guess for next period
        initial_guess = list(solution.values())

    # Save history to CSV
    history.to_csv("history.csv", index=False)

    plt.figure(figsize=(12, 9))
    history = pd.read_csv("history.csv")
    
    # Plot inflation and target
    plt.subplot(3, 1, 1)
    plt.plot(history['t'], history['π'], label='Actual Inflation', color='blue')
    plt.axhline(y=pi_target, color='r', linestyle='--', label='Target Inflation')
    plt.fill_between(history['t'], 
                    pi_target + threshold, 
                    pi_target - threshold, 
                    color='green', alpha=0.1, label=f'Target Zone (±{threshold:.3%})')
    plt.ylabel('Inflation Rate')
    plt.legend()
    plt.grid(True)

    # Plot interest rates
    plt.subplot(3, 1, 2)
    plt.plot(history['t'], history['r_b'], label='Policy Rate', color='orange')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)

    # Plot GDP Growth
    plt.subplot(3, 1, 3)
    gdp_growth = np.diff(history['Y']) / history['Y'][:-1] * 100
    plt.plot(history['t'][1:], gdp_growth, label='Growth Rate(%)', color='tab:purple')
    plt.ylabel('GDP')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    
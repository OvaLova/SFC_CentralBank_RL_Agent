from equation_parser import parse_equations, print_variabels, print_equations
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from solve_model import solve_period, initialize_guess
import pandas as pd
import matplotlib.pyplot as plt
from model_matrices import balance_sheet_map, state, build_matrix, check_matrix_consistency
import random as rd
from gym_environment import pi_target, threshold_pi
import numpy as np


def _apply_shocks():
    # Apply random shocks to non-lagged variables before solving
    shockable_vars = [var for var in state 
                    if not var.endswith('-1') 
                    and var != 'r_b_' and not var.startswith('λ')] # could shock lambdas also tho
    # shockable_vars = ["Ω_0", "gr_g", "θ", "α_1", "α_2"] # variables shocked in the book
    shockable_vars = rd.sample(shockable_vars, 5)
    for var in shockable_vars:
        shock = np.clip(np.random.normal(scale=0.02), -0.1, 0.1)
        print(f"the shock to {var} is: {shock*100}%")
        sign = np.sign(state[var])
        mag  = max(abs(state[var]), 1e-4)  # avoid zero trap
        shocked_value = mag * (1 + shock)
        shocked_value = np.clip(shocked_value, 0.0, 1.0)
        state[var] = sign * shocked_value


# Testing the model under different policies and conditions => trajectories
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

    T = 100
    history = pd.DataFrame()
    # Set initial values (includes lags and exogenous vars)
    init_state = state.copy()
    state = state
    print(f'Initial interest rate is: {state["r_b_"]}')
    # Initial guess for nsolve
    initial_guess = initialize_guess(state, eqs)

    # for t in range(T):
    #     timestep = t+1
    #     balance_sheet = build_matrix(balance_sheet_map, state)
    #     print(balance_sheet)
    #     check_matrix_consistency(balance_sheet, verbose=True)
    #     # print(state)
    #     print(f"\n-> Solving for timestep {timestep}...")
    #     # Solve equations at time t
    #     solution = solve_period(eqs, state, initial_guess)
    #     print("Done!")

    #     # Store solution
    #     row = {"t": timestep}
    #     row.update(solution)
    #     history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    #     # Update state: copy solution and prepare lagged versions
    #     # _apply_shocks()
    #     for var in solution:
    #         lagged_key = f"{var}_-1"
    #         if lagged_key in state:
    #             state[lagged_key] = solution[var]
    #         else:
    #             lagged_key = f"{var}-1"
    #             if lagged_key in state:
    #                 state[lagged_key] = solution[var]

    #     ### Policy one-time shocks
    #     if timestep == 1:
    #         # new_rate = state["r_b_"] + 0.01     # increase
    #         new_rate = state["r_b_"] - 0.0075     # decrease
    #         state["r_b_"] = new_rate

    #     ### Policy simulations
    #     # Only increase/decrease
    #     # new_rate = min(0.08, state["r_b_"] + 0.005)     # only increase 
    #     # new_rate = max(0.0, state["r_b_"] - 0.005)    # only decrease 
    #     # Random walk
    #     # new_rate = rd.choice([state["r_b_"] + 0.005, state["r_b_"] - 0.005])    # random walk
    #     # new_rate = np.clip(new_rate, 0.0, 0.15)
    #     # state["r_b_"] = new_rate

    #     # print(f'Agent selected r_b_ = {state["r_b_"]}')

    #     # Update guess for next period
    #     initial_guess = list(solution.values())

    # # Save history to CSV
    # history.to_csv("history.csv", index=False)


# Plotting
history = pd.read_csv(f"history.csv")
plt.figure(figsize=(8, 8))
plt.suptitle(f"Model: Drop", fontsize=12)

# Plot inflation and target
plt.subplot(4, 1, 1)
plt.plot(history['t'], history['π'], label='Inflation', color='blue')
plt.axhline(y=pi_target, color='r', linestyle='--', label='Target Inflation')
plt.fill_between(history['t'], 
                 pi_target - threshold_pi, 
                 pi_target + threshold_pi, 
                 color='green', alpha=0.1, label=f'Target Zone (±{threshold_pi:.3%})')
plt.ylabel('Inflation Rate')
plt.legend(loc='lower right')
plt.grid(True)

# Plot growth
plt.subplot(4, 1, 2)
gdp = np.array(history['y'])
initial_gdp = init_state['y_-1']
gdp_prev = np.concatenate(([initial_gdp], gdp[:-1]))
growth = list(1 - gdp_prev/gdp)
plt.plot(history['t'], growth, label='Growth', color='purple')
plt.ylabel('Real GDP Growth')
plt.legend()
plt.grid(True)

# Plot GDP
plt.subplot(4, 1, 3)
plt.plot(history['t'], history['y'], label='GDP', color='pink')
plt.ylabel('Real GDP')
plt.legend()
plt.grid(True)

# Plot interest rates
plt.subplot(4, 1, 4)
plt.plot(history['t'], history['r_b'], label='Policy Rate', color='red')
plt.ylabel('Interest Rate for Bills')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
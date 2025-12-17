import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random as rd
import pandas as pd
from solve_model import solve_period, initialize_guess
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from model_matrices import build_matrix, check_matrix_consistency
from equation_parser import parse_equations, print_variabels, print_equations


# Policy targets
pi_target = 0.0075  
# u_target = 1   
# Weights for penalty (λ values)
lambda_pi = 1.0
# lambda_u = 0.5
# lambda_y = 0.25
lambda_vol = 1.0
# Allowed deviation
threshold_pi = 0.0025
# threshold_u = 0.05


class SFCEnv(gym.Env):
    def __init__(self, eq_file="equations.txt", balance_sheet_map=None, init_state=None, T=100, verbose=False, loss="hinge"):
        super().__init__()
        self.verbose = verbose
        self.T = T
        self.t = 0
        self.balance_sheet_map = balance_sheet_map
        self.loss = loss

        # Load and parse equations
        with open(eq_file) as f:
            lines = f.readlines()
        self.deps, self.eqs = parse_equations(lines)

        # G = build_dependency_graph(self.deps)
        # cond_graph, sccs = build_condensation_graph(G)
        # visualize_dependency_graph(G, sccs, title="Variable Dependency Graph", filename="dependency_graph.pdf")
        # visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf")

        ### Prints ###
        if self.verbose:
            print_equations(self.eqs)
            print_variabels(self.eqs)

        # State and guess
        self.init_state = init_state.copy()
        self.state = self.init_state.copy()
        self.initial_guess = initialize_guess(self.state, self.eqs)

        # Define action and observation space
        self.action_vars = ["r_b_"]  
        action_deltas = [-0.01, -0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075, 0.01]
        self.action_value_ranges = {
            "r_b_": action_deltas
        }
        self.action_space = spaces.MultiDiscrete([len(self.action_value_ranges[var]) for var in self.action_vars])
        all_vars = list(self.state.keys())
        vars_to_exculde = [var for var in self.state 
                        if not var.endswith('_-1') and not var.endswith('-1')    # exclude parameters
                        and '^T' not in var    # exclude targets
                        and '^e' not in var    # exclude expectations
                        ]  
        self.observation_vars = [var for var in all_vars if var not in vars_to_exculde]
        self.observation_space = spaces.Box(
            low=-1e2, high=1e12, shape=(len(self.observation_vars),), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.verbose:
            print("============= Model is being reset... =============")
        self.t = 0
        self.state = self.init_state.copy()
        self.initial_guess = initialize_guess(self.state, self.eqs)
        return self._get_obs(), {}

    def step(self, action):
        balance_sheet = build_matrix(self.balance_sheet_map, self.state)
        passed, differences = check_matrix_consistency(balance_sheet, verbose=self.verbose)
        if self.verbose:
            print(balance_sheet)
            if passed:
                print("=== PASSED ===")
            else:
                print("=== FAILED ===")

        # Apply random shocks to non-lagged variables before solving (introduce stochasticity)
        # self._apply_shocks()

        # Solve model
        if self.verbose:
            print(f"\n-> Solving for timestep {self.t}...")
        try:
            solution = solve_period(self.eqs, self.state, self.initial_guess)
        except Exception as e:
            if self.verbose:
                print("Solve failed:", e)
            truncated = True
            terminated = False  # No terminal state
            return self._get_obs(), -1e5, terminated, truncated, {}

        # Update state
        prev_rate = self.state["r_b_"]
        prev_y = self.state["y_-1"]
        for var in solution:
            lagged_key = f"{var}_-1"
            if lagged_key in self.state:
                self.state[lagged_key] = solution[var]
            else:
                lagged_key = f"{var}-1"
                if lagged_key in self.state:
                    self.state[lagged_key] = solution[var]

        # Apply agent actions to control variables
        for i, var in enumerate(self.action_vars):
            delta = self.action_value_ranges[var][action[i]]
            prev_value = self.state[var]
            new_value = round(prev_value + delta, 5)
            new_value = np.clip(new_value, 0.0, 0.15)
            self.state[var] = new_value
            if self.verbose:
                print(f"Agent selected {var}: {new_value:.3%} (Δ {delta:+.3%})")

        # Store result
        row = {"t": self.t}
        row.update(solution)

        # Update guess
        self.initial_guess = list(solution.values())
        self.t += 1

        # Define reward (e.g., stable inflation or GDP growth)
        reward = self._compute_reward(prev_rate, prev_y)

        truncated = self.t >= self.T
        terminated = False  # No terminal state
        info = {
            "π": self.state["π_-1"],
            "u": self.state["u_-1"],
            "r_b_": self.state["r_b_"],
            "gdp_growth": self.state.get("y_-1", 0.0) / prev_y
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([self.state[k] for k in self.observation_vars], dtype=np.float32)

    def _compute_reward(self, prev_rate, prev_y):
        # Extract key indicators from state
        inflation = self.state.get("π_-1", 0.0)
        capacity_util = self.state.get("u_-1", 0.0)
        gdp_growth = self.state.get("y_-1", 0.0) / prev_y

        if self.loss == "quadratic":
            # Quadratic penalty for deviation (penalizes any deviation)
            penalty = lambda_pi * (inflation - pi_target)**2 
            # penalty = penalty + lambda_u * (capacity_util - u_target)**2
            # penalty = penalty - lambda_y * gdp_growth
        elif self.loss == "hinge":
            # Hinge penalty for deviation (penalizes only beyond threshold)
            inflation_loss = max(0, abs(inflation - pi_target) - threshold_pi)
            # util_loss = max(0, abs(capacity_util - u_target) - threshold_u)
            penalty = lambda_pi * inflation_loss 
            # penalty = penalty + lambda_u * util_loss    
            # penalty = penalty - lambda_y * gdp_growth
        elif self.loss == "piecewise":
            # Hinge penalty for deviation (rewards for target zone, else penalizes)
            if (abs(inflation - pi_target) < threshold_pi 
                # and abs(capacity_util - u_target) < threshold_u
                ):
                penalty = -1.0   # Bonus for being in target zone
                # - lambda_y * gdp_growth
            else:
                penalty = lambda_pi * (inflation - pi_target)**2 
                # + lambda_u * (capacity_util - u_target)**2
                # - lambda_y * gdp_growth

        # Penalize large rate changes (smoothing)
        new_rate = self.state["r_b_"]
        rate_change_penalty = lambda_vol * (new_rate - prev_rate) ** 2
        penalty = penalty + rate_change_penalty
        # Negative of penalty is reward
        reward = -penalty
        if self.verbose:
            print(f"The reward is: {reward}")
        return reward
    
    def _apply_shocks(self):
        # Apply random shocks to non-lagged variables before solving
        shockable_vars = [var for var in self.state 
                     if not var.endswith('_-1') and not var.endswith('-1') 
                     and var != 'r_b_' and not var.startswith('λ')] # could shock lambdas also tho
        # shockable_vars = ["Ω_0", "gr_g", "θ", "α_1", "α_2"] # variables shocked in the book
        shockable_vars = rd.sample(shockable_vars, 5)
        for var in shockable_vars:
            shock = np.clip(np.random.normal(scale=0.02), -0.1, 0.1)
            if self.verbose:
                print(f"the shock to {var} is: {shock*100}%")
            sign = np.sign(self.state[var])
            mag  = max(abs(self.state[var]), 1e-4)  # avoid zero trap
            shocked_value = mag * (1 + shock)
            shocked_value = np.clip(shocked_value, 0.0, 1.0)
            self.state[var] = sign * shocked_value

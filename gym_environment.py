import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from solve_model import solve_period, initialize_guess
from graph_builder import build_dependency_graph, build_condensation_graph, visualize_dependency_graph, visualize_condensation_graph
from model_matrices import build_matrix, check_matrix_consistency
from equation_parser import parse_equations


class SFCEnv(gym.Env):
    def __init__(self, eq_file="equations.txt", balance_sheet_map=None, init_state=None, T=100):
        super().__init__()
        self.T = T
        self.t = 0
        self.balance_sheet_map = balance_sheet_map
        self.history = pd.DataFrame()

        # Load and parse equations
        with open(eq_file) as f:
            lines = f.readlines()
        self.deps, self.eqs = parse_equations(lines)

        # G = build_dependency_graph(self.deps)
        # cond_graph, sccs = build_condensation_graph(G)
        # visualize_dependency_graph(G, sccs, title="Variable Dependency Graph", filename="dependency_graph.pdf")
        # visualize_condensation_graph(cond_graph, sccs, title="Condensation Graph", filename="condensation_graph.pdf")

        ### Prints ###
        # print_equations(eqs)
        # print_variabels(eqs)

        # State and guess
        self.init_state = init_state.copy()
        self.state = self.init_state.copy()
        self.initial_guess = initialize_guess(self.state, self.eqs)

        # Define action and observation space
        self.action_vars = ["r_b_"]  
        self.action_deltas = [-0.005, 0.0, 0.005]
        self.action_value_ranges = {
            "r_b_": self.action_deltas
        }
        self.action_space = spaces.MultiDiscrete([len(self.action_value_ranges[var]) for var in self.action_vars])
        self.observation_vars = ["r_b_", "Y_-1", "u_-1", "π_-1", "WB_-1", "N_-1", "I_-1", "K_-1", "C_-1", "r_l-1", "r_m-1", "B_s-1", "G_-1", "V_-1", "GD_-1", "NPL_-1"] 
        self.observation_space = spaces.Box(
            low=-1e4, high=1e6, shape=(len(self.observation_vars),), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("============= Model is being reset... =============")
        self.history.to_csv("history.csv", index=False)
        self.t = 0
        self.history = pd.DataFrame()
        self.state = self.init_state.copy()
        self.initial_guess = initialize_guess(self.state, self.eqs)
        return self._get_obs(), {}

    def step(self, action):
        # balance_sheet = build_matrix(self.balance_sheet_map, self.state)
        # print(balance_sheet)
        # check_matrix_consistency(balance_sheet)
        # print(self.state)

        # Solve model
        print(f"\n-> Solving for timestep {self.t}...")
        try:
            solution = solve_period(self.eqs, self.state, self.initial_guess)
        except Exception as e:
            print("Solve failed:", e)
            return self._get_obs(), -1e2, True, False, {}

        # Update state
        self.state.update(solution)
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
            new_value = np.clip(new_value, 0.0, 0.3)
            self.state[var] = new_value
            print(f"Agent selected {var}: {new_value:.3%} (Δ {delta:+.3%})")

        # Store result
        row = {"t": self.t}
        row.update(solution)
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)

        # Update guess
        self.initial_guess = list(solution.values())
        self.t += 1

        # Define reward (e.g., stable inflation or GDP growth)
        reward = self._compute_reward()

        truncated = self.t >= self.T
        terminated = False  # No terminal state
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.state[k] / 1e6 for k in self.observation_vars], dtype=np.float32)

    def _compute_reward(self):
        # Extract key indicators from state
        inflation = self.state.get("π_-1", 0.0)
        capacity_util = self.state.get("u_-1", 0.0)

        # Set targets
        pi_target = 0.02    # 2% inflation target
        u_target = 0.85     # e.g., 85% utilization as full employment proxy

        # Set weights for penalty (λ values)
        lambda_pi = 1.0
        lambda_u = 1.0

        # Quadratic penalty for deviation
        penalty = lambda_pi * (inflation - pi_target)**2 + lambda_u * (capacity_util - u_target)**2

        # Negative of penalty is reward
        reward = -penalty
        print(f"The reward is: {reward}")
        return reward

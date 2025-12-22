import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_environment import SFCEnv, pi_target, threshold_pi
from train_agent import model_name, loss, T
from model_matrices import balance_sheet_map, state 
import pandas as pd
import numpy as np


print(f"========== SIMULATION of: {model_name} ==========")
# Initialize environment (single instance for simulation)
base_env = SFCEnv(
    T=T,
    eq_file="equations_zezza.txt",
    init_state=state,
    balance_sheet_map=balance_sheet_map,
    verbose=True,
    loss=loss
)
env = DummyVecEnv([lambda: base_env])
env = VecNormalize.load(f"{model_name}_vecnormalize.pkl", env)
env.training = False  # Disable stats updates
env.norm_reward = False  # Don't normalize rewards for eval

# Load model with normalized env
model = PPO.load(model_name)

# Storage for analysis
history = pd.DataFrame()

# Run simulation
obs = env.reset()
for t in range(base_env.T):
    timestep = t+1
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, infos = env.step(action)
    obs = env.unnormalize_obs(obs)
    
    # Convert action to integer index
    action_idx = int(action[0])  # Convert numpy array to integer
    action_value = base_env.action_value_ranges["r_b_"][action_idx]

    # Store solution
    if timestep != base_env.T:
        row = {"t": timestep}
        row.update(dict(zip(base_env.observation_vars, obs.flatten())))
        row.update({"action": action_value})
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

# Save history to CSV
history.to_csv(f"history_{model_name}.csv", index=False)

# Plotting
history = pd.read_csv(f"history_{model_name}.csv")
plt.figure(figsize=(8, 8))
plt.suptitle(f"Model: {model_name}", fontsize=12)

# Plot inflation and target
plt.subplot(4, 1, 1)
plt.plot(history['t'], history['π_-1'], label='Inflation', color='blue')
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
gdp = np.array(history['y_-1'])
initial_gdp = base_env.init_state['y_-1']
gdp_prev = np.concatenate(([initial_gdp], gdp[:-1]))
growth = list(1 - gdp_prev/gdp)
plt.plot(history['t'], growth, label='Growth', color='purple')
plt.ylabel('Real GDP Growth')
plt.legend()
plt.grid(True)

# Plot interest rates
plt.subplot(4, 1, 3)
plt.plot(history['t'], history['r_b-1'], label='Policy Rate', color='red')
plt.ylabel('Interest Rate for Bills')
plt.legend()
plt.grid(True)

# Plot policy actions
plt.subplot(4, 1, 4)
plt.step(history['t'], history['action'], label='Policy Action', color='orange', where='post')
plt.scatter(history['t'], history['action'], 
            color='darkorange', 
            zorder=3,
            s=15)     
plt.ylabel('Policy Action')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
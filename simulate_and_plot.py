import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_environment import SFCEnv, pi_target, threshold
from train_agent import model_name, loss
from model_matrices import balance_sheet_map, state 


print(f"========== SIMULATION of: {model_name} ==========")
# Initialize environment (single instance for simulation)
base_env = SFCEnv(
    T=100,
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

# Storage for plotting
inflation_rates = []
interest_rates = []
policy_actions = []
timesteps = list(range(base_env.T))

# Run simulation
obs = env.reset()
for step in range(base_env.T):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, infos = env.step(action)
    
    # Extract info
    # Access info from the first (and only) environment
    current_info = infos[0]  # Get info dict for first env
    current_inflation = current_info.get("π", 0)
    current_interest = current_info.get("r_b_", 0)
    # Convert action to integer index
    action_idx = int(action[0])  # Convert numpy array to integer
    action_value = base_env.action_value_ranges["r_b_"][action_idx]
    
    inflation_rates.append(current_inflation)
    interest_rates.append(current_interest)
    policy_actions.append(action_value)

# Plotting
plt.figure(figsize=(12, 9))

# Plot inflation and target
plt.subplot(3, 1, 1)
plt.plot(timesteps, inflation_rates, label='Actual Inflation', color='blue')
plt.axhline(y=pi_target, color='r', linestyle='--', label='Target Inflation')
plt.fill_between(timesteps, 
                 pi_target - threshold, 
                 pi_target + threshold, 
                 color='green', alpha=0.1, label=f'Target Zone (±{threshold:.3%})')
plt.ylabel('Inflation Rate (%)')
plt.title('Inflation Control Performance')
plt.legend()
plt.grid(True)

# Plot interest rates
plt.subplot(3, 1, 2)
plt.plot(timesteps, interest_rates, label='Policy Rate', color='red')
plt.ylabel('Interest Rate (%)')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)

# Plot policy actions
plt.subplot(3, 1, 3)
plt.step(timesteps, policy_actions, label='Policy Action', color='orange', where='post')
plt.scatter(timesteps, policy_actions, 
            color='darkorange', 
            zorder=3,
            s=15)     
plt.ylabel('Policy Action')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
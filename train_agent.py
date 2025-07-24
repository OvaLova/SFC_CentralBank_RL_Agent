from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_environment import SFCEnv
from model_matrices import balance_sheet_map, state


T = 100

env = SFCEnv(
    T=T,
    eq_file="equations_zezza.txt",
    init_state=state,              
    balance_sheet_map=balance_sheet_map,
)

check_env(env, warn=True)  # will raise errors if step(), reset() or spaces are misconfigured
model = PPO("MlpPolicy", env, verbose=1, n_steps=T, batch_size=int(T/2), tensorboard_log="./ppo_logs")
model.learn(total_timesteps=T*3, progress_bar=True)

# Optional: Save model
model.save("PPO_optimal_interest_rate_policy")
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from gym_environment import SFCEnv
from model_matrices import balance_sheet_map, state
from stable_baselines3.common.monitor import Monitor


T = 100
n_envs = 8
n_steps=int(T/4)
total_timesteps=T*100
rollout_size = n_steps * n_envs
batch_size=int(rollout_size/100)
verbose = 1
multiprocess = True
if multiprocess:
    process = "multiprocess"
else:
    process = "singleprocess"
loss = "piecewise"
model_name = f"PPO_optimal_interest_rate_policy_{process}_{loss}"

def train(multiprocess=False):
    base_env = SFCEnv(     # single process
        T=T,
        eq_file="equations_zezza.txt",
        init_state=state,              
        balance_sheet_map=balance_sheet_map,
        loss=loss
    )
    check_env(base_env, warn=True)  # will raise errors if step(), reset() or spaces are misconfigured

    if multiprocess:
        vectorized_env = SubprocVecEnv([make_env() for _ in range(n_envs)])  
        env = vectorized_env
    else: 
        env = DummyVecEnv([lambda: base_env])
    env = VecNormalize(env)
    model = PPO("MlpPolicy", env, verbose=verbose, n_steps=n_steps, batch_size=batch_size, tensorboard_log="./ppo_logs", normalize_advantage=True)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=model_name)

    # Save model (policy network) and normalization stats
    model.save(model_name)
    env.save(f"{model_name}_vecnormalize.pkl")

def resume_training(additional_timesteps=total_timesteps, multiprocess=False):
    base_env = SFCEnv(     # single process
        T=T,
        eq_file="equations_zezza.txt",
        init_state=state,              
        balance_sheet_map=balance_sheet_map,
        loss=loss
    )
    check_env(base_env, warn=True)  # will raise errors if step(), reset() or spaces are misconfigured
    
    if multiprocess:
        vectorized_env = SubprocVecEnv([make_env() for _ in range(n_envs)])   
        env = vectorized_env
    else: 
        env = DummyVecEnv([lambda: base_env])

    # Load the model with the environment
    try:
        env = VecNormalize.load(f"{model_name}_vecnormalize.pkl", env)
        env.training = False  # Disable stats updates
        env.norm_reward = False  # Don't normalize rewards for eval
        model = PPO.load(
            path=model_name,
            env=env,
            print_system_info=True,  # Helps verify compatibility
            verbose=verbose
        )
        print(f"Successfully loaded model previously trained for {model.num_timesteps} timesteps")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Continue training
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=False,  # Continue counting from previous total
            tb_log_name=model_name,      # Continue same TensorBoard log
            progress_bar=True
        )
        
        # Save and overwrite
        model.save(model_name)
        env.save(f"{model_name}_vecnormalize.pkl")  # Save updated stats!
        print(f"Resumed training complete. Total timesteps: {model.num_timesteps}")
    except Exception as e:
        print(f"Error during resumed training: {e}")

def make_env():
    def _init():
            env = SFCEnv(
                T=T,
                eq_file="equations_zezza.txt",
                init_state=state.copy(), 
                balance_sheet_map=balance_sheet_map,
                loss=loss 
            )
            env = Monitor(env)
            return env
    return _init


if __name__ == "__main__":
    train(multiprocess=multiprocess)
    # resume_training(multiprocess=multiprocess)
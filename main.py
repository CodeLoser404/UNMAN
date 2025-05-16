"""
main函数，用于训练模型并保存
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from envs.train_env import TrainEnv
from utils import custom_model
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

envs_path = './config/envs.yaml'
algs_path = './config/algs.yaml'


def train(continue_train=False, step=100000):
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "csv"])
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./model/",
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    base_env = TrainEnv(config_path=envs_path)
    env = Monitor(base_env)
    env = DummyVecEnv([lambda: env])

    if continue_train:
        vecnorm_path = f"./model/model_vecnormalize_{step}_steps.pkl"
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_obs = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # 创建评估环境（注意：不能和训练环境共享 VecNormalize 实例）
    eval_env = DummyVecEnv([lambda: base_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    # 用于保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./model/',
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    if continue_train:
        model_path = f"./model/model_{step}_steps.zip"
        model = custom_model.load_model(config_path=algs_path, model_path=model_path, env=env)
    else:
        model = custom_model.create_model(config_path=algs_path, env=env)

    model.set_logger(logger)
    model.learn(total_timesteps=100000, progress_bar=True, reset_num_timesteps=False, log_interval=1,
                callback=[checkpoint_callback, eval_callback])


def infer(step=100000):
    model_path = f"./model/model_{step}_steps.zip"
    vecnorm_path = f"./model/model_vecnormalize_{step}_steps.pkl"
    base_env = TrainEnv(config_path=envs_path)
    env = DummyVecEnv([lambda: base_env])

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_obs = True
    env.norm_reward = False

    model = custom_model.load_model(config_path=algs_path, model_path=model_path, env=env)

    reward_list = []
    for _ in range(20):
        obs = env.reset()[0]
        episode_reward = 0
        for _ in range(100):
            action = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break

        reward_list.append(episode_reward)
    print(np.mean(reward_list))


if __name__ == "__main__":
    train()
    # infer()
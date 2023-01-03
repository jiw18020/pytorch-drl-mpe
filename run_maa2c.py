import datetime
import argparse
import gym
import numpy as np
import os
#import tensorflow as tf
from time import time
import pickle
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('Agg')

from MAA2C import MAA2C
from common.utils import agg_double_list
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import sys
# import matplotlib.pyplot as plt

MAX_STEPS = 25
MAX_EPISODES = 1200
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 1
EVAL_INTERVAL = 1
USE_STL = False

# roll out n steps
ROLL_OUT_N_STEPS = 5
# only remember the latest ROLL_OUT_N_STEPS
#MEMORY_CAPACITY = ROLL_OUT_N_STEPS
MEMORY_CAPACITY = 10000
# only use the latest ROLL_OUT_N_STEPS for training A2C
#BATCH_SIZE = ROLL_OUT_N_STEPS
BATCH_SIZE = 100

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.
#
DONE_PENALTY = 0.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = 0.5

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 3

RANDOM_SEED = 2018

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--benchmark", type=bool, default=False, help="benchmark")
    parser.add_argument("--num-episodes", type=int, default=2, help="number of episodes")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env



def get_running_reward(reward_array: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(reward_array)
    for i in range(window - 1):
        running_reward[i] = np.mean(reward_array[:i + 1])
    for i in range(window - 1, len(reward_array)):
        running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
    return running_reward

def run(arglist):
    start = time()
    
    # create folder to save result
    env_dir = os.path.join('results', arglist.scenario)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    res_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(res_dir)
    model_dir = os.path.join(res_dir, 'model')
    os.makedirs(model_dir) 
    
    
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    env_eval = make_env(arglist.scenario, arglist, arglist.benchmark)
    # Create agent trainers
    # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    #num_adversaries = min(env.n, arglist.num_adversaries)

    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    print('obs_shape_n', obs_shape_n)
    act_shape_n = [env.action_space[i].n for i in range(env.n)]
    print('act_shape_n', act_shape_n)

    #maa2c = MAA2C(env, env.n, obs_shape_n, act_shape_n)
    maa2c = MAA2C(env=env, n_agents = env.n, state_dim = obs_shape_n[0], action_dim = act_shape_n[0], memory_capacity=MEMORY_CAPACITY,
              batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
              done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN, max_steps = MAX_STEPS, 
              critic_loss=CRITIC_LOSS)

    episodes =[]
    eval_rewards =[]
    while maa2c.n_episodes < MAX_EPISODES:
        maa2c.interact()
        maa2c.train()
        #env.render()
        #if maa2c.n_episodes >= EPISODES_BEFORE_TRAIN:
        #    maa2c.train()
        #maa2c.train()
        if maa2c.episode_done and ((maa2c.n_episodes)%EVAL_INTERVAL == 0):
            rewards, _ = maa2c.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            # print(rewards)
            #print("Episode %d, Average Reward %.2f, STD %.2f" % (maa2c.n_episodes, rewards_mu, rewards_std))
            print(f'episode {maa2c.n_episodes}: cumulative reward: {rewards_mu}')
            episodes.append(maa2c.n_episodes)
            eval_rewards.append(rewards_mu.tolist())
    # all episodes performed, training finishes
    # save agent parameters
    torch.save([actor.state_dict() for actor in maa2c.actors], os.path.join(res_dir, 'model.pt'))
    # save training reward
    # np.save(os.path.join(res_dir, 'rewards.npy'), total_reward)

    # plot result
    eval_rewards = np.array(eval_rewards)
    fig, ax = plt.subplots()
    x = range(1, MAX_EPISODES+1)
    for agent in range(env.n):
        ax.plot(x, eval_rewards[:, agent], label=agent)
        ax.plot(x, get_running_reward(eval_rewards[:, agent]))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maa2c solve {arglist.scenario}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    print(f'training finishes, time spent: {datetime.timedelta(seconds=int(time() - start))}')



if __name__ == '__main__':
    arglist = parse_args()
    run(arglist)

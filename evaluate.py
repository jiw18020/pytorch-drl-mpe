import datetime
import argparse
import gym
import numpy as np
import os
#import tensorflow as tf
import time
import pickle
from matplotlib import pyplot as plt

from MAA2C import MAA2C
from common.utils import agg_double_list
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import sys


MAX_EPISODES = 3000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 1
EVAL_INTERVAL = 1

# roll out n steps
ROLL_OUT_N_STEPS = 5
# only remember the latest ROLL_OUT_N_STEPS
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
# only use the latest ROLL_OUT_N_STEPS for training A2C
BATCH_SIZE = ROLL_OUT_N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.01
#
DONE_PENALTY = 0.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2018

    # create env
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

def run(args):


    # create folder to save result
    
    env = make_env(args.scenario, args)

    # get dimension info about observation and action
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    print('obs_shape_n', obs_shape_n)
    act_shape_n = [env.action_space[i].n for i in range(env.n)]
    print('act_shape_n', act_shape_n)

    maa2c = MAA2C(env=env, n_agents = env.n, state_dim = obs_shape_n[0], action_dim = act_shape_n[0], memory_capacity=MEMORY_CAPACITY,
              batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
              done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN,
              critic_loss=CRITIC_LOSS)
    model_dir = os.path.join('results', args.scenario, args.folder)
    assert os.path.exists(model_dir)
    data = torch.load(os.path.join(model_dir, 'model.pt'))
    for actor, actor_parameter in zip(maa2c.actors, data):
        actor.load_state_dict(actor_parameter)
    print(f'MAA2C load model.pt from {model_dir}')

    total_reward = np.zeros((args.episode_num, env.n))  # reward of each episode
    for episode in range(args.episode_num):
        obs = env.reset()
        # record reward of each agent in this episode
        episode_reward = np.zeros((args.episode_length, env.n))
        for step in range(args.episode_length):  # interact with the env for an episode
            actions = maa2c.eval_action(obs)
            print('actions', actions)
            next_obs, rewards, dones, infos = env.step(actions)
            #print(next_obs[1])
            episode_reward[step] = rewards
            env.render()  # connect to a display
            time.sleep(0.02)
            obs = next_obs

        # episode finishes
        # calculate cumulative reward of each agent in this episode
        cumulative_reward = episode_reward.sum(axis=0)
        total_reward[episode] = cumulative_reward
        print(f'episode {episode + 1}: cumulative reward: {cumulative_reward}')
    mean_episode_reward = total_reward.mean(axis=0)
    print(f'mean reward: {mean_episode_reward}')

    # all episodes performed, evaluate finishes
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, total_reward[:, agent], label=agent)
        # ax.plot(x, get_running_reward(total_reward[:, agent]))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'evaluating result of maa2c without STL solve {args.scenario}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument('--folder', type=str, default='5', help='name of the folder where model is saved')
    parser.add_argument('--episode-length', type=int, default=25, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=10, help='total number of episode')
    args = parser.parse_args()
    run(args)

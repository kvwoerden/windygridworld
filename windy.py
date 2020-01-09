import argparse
import sys
import os

import gym
from gym import wrappers, logger
import gym_windy_gridworlds

import numpy as np
import random

import time
from pathlib import Path

import matplotlib.pyplot as plt

import asyncio

from copy import deepcopy

import itertools

import math

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class KeyboardAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.actions = { 'w': 'U', 'a': 'L', 's': 'D', 'd': 'R'}

    def act(self):
        print("Enter one of the keys [a,w,s,d] (lower case) to make a move")
        key = input()
        if key in self.actions:
            return self.actions[key]
        else:
            return ''

class SarsaAgent(object):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.actions = { 'w': 'U', 'a': 'L', 's': 'D', 'd': 'R'}
        self.height = self.state_space[0].n
        self.width = self.state_space[1].n
        self.nA = self.action_space.n
        self.Q = np.zeros((self.height, self.width, self.nA))

    def choose_act(self, state, epsilon):
        if np.random.random() > epsilon:
            Qa = self.Q[state[0], state[1]]
            Qamax = np.max(Qa)
            action = random.choice(*np.where(Qa == Qamax))
        else:
            action = np.random.randint(self.nA)
        return action

    def updateQ(self, state, action, stateprime, actionprime, reward, gamma, alpha):
        self.Q[state[0], state[1], action] += alpha*(reward + gamma*self.Q[stateprime[0], stateprime[1], actionprime] - self.Q[state[0], state[1], action]) 

class NStepSarsaAgent(object):
    def __init__(self, state_space, action_space, n_step):
        self.state_space = state_space
        self.action_space = action_space
        self.actions = { 'w': 'U', 'a': 'L', 's': 'D', 'd': 'R'}
        self.height = self.state_space[0].n
        self.width = self.state_space[1].n
        self.nA = self.action_space.n
        self.Q = np.zeros((self.height, self.width, self.nA))
        self.n_step = n_step

    def choose_act(self, state, epsilon):
        if np.random.random() > epsilon:
            Qa = self.Q[state[0], state[1]]
            Qamax = np.max(Qa)
            action = random.choice(*np.where(Qa == Qamax))
        else:
            action = np.random.randint(self.nA)
        return action

    def updateQ(self, state, action, stateprime, actionprime, reward, gamma, alpha):
        self.Q[state[0], state[1], action] += alpha*(reward + gamma*self.Q[stateprime[0], stateprime[1], actionprime] - self.Q[state[0], state[1], action]) 

async def savefigure(Q, axis, cmap, interpolation, images_path, step, zeroes, title, episode, min_steps):
    print("Saving the estimation of Q at step " + str(step))
    font = {'weight' : 'bold',
        'size'   : 22}
    plt.rc('font', **font)
    plt.figure(figsize=(19.2, 10.8))
    plt.imshow(np.max(Q, axis=axis), cmap=cmap, interpolation=interpolation)
    cbar = plt.colorbar()
    cbar.set_label('Value estimate')
    plt.title(title)
    ax = plt.gca()
    ax.set_xticks(range(10))
    ax.set_xticklabels([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
    ax.set_yticks([])
    ax.set_xlabel("Step: " + str(step).zfill(5) + "\nEpisode: " + str(episode).zfill(3) + ", Length of best path found so far: " + str(min_steps))
    plt.savefig(images_path / ('Q' + str(step).zfill(zeroes) + '.png'), dpi=100)
    plt.close()



async def main():
    logger.set_level(logger.INFO)

    # env = gym.make(args.env_id)
    env = gym.make('WindyGridWorld-v0')
    # rec = VideoRecorder(env, path='./video/output01.mp4')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = './video/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    # agent = RandomAgent(env.action_space)
    

    episode_count = 100
    reward = 0
    done = False
    EPSILON = 0.05
    GAMMA = 1
    ALPHA = 0.1
    steps = np.zeros((episode_count, 2))
    steps[:, 0] = np.arange(episode_count)
    INTERVAL = 1
    SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    IMAGES_PATH = SCRIPT_DIR / 'images' / TIMESTR
    print(IMAGES_PATH)
    os.mkdir(IMAGES_PATH)
    STEP = 0
    STEP_INTERVAL = 6
    EPSILON_INTERACTIVE = False
    RENDER = False
    ZEROES = 5
    tasks = []
    Q_state = {}

    PLOT = True
    N_STEP = 2
    TITLE = "Windy Gridworld " + str(N_STEP) +"-step Sarsa agent \n epsilon=" + str(EPSILON) + ", alpha=" + str(ALPHA)
    # a = np.random.random((16, 16))
    # b = np.random.random((16, 16))
    # plt.imshow(a, cmap='bwr', interpolation='nearest')
    # plt.savefig(IMAGES_PATH / 'test.png')
    # plt.imshow(b, cmap='bwr', interpolation='nearest')
    # plt.savefig(IMAGES_PATH / 'test02.png')
    agent = NStepSarsaAgent(env.observation_space, env.action_space, N_STEP)

    for i in range(episode_count):
        state = env.reset()
        action = agent.choose_act(state, EPSILON)
        moves = []
        states = {0: state}
        actions = {0: action}
        rewards = {}
        T = math.inf
        for t in itertools.count():
            if i % INTERVAL == 0 and RENDER:
                # a = input()
                env.render()
            if STEP % STEP_INTERVAL == 0 and PLOT:
                # print("### 001 in print block")
                Q_state[STEP] = np.copy(agent.Q)
                try:
                    min_steps = np.amin(steps[:,1][:i])
                    # print("### steps\n", steps[:,1][:i])
                    # print("### npmin", min_steps)
                except ValueError:
                    min_steps = np.inf
                task = asyncio.create_task(savefigure(Q_state[STEP], axis=2, cmap='bwr', interpolation='nearest', images_path=IMAGES_PATH, step=STEP, zeroes=ZEROES, title=TITLE, episode=i, min_steps=min_steps))
                tasks.append(task)

            # print("Currently at position: ", env.observation)
            if t < T:
                stateprime, reward, done, _ = env.step(action)
                rewards[t+1] = reward
                states[t+1] = stateprime
                moves.append((state, action))
                steps[i, 1] += 1
                if done:
                    T = t + 1
                else:
                   
                    # env.render()
                    # action = agent.act(ob, reward, done)
                    actionprime = agent.choose_act(stateprime, EPSILON)
                    actions[t+1] = actionprime
                    # agent.updateQ(state, action, stateprime, actionprime, reward, GAMMA, ALPHA)
                    state = stateprime
                    action = actionprime
                    STEP += 1
            tau = t - agent.n_step + 1
            if tau >= 0:
                G = sum((rewards[ix] for ix in range(tau+1, min(tau + agent.n_step, T) + 1)))
                if tau + agent.n_step < T:
                    G = G + agent.Q[states[tau+agent.n_step][0], states[tau+agent.n_step][1], actions[tau+agent.n_step]]
                agent.Q[states[tau][0], states[tau][1], actions[tau]] += ALPHA*(G - agent.Q[states[tau][0], states[tau][1], actions[tau]])
            if tau == T - 1:
                break 
            # if done:
            #     if steps[:,1][:i].size != 0 and steps[i, 1] < np.amin(steps[:,1][:i]):
            #         print("New shortest path of length " + str(int(steps[i,1])) + " found: \n", moves)
            #     if i % INTERVAL == 0:
            #         if RENDER:
            #             env.render()
            #         print("Done with episode " + str(i))
            #         print("This took " + str(steps[i, 1]) + " steps")    
            #         if EPSILON_INTERACTIVE:
            #             print("Pick new epsilon")
            #             epsilon_string = input()
            #             if epsilon_string != '':
            #                 EPSILON = float(epsilon_string)
            #     # print("Done with episode " + str(i) + " at step " + str(STEP))
            #     break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
           
    env.close()
    asyncio.wait(tasks)
    print(steps[-20:,:])

if __name__ == '__main__':
    asyncio.run(main())
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='WindyGridWorldEnv-v0', help='Select the environment to run')
    # args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.

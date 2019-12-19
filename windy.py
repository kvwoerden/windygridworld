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

        


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='WindyGridWorldEnv-v0', help='Select the environment to run')
    # args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
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
    agent = SarsaAgent(env.observation_space, env.action_space)

    episode_count = 250
    reward = 0
    done = False
    EPSILON = 0.1
    GAMMA = 1
    ALPHA = 0.5
    steps = np.zeros((episode_count, 2))
    steps[:, 0] = np.arange(episode_count)
    INTERVAL = 100
    SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    IMAGES_PATH = SCRIPT_DIR / 'images' / TIMESTR
    print(IMAGES_PATH)
    os.mkdir(IMAGES_PATH)
    STEP = 0
    STEP_INTERVAL = 20
    EPSILON_INTERACTIVE = False
    RENDER = False
    ZEROES = 4
    # a = np.random.random((16, 16))
    # b = np.random.random((16, 16))
    # plt.imshow(a, cmap='bwr', interpolation='nearest')
    # plt.savefig(IMAGES_PATH / 'test.png')
    # plt.imshow(b, cmap='bwr', interpolation='nearest')
    # plt.savefig(IMAGES_PATH / 'test02.png')

    for i in range(episode_count):
        state = env.reset()
        action = agent.choose_act(state, EPSILON)

        while True:
            if i % INTERVAL == 0 and RENDER:
                env.render()
            if STEP % STEP_INTERVAL == 0:
                plt.imshow(np.max(agent.Q, axis=2), cmap='bwr', interpolation='nearest')
                plt.savefig(IMAGES_PATH / ('Q' + str(STEP).zfill(ZEROES) + '.png'))

            # print("Currently at position: ", env.observation)
            stateprime, reward, done, _ = env.step(action)
            steps[i, 1] += 1
            # env.render()
            # action = agent.act(ob, reward, done)
            actionprime = agent.choose_act(stateprime, EPSILON)
            agent.updateQ(state, action, stateprime, actionprime, reward, GAMMA, ALPHA)
            state = stateprime
            action = actionprime
            STEP += 1
            if done:
                if i % INTERVAL == 0:
                    if RENDER:
                        env.render()
                    print("Done with episode " + str(i))
                    print("This took " + str(steps[i, 1]) + " steps")    
                    if EPSILON_INTERACTIVE:
                        print("Pick new epsilon")
                        epsilon_string = input()
                        if epsilon_string != '':
                            EPSILON = float(epsilon_string)
                print("Done with episode " + str(i) + " at step " + str(STEP))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
           
    env.close()
    print(steps[-20:,:])

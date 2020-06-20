from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
import numpy as np
import statistics
import json
import argparse

from connect4 import Connect4
from dqn_agent import DQNAgent

# Environment settings
TESTS = 100
EPSIOLN_START = 0.2
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_STATS_EVERY = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", help="Games played",
                        type=int)
    parser.add_argument("learning_rate", help="Learning Rate",
                        type=float)
    parser.add_argument("batch_size", help="Batch Size",
                        type=int)

    args = parser.parse_args()

    episodes = args.episodes
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    print(f"episodes = {episodes}")
    print(f"TESTS = {TESTS}")
    print(f"EPSIOLN_START = {EPSIOLN_START}")
    print(f"learning_rate = {learning_rate}")
    print(f"batch_size = {batch_size}")

    for i in range(1):
        env = Connect4()
        dqn_agent = DQNAgent(learning_rate, batch_size)
        print(f"Simulation: {i + 1}")
        training(dqn_agent, env, learning_rate, batch_size, episodes)


def pick_best_action(available, sorted_actions):
    for action in reversed(sorted_actions):
        if action in available:
            return action


def training(agent, environment, learning_rate, batch_size, episodes):
    epsilon = EPSIOLN_START
    score_list = []
    random_score_list = []
    random_score_list.append(testing(agent, environment, "random"))

    if not os.path.isdir(f'stats_{learning_rate}_{batch_size}'):
        os.makedirs(f'stats_{learning_rate}_{batch_size}')

    with open(os.path.join(f"stats_{learning_rate}_{batch_size}", "weights_before.txt"), "w") as f:
        f.write(f"{agent.model.get_weights()}")

    for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes', position=0, leave=True):
        # Restarting episode - reset episode reward and step number
        move = 0
        states = []
        actions = []

        # Reset environment and get initial state
        state = environment.reset()
        states.append(state * 1)

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            move += 1
            # DQN-Agents turn
            if move % 2 == 0:
                # Pick action
                if np.random.random() > epsilon:
                    # Get action from Q table
                    actions_sorted_lo_hi = np.argsort(
                        agent.get_qs(environment.board))
                    available_actions = environment.get_available_moves()
                    action = pick_best_action(
                        available_actions, actions_sorted_lo_hi)
                else:
                    # Get random action
                    action = environment.random_move()

                # Make move
                new_state, reward, done = environment.step(1, action)
                states.append(new_state * -1)
                actions.append(action)

            # Benchplayers turn
            if move % 2 != 0:
                action = environment.bench_move()
                new_state, reward, done = environment.step(-1, action)
                states.append(new_state * 1)
                actions.append(action)

            agent.train(done)

        # Update Replay Memory Buffer after each game
        for i in range(2, move):
            agent.update_replay_memory(
                (states[i - 2], actions[i - 2], 0, states[i], False))

        agent.update_replay_memory(
            (states[move - 2], actions[move - 2], -1, states[move], True))
        agent.update_replay_memory(
            (states[move - 1], actions[move - 1], 1, states[move] * reward, True))

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        # print(agent.replay_memory)
        # From here on stats only
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            win_score = testing(agent, environment)
            score_list.append(win_score)

        if not episode % SHOW_STATS_EVERY:
            random_score_list.append(testing(agent, environment, "random"))
            with open(os.path.join(f"stats_{learning_rate}_{batch_size}", f"Experiment_{time.time()}.json"), "w") as f:
                json.dump(score_list, f)

    with open(os.path.join(f"stats_{learning_rate}_{batch_size}", f"Experiment_{time.time()}.json"), "w") as f:
        json.dump(score_list, f)

    with open(os.path.join(f"stats_{learning_rate}_{batch_size}", "weights_after.txt"), "w") as f:
        f.write(f"{agent.model.get_weights()}")

    print(random_score_list)


def testing(agent, environment, mode="bench"):
    score = 0

    for test in range(TESTS):
        current_state = environment.reset()

        done = False
        while not done:
            actions_sorted_lo_hi = np.argsort(agent.get_qs(environment.board))
            available_actions = environment.get_available_moves()
            action = pick_best_action(available_actions, actions_sorted_lo_hi)

            new_state, reward, done = environment.step(1, action)

            # Opponents move
            if not done:
                if mode == "random":
                    opponent_action = environment.random_move()
                else:
                    opponent_action = environment.bench_move()
                new_state, reward, done = environment.step(-1, opponent_action)

        if reward == 1:
            score += 1
        if reward == 0:
            score += 0.5

    return score


if __name__ == "__main__":
    main()

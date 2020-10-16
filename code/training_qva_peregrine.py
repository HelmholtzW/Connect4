from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
import numpy as np
import statistics

from connect4 import Connect4
from dqva_agent import DQVAAgent

# Environment settings
EPISODES = 50000
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
TESTS = 1000
EPSIOLN_START = 0.2
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
REPLAY_MEMORY_SIZE = 100000
MIN_REPLAY_MEMORY_SIZE = 1000
DISCOUNT = 1.0
UPDATE_TARGET_EVERY = 5

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # EPISODES
SHOW_STATS_EVERY = 1000


def main():
    for i in range(1):
        env = Connect4()
        dqn_agent = DQVAAgent(LEARNING_RATE, BATCH_SIZE, REPLAY_MEMORY_SIZE,
                              MIN_REPLAY_MEMORY_SIZE, UPDATE_TARGET_EVERY)
        training(dqn_agent, env, LEARNING_RATE, BATCH_SIZE, EPISODES)


def pick_best_action(available, sorted_actions):
    for action in reversed(sorted_actions):
        if action in available:
            return action


def training(agent, environment, learning_rate, batch_size, episodes):
    epsilon = EPSIOLN_START
    score_list = []
    # random_score_list = []
    # random_score_list.append(testing(agent, environment, "random"))

    # if not os.path.isdir(f'stats_{learning_rate}_{batch_size}'):
    #     os.makedirs(f'stats_{learning_rate}_{batch_size}')

    # with open(os.path.join(f"stats_{learning_rate}_{batch_size}", "weights_before.txt"), "w") as f:
    #     f.write(f"{agent.model.get_weights()}")

    # test_moves(agent)

    for episode in range(1, episodes + 1):
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
            if move % 2 != 0:
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
            if move % 2 == 0:
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

    #     if not episode % SHOW_STATS_EVERY:
    #         with open(os.path.join(f"stats_{learning_rate}_{batch_size}", f"Experiment_{time.time()}.json"), "w") as f:
    #             json.dump(score_list, f)

    # with open(os.path.join(f"stats_{learning_rate}_{batch_size}", f"Experiment_{time.time()}.json"), "w") as f:
    #     json.dump(score_list, f)

    # with open(os.path.join(f"stats_{learning_rate}_{batch_size}", "weights_after.txt"), "w") as f:
    #     f.write(f"{agent.model.get_weights()}")

    # test_moves(agent)
    print([score_list, learning_rate, batch_size,
           REPLAY_MEMORY_SIZE, UPDATE_TARGET_EVERY])


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

    return score / TESTS * 100


def test_moves(agent):
    test_row = [[1, -1, 1, 1, -1, 0, -1],
                [1, 1, 1, 0, -1, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
    test_column = [[1, -1, 1, 1, -1, 0, -1],
                   [1, -1, 1, 0, -1, 0, -1],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]
    test_diag = [[1, -1, 1, 1, -1, -1, -1],
                 [-1, -1, 1, 0, -1, 0, -1],
                 [1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]

    print(f"Row: 3 / {agent.get_qs(np.array(test_row))}")
    print(f"Col: 0 / {agent.get_qs(np.array(test_column))}")
    print(f"Diag: 0 / {agent.get_qs(np.array(test_diag))}")


if __name__ == "__main__":
    main()

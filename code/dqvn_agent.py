from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np


DISCOUNT = 1.0
SOFT_UPDATE = 0.001


class DQVNAgent:
    def __init__(self, learning_rate, minibatch_size, replay_memory_size, min_replay_memory_size, update_rate):

        # main model  # gets trained every step
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.min_replay_memory_size = min_replay_memory_size
        self.update_rate = update_rate
        self.q_model = self.create_q_model()
        self.v_model = self.create_v_model()

        # Target model this is what we .predict against every step
        self.target_v_model = self.create_v_model()
        self.target_v_model.set_weights(self.v_model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)
        # self.tensorboard = ModifiedTensorBoard(
        #     log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def create_q_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(6, 7)))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(7, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])
        return model

    def create_v_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(6, 7)))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0]
                                   for transition in minibatch])
        current_qs_list = self.q_model.predict(current_states)
        current_vs_list = self.v_model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        future_vs_list = self.target_v_model.predict(new_current_states)

        X = []
        qy = []
        vy = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if done:
                y_t = reward
            else:
                y_t = reward + DISCOUNT * future_vs_list[index]

            # Update Q and V value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = y_t

            current_v = y_t

            # And append to our training data
            X.append(current_state)
            qy.append(current_qs)
            vy.append(current_v)

        # Fit on all samples as one batch, log only on terminal state
        self.q_model.fit(np.array(X), np.array(qy), batch_size=self.minibatch_size, verbose=0,
                         shuffle=False)
        self.v_model.fit(np.array(X), np.array(vy), batch_size=self.minibatch_size, verbose=0,
                         shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.update_rate:
            v_model_weights = np.array(self.v_model.get_weights())
            v_target_weights = np.array(self.target_v_model.get_weights())

            self.target_v_model.set_weights(
                v_model_weights * SOFT_UPDATE + v_target_weights * (1 - SOFT_UPDATE))

            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.q_model.predict(np.array(state).reshape(-1, *state.shape))[0]

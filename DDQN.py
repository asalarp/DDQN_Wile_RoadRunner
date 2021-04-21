import time
import sys
import copy
import random
import numpy as np
import tensorflow
from Environment import Env
from collections import deque
from tensorflow.keras.layers import Dense, Conv2D, Reshape, MaxPool2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential


EPISODES = 5000
STATE_SZIE = (128, 15, 15, 1)
TEST = False


class DDQNAgent:
    def __init__(self):

        self.render = False
        self.load = False
        self.save_loc = './DDQN'
        self.action_size = 4
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.99997
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.train_start = 500
        self.memory = deque(maxlen=2000)
        self.state_size = STATE_SZIE
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if self.load:
            self.load_model('DDQN_Model.h5')

    # Neural Network for Deep Q-learning
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5),  strides=1,  padding='same',
                  input_shape=self.state_size[1:], activation='relu'))
        model.add(Conv2D(128, (3, 3), strides=1,
                  padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), strides=1,
                  padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), strides=1,
                  padding='same', activation='relu'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))

        return model

    # select action using epsilon-greedy
    def select_action(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])


    def MEMORY(self, state, action, reward, next_state, goal, dynamite):
        self.memory.append((state, action, reward, next_state, goal, dynamite))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_replay(self):

        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros(
            (batch_size, self.state_size[1], self.state_size[2], self.state_size[3]))
        update_target = np.zeros(
            (batch_size, self.state_size[1], self.state_size[2], self.state_size[3]))
        action, reward, goal = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            goal.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            if goal[i]:
                target[i][action[i]] = reward[i]

            else:
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + \
                    self.discount_factor * (target_val[i][a]) # #belman eq

        self.model.fit(update_input, target,
                       batch_size=self.batch_size, epochs=1, verbose=0)

    # save the model which is under training
    def save_model(self):
        self.model.save_weights('model.h5')

    # load the saved model
    def load_model(self):
        self.model.load_weights('model.h5')


if __name__ == "__main__":

    # create environment
    env = Env()
    agent = DDQNAgent()
    scores, episodes = [], []
    global_step = 0
    for e in range(EPISODES):
        state = env.reset()
        goal = False
        dynamite = False
        reset_step = False
        score = 0
        reset = True
        state = np.reshape(
            state, (1, agent.state_size[1], agent.state_size[2], agent.state_size[3]))
        while (not goal) and (not dynamite):
            if agent.render:
                env.render()
            global_step += 1
            action = agent.select_action(state)
            next_state, reward, goal, dynamite = env.step(action)
            next_state = np.reshape(
                next_state, (1, agent.state_size[1], agent.state_size[2], agent.state_size[3]))
            agent.MEMORY(state, action, reward, next_state, goal, dynamite)
            agent.train_replay()
            score += reward
            state = copy.deepcopy(next_state)

            if goal == True:
                env.reset()
                agent.update_target_model()
                print("episode: {:3}   score: {:8.6}    epsilon {:.3}"
                      .format(e, float(score), float(agent.epsilon)))

            elif dynamite == True:
                env.reset()
                agent.update_target_model()

                print("episode: {:3}   score: {:8.6}    epsilon {:.3}"
                      .format(e, float(score), float(agent.epsilon)))

        # save the model every 100 episodes
        if e % 100 == 0:
            agent.save_model()

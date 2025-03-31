""" Implement the Healthcare Agents """
import numpy as np
import gym
import torch
import random
from collections import deque
from tensorflow import keras
from keras import models, Input
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from keras.optimizers import Adam # type: ignore


class HealthQLearnVFA:
    """
    Q-learning VFA implementation
    """
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float,
                 epsilon_decay: float,
                  final_epsilon: float,
                   gamma: float =.95, initial_w: np.ndarray = None) -> None:
        self.env = environment
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []

        if initial_w is None:
            num_features = len(self.env.observation_space.keys())
            i_w = np.ones((self.env.action_space.n, num_features))
        else:
            i_w = initial_w
        self.w = torch.tensor(i_w, dtype=float, requires_grad=True)

    def x(self, state: dict) -> np.ndarray:
        """Returns the features that represents the state

        Args:
            state (dict): The state as a dictionary obtained as the observation object returned by a step
        Returns:
            torch.tensor: state observed by the agent in terms of state features 
        """
        features = np.array([float(value) for value in state.values()])  # Convert to float
        return torch.tensor(features, dtype=float, requires_grad=False)
    
    def q(self, state: int, action: int) -> float:
        return self.x(state) @ self.w[:, action]
        
    def policy(self, state: int) -> int:
        """Implements e-greedy strategy for action selection

        Args:
            state (tuple[int, int, bool]): state observed according to the environment

        Returns:
            int: action
        """
        available = self.env.actions_available
        options = [i for i, a in enumerate(available) if a == 1]
        if np.random.random() < self.epsilon:
            return np.random.choice(options)
        else:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return options[np.argmax(available_values)]

    def update(self, state: int, action: int, reward: float, s_prime: int):
        next_action = self.policy(state)
        q_target = reward + self.gamma * self.q(s_prime, next_action)
        q_value = self.q(state, action)
        delta = q_target - q_value
        q_value.backward()
        with torch.no_grad():
            self.w += self.alpha * delta * self.w.grad 
            self.w.grad.zero_()
        self.training_error.append(delta.detach().numpy())
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

class HealthcareDQL:
        """
        Deep Q-Learning Agent
        """
        def __init__(self, environment: gym.Env, state_size, action_size, learning_rate: float,
                     gamma: float, epsilon: float, final_epsilon: float, epsilon_decay: float):
            self.env = environment
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.epsilon = epsilon
            self.final_epsilon = final_epsilon
            self.epsilon_decay = epsilon_decay
            self.alpha = learning_rate
            self.memory = deque(maxlen=2000)

            self.model = self.DQNetwork_model() # build the DQN model
    
        def DQNetwork_model(self):
                    """
                    Simple neural network for Q-value approximation
                    """
                    model = Sequential()
                    model.add(Input(shape=(self.state_size,)))
                    model.add(Dense(24, input_dim=self.state_size, activation='relu'))
                    model.add(Dense(24, activation='relu'))
                    model.add(Dense(self.action_size, activation='linear'))
                    model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
                    return model
        
        def remember(self, state, action, reward, next_state, done):
            """Store experience in memory"""
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            """Epsilon-greedy action selection"""
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            state = np.array(state).reshape(1, -1)
            act_values = self.model.predict(state) # predict q-values
            return np.argmax(act_values[0]) # return action with max q-values
        
        def replay(self, batch_size):
            """Trains the agent with experiences from the memory buffer"""
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    next_state = np.array(next_state).reshape(1, -1)
                    target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def update_target_network(self):
            """Update the target network with the weights of the current model"""
            self.target_network.load_state_dict(self.model.state_dict())

        def decay_epsilon(self):
            """Decay epsilon value over time"""
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd

from environment import Status
from models import AbstractModel


class SarsaNStep(AbstractModel):
    """ Temopral Difference Learning based prediction model.

        For every state (here: the agents current location ) the value for each of the actions is stored in a table.
        The key for this table is (state + action). Initially all values are 0. When playing training games after 
        every move the value in the table is updated based on the rewards gained N steps after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).
    """
    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, game, **kwargs):
        """ Create a new prediction model for 'game'.

        :param class Maze game: Maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="TDLearning", **kwargs)
        self.Q = dict()  # table with value per (state, action) combination

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train the model.

            :param stop_at_convergence: stop training as soon as convergence is reached

            Hyperparameters:
            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
            :keyword int steps: number of steps in bootstrapping
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        steps = max(kwargs.get("steps", 1), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        mean_episode_length = 0

        # training starts here
        for episode in range(1, episodes + 1):
            # optimization: make sure to start from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key

            if np.random.random() < exploration_rate:
                action = random.choice(self.environment.actions)
            else:
                action = self.predict(state)

            state_history = []
            state_history.append( state )

            action_history = []
            action_history.append( action )

            reward_history = []

            while True:
                mean_episode_length += 1

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                next_action = self.predict(next_state)  # choose A(t+1) greedily

                reward_history.append(reward)
                cumulative_reward += reward

                if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0
                
                if len(state_history) >= steps:
                    update_state = state_history[0]
                    state_history = state_history[1:]

                    update_action = action_history[0]
                    action_history = action_history[1:]

                    reward_term = 0

                    for idx, reward_old in enumerate(reward_history):
                        reward_term += discount ** idx * reward_old

                    reward_history = reward_history[1:]

                    next_Q = self.Q.get((next_state, next_action), 0.0)

                    self.Q[(update_state, update_action)] += learning_rate * (reward_term + discount ** steps * next_Q - self.Q[(update_state, update_action)])

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break

                state = next_state
                action = next_action

                state_history.append(next_state)
                action_history.append(next_action)

                self.environment.render_q(self)

            for idx in range(len(reward_history)):
                update_state = state_history[idx]
                update_action = action_history[idx]

                reward_term = 0

                for idx_r, reward_old in enumerate(reward_history[idx:]):
                    reward_term += discount ** idx_r * reward_old
                
                self.Q[(update_state, update_action)] += learning_rate * (reward_term - self.Q[(update_state, update_action)])

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            if episode % check_convergence_every == 0:
                # check if the current model does win from all starting cells
                # only possible if there is a finite number of starting states
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        mean_episode_length /= episode
        logging.info('cumulative reward: {:f} | mean episode length: {:f}'.format(cumulative_reward, mean_episode_length))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: game state
            :return int: selected action
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)

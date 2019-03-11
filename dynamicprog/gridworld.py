import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import *

import utils

ACTIONS = ['N', 'W', 'S', 'E']


def show_values(values, save=None, show=True, annotate=True, size=5):
    plt.figure(figsize=(2 * size, size))
    sns.heatmap(values, annot=annotate, square=True, cbar=not annotate)

    if save:
        assert isinstance(save, str)
        plt.savefig(
            'graphics/' + save + '.pdf',
            bbox_inches='tight'
        )

    if not show:
        plt.close()


def show_policy(policy, save=None, show=True, annotate=True, size=5):
    p = policy + 1
    p[0, 0] = p[0, -1] = 0

    if annotate:
        n = len(policy)

        annotate = np.empty_like(policy, dtype=str)
        for i, j in product(range(n), range(n)):
            annotate[i, j] = ACTIONS[policy[i, j]]
        annotate[0, 0] = annotate[0, -1] = ''

    plt.figure(figsize=(2 * size, size))
    sns.heatmap(p, annot=annotate, square=True, cbar=False, fmt='')

    if save:
        assert isinstance(save, str)
        plt.savefig(
            'graphics/' + save + '.pdf',
            bbox_inches='tight'
        )

    if not show:
        plt.close()


def show_training(training_values, save=None, title=None, show=True, size=(10, 6)):
    plt.figure(figsize=size)

    x = list(range(len(training_values)))

    plt.plot(x, [v for v in training_values[:, 0]], label='bottom left')
    plt.plot(x, [v for v in training_values[:, 1]], label='bottom right')

    plt.xlabel(r'Number of evaluations of the greedy policy', fontsize=13)
    plt.legend(fontsize=13)

    if title is not None:
        assert isinstance(title, str)
        plt.title(title, size=15)

    if save:
        assert isinstance(save, str)
        plt.savefig(
            'graphics/' + save + '.pdf',
            bbox_inches='tight'
        )

    if not show:
        plt.close()
    else:
        plt.show()


class GridWorld:

    def __init__(self, n, verbose=True, p=.9, discount=.9, save_values=False):
        self.n = n

        self.reward = np.zeros((n, n))
        self.reward[0, 0] = 1
        self.reward[0, -1] = 10

        self.p = p
        self.discount = discount

        self.transitions = None

        self.verbose = verbose

        self.compute_transitions()

        self.save_values = save_values
        self.training_values = []

    def print(self, text, end='\n'):
        if self.verbose:
            print(text, end=end)

    def transition_matrix(self, action, p=.7):
        """
        Returns the transition probability matrix associated to an action (same action on the entire gridworld).

        Args:
            action (str): 'north', 'west', 'south' or 'east'.
            p (float): the probability of the action to actually performed.

        Returns:
            The transition probability matrix associated to an action.

        Raises:
            AssertionError: if the action is an illegal move.
        """

        assert action in ACTIONS

        v = 1 * (action == 'S') - 1 * (action == 'N')
        h = 1 * (action == 'W') - 1 * (action == 'E')

        transition = np.zeros((self.n ** 2, self.n ** 2))

        moves = [(i, j) for i, j in zip([0, 0, 1, -1], [1, -1, 0, 0])]

        q = (1 - p) / 4

        probability = {move: q for move in moves}
        probability[(v, h)] += p

        for i, j in product(range(self.n), range(self.n)):

            if (i, j) in {(0, 0), (0, self.n - 1)}:
                # terminal states have 0 probability of accessing any other state
                continue

            for move in moves:
                v, h = move

                i_ = i + v
                j_ = j + h

                within = (-1 < i_ < self.n) and (-1 < j_ < self.n)

                if within:
                    transition[i * self.n + j, (i + v) * self.n + (j + h)] += probability[move]
                else:
                    # If the action takes us over the edge, we stay in the same place.
                    transition[i * self.n + j, i * self.n + j] += probability[move]

        return transition

    @utils.description('Computing transition matrices')
    def compute_transitions(self):
        r"""
        Compute the transition probability tensor.

        .. math::

            `\texttt{transition[a][i, j]} = p(j | i, a) = P(S_{t+1} = j | S_t = i, A_t = a)`
        """
        self.transitions = np.array([self.transition_matrix(action, p=self.p) for action in ACTIONS])

    def transition_policy(self, policy):
        """
        Computes the transition probability for a given policy ("where the wind blows us").

        Args:
            policy: the deterministic policy to consider.

        Returns:
            transition (numpy array): the transition probability matrix for a given deterministic policy.

        """
        return np.array([self.transitions[a, i, :] for i, a in enumerate(policy.reshape(-1))])

    def compute_exact_values(self, policy):
        """
        Compute the exact value function for a given deterministic policy.

        Args:
            policy: the deterministic policy to consider.

        Returns:
            value_function (numpy array): the state-value function for the given policy.

        Notes:
            We use `np.linalg.solve`, more stable than using the inverse.
        """

        transition = self.transition_policy(policy)

        return np.linalg.solve(
            np.identity(self.n ** 2) - self.discount * transition,
            transition @ self.reward.reshape(-1)
        ).reshape(self.n, self.n)

    def save_true_value(self, policy):
        if self.save_values:
            true_value = self.compute_exact_values(policy)
            self.training_values.append((true_value[-1, 0], true_value[-1, -1]))

    def backup(self, v_old):
        """
        Performs a backup operation, for all possible policies. Returns a value function tensor.

        Args:
            v_old: the value function to update.

        Returns:
            full_value_function (numpy array): the value function tensor
            v[a][i, j] = "updated value at state i,j if we take action a"
        """

        return np.array([
            transition @ (self.reward.reshape(-1) + self.discount * v_old) for transition in self.transitions
        ])

    def backup_policy(self, v_old, transition):
        """
        Performs a backup operation, for a given policy. Returns a value function matrix.

        Args:
            v_old: the value function to update.
            transition: the transition probability matrix associated to the policy to consider.

        Returns:
            value_function (numpy array): the value function matrix
            `v[i, j] = "updated value at state i,j for the policy"`
        """

        return transition @ (self.reward.reshape(-1) + self.discount * v_old)

    def greedy_policy(self, value):
        values = self.backup(value)
        return values.max(axis=0), values.argmax(axis=0).reshape(self.n, self.n)

    @utils.description('Running value iteration')
    def value_iteration(self, threshold=.001):

        # We initialize the state value function with 0.
        v_new = np.zeros(self.n ** 2)

        error = 2 * threshold

        # We begin the iteration
        while error > threshold:
            v_old = v_new.copy()

            v_new, policy = self.greedy_policy(v_old)

            self.save_true_value(policy)

            error = np.absolute(v_new - v_old).max()

        return v_new.reshape(self.n, self.n), policy

    def policy_evaluation(self, policy, v_old=None, max_iter=1000, threshold=.001):

        # We initialize the state value function with 0.
        if v_old is not None:
            v_new = v_old.copy()
        else:
            v_new = np.zeros(self.n ** 2)

        transition = self.transition_policy(policy)

        # We begin the iteration
        for _ in range(max_iter):
            v_old = v_new.copy()
            v_new = self.backup_policy(v_old, transition)

            if np.absolute(v_new - v_old).max() < threshold:
                break

        return v_new.reshape(self.n, self.n)

    @utils.description('Running policy iteration')
    def policy_iteration(self, threshold=.001):

        # We initialise with a dumb policy, eg always north
        policy = np.zeros((self.n, self.n), dtype=int)

        v_new = self.policy_evaluation(policy, threshold=threshold).reshape(-1)

        error = 2 * threshold

        while error > threshold:
            v_old = v_new.copy()
            _, policy = self.greedy_policy(value=v_new)
            v_new = self.policy_evaluation(policy, threshold=threshold).reshape(-1)

            self.save_true_value(policy)

            error = np.absolute(v_new - v_old).max()

        return v_new.reshape(self.n, self.n), policy

    @utils.description('Running modified policy iteration')
    def modified_policy_iteration(self, threshold=.001, max_iter=20):

        # We initialise with a dumb policy, eg always north
        policy = np.zeros((self.n, self.n), dtype=int)

        v_new = self.policy_evaluation(policy, threshold=threshold).reshape(-1)

        error = 2 * threshold

        while error > threshold:
            v_old = v_new.copy()
            _, policy = self.greedy_policy(value=v_new)
            v_new = self.policy_evaluation(policy, v_old=v_old, threshold=threshold, max_iter=max_iter).reshape(-1)

            self.save_true_value(policy)

            error = np.absolute(v_new - v_old).max()

        return v_new.reshape(self.n, self.n), policy

    def get_values(self):

        values = np.array(self.training_values)
        self.training_values = []
        return values


if __name__ == '__main__':
    # For debugging purposes

    gridworld = GridWorld(5)
    gridworld.modified_policy_iteration()

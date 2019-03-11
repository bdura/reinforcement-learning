# Dynamic Programming

Here we show an example of dynamic programming applied to solving a grid world environment, in which we know the dynamics.

In this example, we use a square grid world with two terminal states which respectively yield a reward upon arrival of:

* `1` in the top left corner
* `10` in the top right corner.

Every other transition yields a reward of `-1`.

The dynamics of the grid world are such that when the agent takes an action, it is performed with probability 0.9, otherwise an other action is taken. Possible actions are going North, West, South or East. If the action effectively performed takes the agent outside the grid world, then the latter stays in place.

Take a look at the [notebook](../notebooks/DP — Gridworld.ipynb) to see the training loop.

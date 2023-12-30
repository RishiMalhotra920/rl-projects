# About

![CliffWalking](https://github.com/RishiMalhotra920/rl-projects/blob/master/images/Cliffwalking.png)


This is me experimenting with various algorithms in the Reinforcement Learning Book by Richard Sutton.
I chose the Cliffwalking-v0 environment in gymnasium to try out these algorithms

I experimented with n-step Q Learning, n-step Sarsa and Monte Carlo methods. I used some basic graphing analysis
techniques to guide my hyperparameter search and reward modelling. One interesting thing was that the agent would go in loops; so to disincetivize this, I introduced a large negative reward if the agent revisited a state.

The concepts are crystal clear to me now although the one-off errors when implementing the
Q Learning and Sarsa Algorithms were very very annoying.

I don't guarantee that these implementations are fully correct and follow the specficiation. There may be one-off errors but it was enough to solve this toy problem.

I was able to achieve a maximum reward of 18 steps. The optimal trajectory would be 14 steps. With some more hyperparameter
search and reward modelling, I could have gotten there.

# To run

pip install requirements.txt
python main.py

# Experiments

As you can see all three algorithms converged. Although all three algorithms could have been improved, Q Learning was particularly more effective.

![Experiments](https://github.com/RishiMalhotra920/rl-projects/blob/master/images/Experiments.png)


![ExperimentsUpClose](https://github.com/RishiMalhotra920/rl-projects/blob/master/images/ExperimentsUpClose.png)

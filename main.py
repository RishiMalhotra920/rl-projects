import gymnasium as gym
import cv2
import argparse
from MonteCarloAgent import MonteCarloAgent
from TDAgent import TDAgent
import matplotlib.pyplot as plt
import pandas as pd


def smoothen(rewards):
    window = 100
    return pd.Series(rewards).rolling(window=window, min_periods=1).mean()


def main(isRender=False):
    # Create the CliffWalking environment
    env = gym.make('CliffWalking-v0', render_mode='rgb_array')

    states = [num for num in range(env.observation_space.n)]
    actions = [num for num in range(env.action_space.n)]

    # To test monte carlo, uncomment the following lines
    monte_carlo_agent = MonteCarloAgent(env, states, actions, isRender)
    monte_carlo_rewards = monte_carlo_agent.train(
        num_episodes=500, episode_length=100, gamma=0.5, alpha=0.01, epsilon=0.5, epsilon_decay=0.99, isRender=isRender)
    # monte_carlo_agent.test(
    # epsilon=0.0, max_episode_length=100, isRender=True)

    # To test sarsa, uncomment the following lines
    td_agent = TDAgent(env, states, actions, isRender)
    sarsa_rewards_by_episode = td_agent.train(
        algorithm="sarsa", num_episodes=500, episode_length=100, n=5, gamma=0.9, alpha=0.1, epsilon=0.5, epsilon_decay=0.99, isRender=isRender)
    # td_agent.test(
    # epsilon=0.0, max_episode_length=100, isRender=True)

    td_agent = TDAgent(env, states, actions, isRender)
    qlearning_rewards = td_agent.train(
        algorithm="qlearning", num_episodes=500, episode_length=100, n=5, gamma=0.99, alpha=0.1, epsilon=0.2, epsilon_decay=0.99, isRender=isRender)

    td_agent.test(
        epsilon=0.0, max_episode_length=100, isRender=True)

    plt.plot(smoothen(monte_carlo_rewards), label="monte carlo")
    plt.plot(smoothen(sarsa_rewards_by_episode), label="sarsa n=5")
    plt.plot(smoothen(qlearning_rewards), label="qlearning n=5")
    plt.title("Reward by Episode")
    plt.legend()
    plt.show()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    args = parser.parse_args()

    main(isRender=args.render)

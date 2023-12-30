import random
import cv2
import time


class MonteCarloAgent:
    def __init__(self, env, states, actions, isRender):
        self.env = env
        self.states = states
        self.actions = actions

        self.Q = {(state, action): 0 for state in states for action in actions}

    def pi(self, state, epsilon):
        # with epsilon probability, pick a random action
        # with 1 - epsilon probability, pick the action with the highest Q value

        if random.random() < epsilon:
            return random.choice(self.actions)

        best_action = None
        for action in self.actions:
            if best_action is None:
                best_action = action
            elif self.Q[(state, action)] > self.Q[(state, best_action)]:
                best_action = action

        return best_action

    def generate_episode(self, episode_length, epsilon, isRender):
        state, _ = self.env.reset()
        action = self.pi(state, epsilon)
        episode = [(None, state, action)]
        episode_states_visited = set()
        for t in range(episode_length):
            next_state, reward, done, info, prob_dict = self.env.step(action)
            action = self.pi(next_state, epsilon)
            if next_state in episode_states_visited:
                reward = -100  # same as falling off cliff
            episode.append((reward, next_state, action))
            episode_states_visited.add(next_state)

            self.render(isRender)
            if done:
                break

        return episode

    def update_Q_pi(self, episode, gamma, alpha):
        G = 0
        # make a dictionary from s,a pair to the first time it appears in the episode
        first_occurence = {}
        for t in range(len(episode)):
            _, state, action = episode[t]
            if (state, action) not in first_occurence:
                first_occurence[(state, action)] = t

        for t in range(len(episode)-1, 0, -1):
            _, state, action = episode[t-1]  # state, action from i-1
            if first_occurence[(state, action)] != t-1:
                continue
            reward, _, _ = episode[t]  # reward from i
            # add negative reward if state revisited

            G = G * gamma + reward
            self.Q[(state, action)] = self.Q[(state, action)] + \
                alpha * (G - self.Q[(state, action)])

        return G

    def train(self, num_episodes, episode_length, gamma, alpha, epsilon, epsilon_decay, isRender):

        reward_by_episode = []
        for t in range(num_episodes):
            # Generate an episode using pi
            episode = self.generate_episode(episode_length, epsilon, isRender)

            # Update Q and pi
            G = self.update_Q_pi(episode, gamma, alpha)

            reward_by_episode.append(
                sum([reward for reward, _, _ in episode[1:]]))

            print(f"--------------------------------------------")
            print(f"{t}: Episode Length: {len(episode)} | G: {G}")
            print(f"Episode: {episode}")
            print(f"Q: {self.Q}")
            # print(f"Q: {self.Q}")
            epsilon = epsilon * epsilon_decay

        self.stopRender(isRender)
        return reward_by_episode

    def test(self, epsilon, max_episode_length, isRender):
        state, _ = self.env.reset()
        action = self.pi(state, epsilon)
        episode = [(None, state, action)]
        for t in range(max_episode_length):
            next_state, reward, done, info, prob_dict = self.env.step(action)
            action = self.pi(next_state, epsilon)
            episode.append((reward, next_state, action))

            self.render(isRender)
            time.sleep(0.2)
            if done:
                break

        self.stopRender(isRender)
        print(f"Test Episode Length: {len(episode)}")
        print(f"episode:{episode}")

        return episode

    def render(self, isRender):
        # Render the environment to visualize
        if isRender:
            frame = self.env.render()
            if frame is not None:
                cv2.imshow('CliffWalking', frame)
                cv2.waitKey(1)  # Refresh the display

    def stopRender(self, isRender):
        if isRender:
            cv2.destroyAllWindows()

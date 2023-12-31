import random
import cv2
import time


class TDAgent:
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

    def update_Q(self, algorithm, episode_length, n, gamma, alpha, epsilon, isRender):
        state, _ = self.env.reset()

        action = self.pi(state, epsilon)
        states = [state]
        actions = [action]
        rewards = [None]
        episode_states_visited = set({state})

        T = episode_length  # episode ends at T-1

        t = 0
        while True:  # stop when tau = T-1 but looping in terms of t

            if t < T:
                state, reward, done, info, prob_dict = self.env.step(action)

                action = self.pi(state, epsilon)

                if state in episode_states_visited:
                    reward = -200

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                episode_states_visited.add(state)

                if done:
                    T = t + 1

            tau = t - n + 1

            if tau >= 0:
                G = sum([gamma**i * rewards[i]
                        for i in range(tau+1, min(tau+n, T-1) + 1)])
                # print(rewards[tau+1])
                # # print(t, tau, [gamma**i * rewards[i]
                #                for i in range(tau+1, min(tau+n, T-1) + 1)])
                if tau + n < T:
                    # sarsa update
                    if algorithm == 'sarsa':
                        G = G + gamma**n * \
                            self.Q[(states[tau+n], actions[tau+n])]
                    elif algorithm == 'qlearning':
                        # Q learning update
                        G = G + gamma**n * max([self.Q[(states[tau+n], action)]
                                                for action in self.actions])

                self.Q[(states[tau], actions[tau])] = self.Q[(states[tau], actions[tau])] + \
                    alpha * (G - self.Q[(states[tau], actions[tau])])

            self.render(isRender)

            t += 1
            # print('this is tau', tau)
            if done or tau == T - 1:
                break

        return states, actions, rewards

    def train(self, algorithm, num_episodes, episode_length, n, gamma, alpha, epsilon, epsilon_decay, isRender):

        reward_by_episode = []
        for t in range(num_episodes):

            states, actions, rewards = self.update_Q(
                algorithm, episode_length, n, gamma, alpha, epsilon, isRender)

            reward_by_episode.append(sum(rewards[1:]))

            print(f"--------------------------------------------")
            print(f"{t}: Episode Length: {len(states)}")
            # print(f"Episode: {episode}")
            # print(f"Q: {self.Q}")
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
            time.sleep(0.5)
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

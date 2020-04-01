#  MIT License
#
#  Copyright (c) 2020 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import numpy as np

from collections import deque


class DQN:

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

        # get the default brain
        self.brain_name = env.brain_names[0]

    def train(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

        scores = []
        scores_last = deque(maxlen=100)
        eps = eps_start
        solved = False

        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for t in range(max_t):
                action = self.agent.act(state, eps)
                env_info = self.env.step(int(action))[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                self.agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if done:
                    break

            scores_last.append(score)
            scores.append(score)

            eps = max(eps_end, eps_decay * eps)

            if i_episode % 100 == 0:
                print("\rEpisode {} average score: {:.2f}".format(i_episode, np.mean(scores_last)))

            if np.mean(scores_last) > 13.0 and not solved:
                print("Environment solved in {} episodes. Average score of last 100 episodes: {:.2f}".format(
                    i_episode, np.mean(scores_last)
                ))
                solved = True

            # Save state.
            self.agent.save("checkpoint.pth")

        return scores

    def test(self, games=100, max_t=1000):

        scores = []

        for i_game in range(1, games + 1):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for t in range(max_t):
                action = self.agent.act(state)
                env_info = self.env.step(int(action))[self.brain_name]

                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                score += reward
                state = next_state

                if done:
                    break

            scores.append(score)

        return np.mean(scores)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state).cpu()
        action = np.random.choice(self.action_size, p=probs.detach().numpy().squeeze())
        return action

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def learn(self):
        states, actions, rewards = zip(*self.memory)
        discounts = np.array([0.99**i for i in range(len(rewards))])
        returns = np.array(
            [
                sum(rewards[i:] * discounts[: len(rewards) - i])
                for i in range(len(rewards))
            ]
        )

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        returns = torch.tensor(returns, dtype=torch.float)

        self.optimizer.zero_grad()
        log_probs = torch.log(self.policy_network(states))
        selected_log_probs = returns * log_probs[np.arange(len(actions)), actions]
        loss = -selected_log_probs.mean()
        loss.backward()
        self.optimizer.step()
        self.memory = []


def train():
    env = gymnasium.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size, learning_rate=0.001)

    n_episodes = 1000
    max_t = 1000
    print_interval = 100
    episode_rewards = []

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_rewards = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            agent.remember(state, action, reward)
            state = next_state
            if terminated or truncated:
                break

        agent.learn()
        episode_rewards.append(total_rewards)
        if i_episode % print_interval == 0:
            print(f"Episode {i_episode}/{n_episodes} - Score: {total_rewards}")

            torch.save(
                agent.policy_network.state_dict(),
                f"checkpoints/reinforce_model_{i_episode}.pt",
            )

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes")
    plt.savefig("reinforce_rewards.png")


def test():
    env = gymnasium.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size, learning_rate=0.001)

    last_episode = 1000
    agent.policy_network.load_state_dict(
        torch.load(f"checkpoints/reinforce_model_{last_episode}.pt")
    )

    for _ in range(5):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            if terminated or truncated:
                break
            state = next_state
        print(f"Test Completed! Total Rewards: {total_rewards}")
    env.close()


if __name__ == "__main__":
    train()
    test()

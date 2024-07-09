import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, clip_epsilon=0.2, update_every=4, k_epochs=4, entropy_coeff=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_every = update_every
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.memory = []

    def act(self, state, sample=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state).cpu()
        if sample:
            action = np.random.choice(self.action_size, p=probs.detach().numpy().squeeze())
        else:
            action = probs.argmax(dim=-1).item()
        return action, probs[0, action].item()

    def step(self, state, action, reward, log_prob, done):
        self.memory.append((state, action, reward, log_prob, done))

    def learn(self):
        states, actions, rewards, log_probs, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        log_probs = torch.tensor(log_probs, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        returns = []
        Gt = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                Gt = 0
            Gt = reward + self.gamma * Gt
            returns.insert(0, Gt)
        returns = torch.tensor(returns, dtype=torch.float)

        for _ in range(self.k_epochs):
            values = self.value_network(states).squeeze()
            advantages = returns - values.detach()
            new_log_probs = torch.log(self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze())
            ratios = torch.exp(new_log_probs - log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = -torch.sum(self.policy_network(states) * torch.log(self.policy_network(states) + 1e-10), dim=1).mean()
            policy_loss -= self.entropy_coeff * entropy

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            value_loss = F.mse_loss(values, returns)
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        self.memory = []


def train():
    env = gymnasium.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(state_size, action_size, learning_rate=0.001, gamma=0.99, clip_epsilon=0.2, update_every=4, k_epochs=4, entropy_coeff=0.01)

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
            action, log_prob = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            done = terminated or truncated
            agent.step(state, action, reward, log_prob, done)
            state = next_state
            if done:
                break

        if i_episode % agent.update_every == 0:
            agent.learn()

        episode_rewards.append(total_rewards)
        if i_episode % print_interval == 0:
            print(f"Episode {i_episode}/{n_episodes} - Score: {total_rewards}")

            torch.save(agent.policy_network.state_dict(), f"checkpoints/ppo_policy_model_{i_episode}.pt")
            torch.save(agent.value_network.state_dict(), f"checkpoints/ppo_value_model_{i_episode}.pt")

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('ppo_rewards.png')


def test():
    env = gymnasium.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(state_size, action_size, learning_rate=0.001, gamma=0.99, clip_epsilon=0.2, update_every=4, k_epochs=4, entropy_coeff=0.01)

    last_episode = 1000
    agent.policy_network.load_state_dict(torch.load(f"checkpoints/ppo_policy_model_{last_episode}.pt"))
    agent.value_network.load_state_dict(torch.load(f"checkpoints/ppo_value_model_{last_episode}.pt"))

    for _ in range(5):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            env.render()
            action, _ = agent.act(state)
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

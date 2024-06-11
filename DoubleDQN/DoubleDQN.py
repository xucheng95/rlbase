import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            np.vstack(states),
            np.vstack(actions),
            np.vstack(rewards),
            np.vstack(next_states),
            np.vstack(dones),
        )

    def __len__(self):
        return len(self.memory)


class DoubleDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        tau,
        update_every
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()

        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)

        Q_targets_next = (
            self.qnetwork_target(next_states).gather(1, next_actions).detach()
        )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


def train():
    env = gymnasium.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleDQNAgent(
        state_size,
        action_size,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.01,
        update_every=4
    )

    n_episodes = 1000
    max_t = 1000
    print_interval = 100
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    episode_rewards = []

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints") 

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_rewards = 0
        for t in range(max_t):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_rewards)
        if i_episode % print_interval == 0:
            print(
                f"Episode {i_episode}/{n_episodes} - Score: {total_rewards} - Epsilon: {epsilon:.2f}"
            )
            
            torch.save(agent.qnetwork_local.state_dict(), f"checkpoints/doubledqn_model_{i_episode}.pt")

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('doubledqn_rewards.png')


def test():
    env = gymnasium.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleDQNAgent(state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, learning_rate=0.001, tau=0.01, update_every=4)
    
    last_episode = 1000 
    agent.qnetwork_local.load_state_dict(torch.load(f"checkpoints/doubledqn_model_{last_episode}.pt"))
    
    # 进行测试
    for _ in range(5):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            env.render()
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated , _ = env.step(action)
            total_rewards += reward
            if terminated or truncated:
                break
            state = next_state
        print(f"Test Completed! Total Rewards: {total_rewards}")
    env.close()

if __name__ == "__main__":
    train()
    test()


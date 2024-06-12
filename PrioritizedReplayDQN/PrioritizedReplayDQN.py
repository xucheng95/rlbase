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


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.ptr = 0

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            else:
                if value <= self.tree[left]:
                    parent = left
                else:
                    value -= self.tree[left]
                    parent = right
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha

    def add(self, error, sample):
        priority = (error + 1e-5) ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.size * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights

    def update(self, idx, error):
        priority = (error + 1e-5) ** self.alpha
        self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)


class PrioritizedDQNAgent:
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, learning_rate, tau, update_every, alpha, beta_start, beta_frames):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.t_step = 0
        self.frame_idx = 0
    
    def step(self, state, action, reward, next_state, done):
        self.frame_idx += 1
        self.memory.add(abs(reward), (state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
            self.learn(experiences, idxs, is_weights)
            self.beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
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
    
    def learn(self, experiences, idxs, is_weights):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        is_weights = torch.from_numpy(np.vstack(is_weights)).float()
        
        # 使用当前Q网络选择下一个动作
        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        
        # 使用目标Q网络评估下一个动作的Q值
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        errors = torch.abs(Q_expected - Q_targets).cpu().data.numpy()
        for i in range(len(idxs)):
            self.memory.update(idxs[i], errors[i])
        
        loss = (is_weights * (Q_expected - Q_targets) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def train():
    env = gymnasium.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PrioritizedDQNAgent(
        state_size,
        action_size,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.01,
        update_every=4,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1000
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
            
            torch.save(agent.qnetwork_local.state_dict(), f"checkpoints/prioritized_model_{i_episode}.pt")

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('prioritized_rewards.png')


def test():
    env = gymnasium.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PrioritizedDQNAgent(
        state_size,
        action_size,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.01,
        update_every=4,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1000
    )
    
    last_episode = 1000 
    agent.qnetwork_local.load_state_dict(torch.load(f"checkpoints/prioritized_model_{last_episode}.pt"))
    
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


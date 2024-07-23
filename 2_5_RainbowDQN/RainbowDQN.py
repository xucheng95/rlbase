import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.value_stream = NoisyLinear(128, 1)
        self.advantage_stream = NoisyLinear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean())
        return q_values

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()


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


class RainbowAgent:
    def __init__(
        self,
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
        beta_frames=1000,
        n_step=3,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.n_step = n_step

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.n_step_memory = deque(maxlen=n_step)
        self.t_step = 0
        self.frame_idx = 0

    def select_action(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def remember(self, state, action, reward, next_state, done):
        self.frame_idx += 1
        self.n_step_memory.append((state, action, reward, next_state, done))

        if len(self.n_step_memory) == self.n_step or done:
            reward, next_state, done = self.compute_n_step_return()
            self.memory.add(abs(reward), (state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences, idxs, is_weights = self.memory.sample(
                self.batch_size, self.beta
            )
            self.learn(experiences, idxs, is_weights)
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()
            self.beta = min(
                1.0,
                self.beta_start
                + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames,
            )

    def compute_n_step_return(self):
        R = 0
        for i in range(self.n_step):
            _, _, reward, _, done = self.n_step_memory[i]
            R += (self.gamma**i) * reward
            if done:
                break
        next_state = self.n_step_memory[-1][3]
        done = self.n_step_memory[-1][4]
        return R, next_state, done

    def learn(self, experiences, idxs, is_weights):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        is_weights = torch.from_numpy(np.vstack(is_weights)).float()
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
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
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path, name):
        """
        将当前DQN模型的参数保存到指定路径下。

        Args:
            path (str): 保存模型的路径。
            name (str): 模型的名称，用于生成保存文件的名称。

        Returns:
            None

        """
        torch.save(self.qnetwork_local.state_dict(), f"{path}/rainbow_model_{name}.pt")

    def load(self, path, name):
        """
        从指定路径下加载DQN模型的参数，并更新当前DQN模型。
        Args:
            path (str): 保存模型的路径。
            name (str): 模型的名称，用于生成保存文件的名称。
        Returns:
            None

        """
        self.qnetwork_local.load_state_dict(
            torch.load(f"{path}/rainbow_model_{name}.pt")
        )

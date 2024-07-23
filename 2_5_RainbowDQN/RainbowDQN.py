import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        """
        初始化函数。
        
        Args:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            sigma_init (float, optional): 初始化标准差的值。默认为 0.017。
        
        Returns:
            None
        
        """
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
        """
        重置网络层的参数。

        Args:
            无

        Returns:
            无返回值，直接修改对象的内部状态。

        """
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """
        重置噪声项。
        
        Args:
            无。
        
        Returns:
            None: 该函数没有返回值，直接修改了对象的属性。
        
        """
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        """
        在训练过程中应用加权的权重和偏置进行线性变换，否则只使用平均权重和偏置。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。
        
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)，其中 output_dim 是权重矩阵的第二维的大小。
        
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化函数，用于设置DQN网络结构。
        
        Args:
            state_size (int): 状态空间的维度大小。
            action_size (int): 动作空间的维度大小。
        
        Returns:
            None
        
        """
        super().__init__()
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.value_stream = NoisyLinear(128, 1)
        self.advantage_stream = NoisyLinear(128, action_size)

    def forward(self, x):
        """
        前向传播函数，用于计算Q值。
        
        Args:
            x (torch.Tensor): 输入的Tensor，形状为(batch_size, input_dim)。
        
        Returns:
            torch.Tensor: 输出的Q值Tensor，形状为(batch_size, num_actions)。
        
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean())
        return q_values

    def reset_noise(self):
        """
        重置模型中的噪声。
        
        Args:
            无。
        
        Returns:
            无返回值，该函数用于更新模型中的噪声参数。
        
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()


class SumTree:
    def __init__(self, capacity):
        """
        初始化一个完全二叉树。
        
        Args:
            capacity (int): 二叉树的最大容量。
        
        Attributes:
            capacity (int): 二叉树的最大容量。
            tree (np.ndarray): 用于存储二叉树的数组，大小为2*capacity-1。
            data (np.ndarray): 用于存储实际数据的数组，大小为capacity，数据类型为object。
            size (int): 当前二叉树中元素的数量。
            ptr (int): 指向下一个空闲位置的指针，用于在二叉树中插入新元素。
        
        Returns:
            None: 该函数没有返回值，但会初始化一个二叉堆对象。
        
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.ptr = 0

    def add(self, priority, data):
        """
        在二叉树中添加一个数据元素，并更新二叉树的结构以保持其性质。
        
        Args:
            priority (int): 元素的优先级
            data (Any): 元素的数据
        
        Returns:
            None
        
        """
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        """
        更新二叉树中指定索引位置元素的优先级，并重新调整二叉树的结构以保持其性质。
        
        Args:
            idx (int): 二叉树中需要更新优先级的元素的索引。
            priority (int): 更新后的优先级值。
        
        Returns:
            None
        
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, value):
        """
        根据给定的值value在二叉树中找到对应的叶子节点并返回相关信息。
        
        Args:
            value (int): 需要查找的值。
        
        Returns:
            tuple: 包含三个元素的元组，分别代表：
                - leaf (int): 叶子节点在二叉搜索树中的索引。
                - prefix_sum (int): 叶子节点对应的累积和（前缀和）。
                - data (Any): 叶子节点对应的原始数据。
        
        """
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
        """
        初始化一个经验回放优先级队列。
        
        Args:
            capacity (int): 队列的容量，即最多能存储的经验数量。
            alpha (float): 用于计算优先级的指数参数，alpha值越大，优先级差异越显著。
        
        Returns:
            None
        
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha

    def add(self, error, sample):
        """
        将样本添加到优先队列中，并根据误差计算优先级。
        
        Args:
            error (float): 样本的误差值。
            sample (Any): 要添加到优先队列中的样本。
        
        Returns:
            None
        
        """
        priority = (error + 1e-5) ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta):
        """
        从经验回放缓冲区中采样一批数据。
        
        Args:
            batch_size (int): 采样批次大小。
            beta (float): 用于计算重要性采样权重的参数。
        
        Returns:
            tuple: 包含三个元素的元组，分别为：
                - batch (list): 采样得到的数据列表。
                - idxs (list): 与batch中数据对应的索引列表。
                - is_weights (np.ndarray): 重要性采样权重数组。
        
        """
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
        """
        更新索引idx的优先级，并将其在树中的位置重新排列。
        
        Args:
            idx (int): 要更新的索引值。
            error (float): 用于计算优先级的误差值。
        
        Returns:
            None: 此函数没有返回值，但会更新树中的元素。
        
        """
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
        """
        根据当前状态选择动作。

        Args:
            state (numpy.ndarray): 当前环境的状态，形状为 (state_size,)。
            epsilon (float, optional): 用于决定选择随机动作的概率。默认为0.0，表示选择概率最大的动作。

        Returns:
            int: 选择的动作编号，范围在 [0, action_size) 之间。

        """
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
        """
        将经验存储到经验池中，并在满足条件时开始DQN的学习。
        
        Args:
            state (np.ndarray): 当前状态。
            action (int): 当前动作。
            reward (float): 获得的奖励。
            next_state (np.ndarray): 下一个状态。
            done (bool): 是否为终止状态。
        
        Returns:
            None
        
        """
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
        """
        计算n步累积回报。
        
        Args:
            无。
        
        Returns:
            tuple: 包含三个元素的元组，表示n步累积回报、下一个状态以及是否结束。
            - R (float): n步累积回报，即前n步的奖励按照gamma的折扣率累积得到的值。
            - next_state (Any): 下一个状态，即n步记忆序列中的最后一个状态。
            - done (bool): 是否结束，即n步记忆序列中的最后一个状态是否为终止状态。
        
        """
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
        """
        用经验池中随机抽取的一批经验进行学习

        Args:
            experiences (tuple): 包含状态的列表、动作的列表、奖励的列表、下一个状态的列表、是否终止的列表
            idxs (list): 抽取经验的索引列表
            is_weights (list): 每个经验的重要性权重列表

        Returns:
            None

        """
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
        """
        使用soft update规则更新target_model的参数。

        Args:
            local_model (nn.Module): 本地模型，其参数将被用于更新target_model。
            target_model (nn.Module): 目标模型，其参数将被更新。

        Returns:
            None: 该函数直接修改target_model的参数，不返回任何值。

        Note:
            该函数实现了soft update规则，该规则通常用于深度强化学习中的目标网络更新。
            具体来说，对于target_model中的每个参数target_param和local_model中的对应参数local_param，
            target_param的值会被更新为tau * local_param + (1 - tau) * target_param，
            其中tau是一个介于0和1之间的超参数，控制更新的平滑程度。

            这样做的好处是可以使得target_model的更新更加稳定，避免因为local_model的剧烈变化而导致训练不稳定。
        """
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

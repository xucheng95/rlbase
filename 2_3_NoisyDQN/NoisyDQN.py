import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.017):
        """
        初始化NoisyLinear层。

        Args:
            in_features (int): 输入特征的数量。
            out_features (int): 输出特征的数量。
            std_init (float, optional): 用于初始化权重标准差和偏置标准差的初始值。默认为0.5。

        Returns:
            None

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

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
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """
        重置噪声项。

        Args:
            无参数。

        Returns:
            None: 此函数无返回值，但会更新对象内部属性。

        """
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        """
        对输入数据进行线性变换。

        Args:
            input (Tensor): 输入数据，其形状为 (batch_size, input_size)。

        Returns:
            Tensor: 经过线性变换后的输出数据，其形状为 (batch_size, output_size)。

        在训练模式下，会加上可学习的标准差（weight_sigma 和 bias_sigma）与随机噪声（weight_epsilon 和 bias_epsilon）
        的乘积来模拟参数的不确定性。在非训练模式下，则只使用均值（weight_mu 和 bias_mu）作为权重和偏置。

        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(input, weight, bias)


class NoisyQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化函数，用于创建神经网络。

        Args:
            state_size (int): 状态空间的大小。
            action_size (int): 动作空间的大小。

        Returns:
            None: 无返回值，该函数主要用于初始化类的实例变量。

        """
        super().__init__()
        self.fc1 = NoisyLinear(state_size, 64)
        self.fc2 = NoisyLinear(64, 64)
        self.fc3 = NoisyLinear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        """
        重置噪声项。
        Args:
            无参数。
        Returns:
            None: 该函数不返回任何值，仅修改对象内部状态。
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        初始化一个经验回放（Experience Replay）的缓冲区对象。

        Args:
            buffer_size (int): 经验回放缓冲区能够存储的最大样本数。
            batch_size (int): 批量采样时从缓冲区中抽取的样本数。

        Returns:
            None: 该函数不返回任何值，而是初始化内部属性。

        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        将状态转移数据添加到记忆库中。

        Args:
            state (numpy.ndarray): 当前环境的状态，形状为 (state_size,)。
            action (int): 动作编号，范围在 [0, action_size) 之间。
            reward (float): 代理（Agent）执行动作后从环境（Environment）获得的奖励。
            next_state (numpy.ndarray): 执行动作后转移到的下一个状态，形状为 (state_size,)。
            done (bool): 一个布尔值，指示当前回合是否结束。

        Returns:
            None: 该函数不返回任何值，仅将状态转移数据添加到记忆库中。

        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """
        从经验回放池中随机抽取一批经验用于训练。

        Args:
            无参数。

        Returns:
            tuple: 包含五个numpy数组的元组，分别代表：
                - states (np.ndarray): 形状为 (batch_size, state_size) 的数组，表示状态。
                - actions (np.ndarray): 形状为 (batch_size, action_size) 的数组，表示动作。
                - rewards (np.ndarray): 形状为 (batch_size,) 的一维数组，表示奖励。
                - next_states (np.ndarray): 形状为 (batch_size, state_size) 的数组，表示下一个状态。
                - dones (np.ndarray): 形状为 (batch_size,) 的一维布尔数组，表示该经验是否结束了一个回合。

        """
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


class NoisyDQNAgent:
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
    ):
        """
        初始化函数，用于设置Noisy DQN的参数和组件

        Args:
            state_size (int): 环境的状态大小（例如，环境中的状态空间的大小）
            action_size (int): 环境中可采取的动作的数量
            buffer_size (int): 经验回放缓冲区的大小
            batch_size (int): 从经验回放缓冲区中随机抽取的样本数量，用于训练
            gamma (float): 折扣因子（在[0, 1]之间），用于计算长期奖励
            learning_rate (float): 学习率，用于优化器
            tau (float): 用于软更新目标网络的参数
            update_every (int): 每隔多少个步骤更新一次目标网络

        Returns:
            None

        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        self.qnetwork_local = NoisyQNetwork(state_size, action_size)
        self.qnetwork_target = NoisyQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

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
        更新代理（Agent）的经验池并决定何时进行学习。

        Args:
            state (numpy.ndarray): 当前环境的状态，形状为 (state_size,)。
            action (int): 动作编号，范围在 [0, action_size) 之间。
            reward (float): 代理（Agent）执行动作后从环境（Environment）获得的奖励。
            next_state (numpy.ndarray): 执行动作后转移到的下一个状态，形状为 (state_size,)。
            done (bool): 一个布尔值，指示当前回合是否结束。

        Returns:
            None: 该函数没有返回值，但会更新经验池，并在满足条件时进行学习。

        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()

    def learn(self, experiences):
        """
        从经验回放中学习，并更新本地Q网络。

        Args:
            experiences (tuple): 一个包含以下五个元素的元组:
                - states (numpy.ndarray): 状态数组，形状为(batch_size, state_size)。
                - actions (numpy.ndarray): 动作数组，形状为(batch_size,)。
                - rewards (numpy.ndarray): 奖励数组，形状为(batch_size,)。
                - next_states (numpy.ndarray): 下一个状态数组，形状为(batch_size, state_size)。
                - dones (numpy.ndarray): 布尔数组，指示是否达到终止状态，形状为(batch_size,)。

        Returns:
            None

        """
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()

        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = self.loss_fn(Q_expected, Q_targets)
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
        torch.save(self.qnetwork_local.state_dict(), f"{path}/noisydqn_model_{name}.pt")

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
            torch.load(f"{path}/noisydqn_model_{name}.pt")
        )
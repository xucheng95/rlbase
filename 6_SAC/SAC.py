import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化函数，用于创建神经网络。

        Args:
            state_size (int): 状态空间的大小。
            action_size (int): 动作空间的大小。

        Returns:
            None
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, action_size)
        self.fc_log_std = nn.Linear(128, action_size)
 

    def forward(self, x):
        """
        执行网络前向传播。
        
        Args:
            x (torch.Tensor): 输入的Tensor，形状为 (batch_size, input_dim)。
        
        Returns:
            tuple: 包含两个元素的元组，每个元素都是torch.Tensor类型。
                - mean (torch.Tensor): 输出的均值，形状为 (batch_size, output_dim)。
                - std (torch.Tensor): 输出的标准差，形状为 (batch_size, output_dim)。
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制 log_std 的范围
        return mean, log_std
    
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化神经网络结构，包括两个全连接层和一个输出层。
        
        Args:
            state_size (int): 状态空间的维度大小。
            action_size (int): 动作空间的维度大小。
        
        Returns:
            None
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, state, action):
        """
        根据当前状态（state）和动作（action）进行前向传播。
        
        Args:
            state (torch.Tensor): 当前状态，形状为 (batch_size, state_dim)。
            action (torch.Tensor): 当前动作，形状为 (batch_size, action_dim)。
        
        Returns:
            torch.Tensor: 经过前向传播后得到的输出，形状为 (batch_size, output_dim)。
        """
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        初始化一个经验回放优先级队列。

        Args:
            capacity (int): 队列的容量，即最多能存储的经验数量。
            batch_size (int): 每次从队列中抽取的经验数量。

        Returns:
            None
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        将样本添加到队列中。

        Args:
            state (numpy.ndarray): 当前环境的状态，形状为 (state_size,)。
            action (int): 动作编号，范围在 [0, action_size) 之间。
            reward (float): 代理（Agent）执行动作后从环境（Environment）获得的奖励。
            next_state (numpy.ndarray): 执行动作后转移到的下一个状态，形状为 (state_size,)。
            done (bool): 一个布尔值，指示当前回合是否结束。

        Returns:
            None
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """
        从经验回放池队列中随机抽取一批经验用于训练。

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


class SACAgent:
    def __init__(
        self,
        state_size,
        action_size,
        action_range,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.005,
        alpha=0.2,
    ):
        """
        初始化方法，用于初始化智能体的各种参数和组件。
        
        Args:
            state_size (int): 状态空间的维度大小。
            action_size (int): 动作空间的维度大小。
            action_range (tuple): 动作的取值范围，以(low, high)的形式给出。
            buffer_size (int, optional): 经验回放缓冲区的大小。默认为10000。
            batch_size (int, optional): 每次从经验回放缓冲区中采样进行训练的样本数量。默认为64。
            gamma (float, optional): 折扣因子，用于计算累积奖励。默认为0.99。
            learning_rate (float, optional): 学习率，用于优化器。默认为0.001。
            tau (float, optional): 软更新参数，用于目标网络的更新。默认为0.005。
            alpha (float, optional): 优先经验回放的参数，用于计算样本的优先级。默认为0.2。
        
        Returns:
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(state_size, action_size)
        self.critic1 = Critic(state_size, action_size)
        self.critic2 = Critic(state_size, action_size)
        self.target_critic1 = Critic(state_size, action_size)
        self.target_critic2 = Critic(state_size, action_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.update_targets()

    def update_targets(self):
        """
        更新目标网络参数。
        
        Args:
            无参数。
        
        Returns:
            无返回值。
        """
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

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

    def select_action(self, state, deterministic=False):
        """
        根据当前状态选择一个动作。
        
        Args:
            state (numpy.ndarray): 当前环境的状态，形状为(state_dim,)的numpy数组。
            deterministic (bool, optional): 是否选择确定性动作。默认为False，即选择随机动作。
        
        Returns:
            numpy.ndarray: 选择的动作，形状为(action_dim,)的numpy数组。
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        if deterministic:
            action, _ = self.actor(state)
        else:
            action, _ = self.actor.sample(state)
        return self.action_range * action.detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        """
        将经验（state, action, reward, next_state, done）存入记忆库，并在记忆库大小超过批次大小时进行学习。
        
        Args:
            state (object): 代理所处的当前状态。
            action (int): 代理在当前状态下所采取的动作。
            reward (float): 代理在采取动作后所获得的奖励。
            next_state (object): 代理在采取动作后所处的下一个状态。
            done (bool): 是否为终止状态，即是否达到目标或达到最大迭代次数等。
        
        Returns:
            None
        """
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        根据给定的经验列表来更新模型的参数。
        
        Args:
            experiences (tuple): 一个包含五个元素的元组，分别为：
                - states (numpy.ndarray): 形状为 (batch_size, state_dim) 的数组，表示经验中的状态。
                - actions (numpy.ndarray): 形状为 (batch_size, action_dim) 的数组，表示经验中的动作。
                - rewards (numpy.ndarray): 形状为 (batch_size,) 的一维数组，表示经验中的奖励。
                - next_states (numpy.ndarray): 形状为 (batch_size, state_dim) 的数组，表示经验中的下一个状态。
                - dones (numpy.ndarray): 形状为 (batch_size,) 的一维数组，表示经验是否结束（达到终止状态）。
        
        Returns:
            None
        """
        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = self.loss_fn(q1, q_target)
        critic2_loss = self.loss_fn(q2, q_target)
        self.optimizer_critic1.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic1.step()
        self.optimizer_critic2.zero_grad()
        critic2_loss.backward()
        self.optimizer_critic2.step()

        actions_pred, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.update_targets()

    def save(self, path, name):
        """
        将当前模型的参数保存到指定路径下。

        Args:
            path (str): 保存模型的路径。
            name (str): 模型的名称，用于生成保存文件的名称。

        Returns:
            None
        """
        torch.save(self.actor.state_dict(), f"{path}/sac_model_{name}.pt")

    def load(self, path, name):
        """
        从指定路径下加载模型的参数，并更新当前模型。

        Args:
            path (str): 保存模型的路径。
            name (str): 模型的名称，用于生成保存文件的名称。

        Returns:
            None
        """
        self.actor.load_state_dict(torch.load(f"{path}/sac_model_{name}.pt"))

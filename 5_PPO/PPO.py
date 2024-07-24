import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):
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
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        前向传播函数，用于计算动作的概率分布。

        Args:
            x (torch.Tensor): 输入的Tensor，形状为(batch_size, input_dim)。

        Returns:
            torch.Tensor: 输出的动作概率分布，形状为(batch_size, num_actions)。
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        """
        初始化函数，用于创建神经网络。

        Args:
            state_size (int): 状态空间的大小。

        Returns:
            None
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """
        前向传播函数，计算state value。
        
        Args:
            x (torch.Tensor): 输入的Tensor，形状为(batch_size, input_dim)。

        Returns:
            torch.Tensor: 输出的state value，形状为(batch_size, 1)。
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        learning_rate=0.001,
        clip_epsilon=0.2,
        update_every=4,
        k_epochs=4,
        entropy_coeff=0.001,
    ):
        """
        初始化函数，用于设置PPO的参数和组件

        Args:
            state_size (int): 环境的状态大小（例如，环境中的状态空间的大小）
            action_size (int): 环境中可采取的动作的数量
            gamma (float): 折扣因子（在[0, 1]之间），用于计算长期奖励
            learning_rate (float): 学习率，用于优化器
            clip_epsilon (float): 裁剪参数，用于PPO的损失函数
            update_every (int): 更新频率，用于更新网络
            k_epochs (int): 优化器中使用的迭代次数
            entropy_coeff (float): 熵系数，用于计算熵

        Returns:
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_every = update_every
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )
        self.optimizer_value = optim.Adam(
            self.value_network.parameters(), lr=learning_rate
        )
        self.memory = []

    def select_action(self, state, sample=False):
        """
        根据当前状态选择动作。
        
        Args:
            state (np.ndarray): 当前环境的状态，形状为 (state_size,)。
            sample (bool, optional): 是否根据概率采样动作。默认为False，表示选择概率最大的动作。
        
        Returns:
            int: 选择的动作编号，范围在 [0, action_size) 之间。
            float: 动作概率的对数。
        
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state).cpu()
        if not sample:
            action = probs.argmax(dim=-1).item()
        else:
            action = np.random.choice(
                self.action_size, p=probs.detach().numpy().squeeze()
            )
        return action, torch.log(probs[0, action]).item()

    def remember(self, state, action, reward, log_prob, done):
        """
        将经验存储到经验池中，并在满足条件时开始学习。
        
        Args:
            state (np.ndarray): 当前状态。
            action (int): 当前动作。
            reward (float): 获得的奖励。
        
        Returns:
            None
        """
        self.memory.append((state, action, reward, log_prob, done))

    def learn(self):
        """
        根据存储的经验进行策略网络的训练。
        
        Args:
            无参数。
        
        Returns:
            None 
        """
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
            new_log_probs = torch.log(
                self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze()
            )
            ratios = torch.exp(new_log_probs - log_probs)

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = -torch.sum(
                self.policy_network(states)
                * torch.log(self.policy_network(states) + 1e-10),
                dim=1,
            ).mean()
            policy_loss -= self.entropy_coeff * entropy

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            value_loss = F.mse_loss(values, returns)
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        self.memory = []

    def save(self, path, name):
        """
        将当前模型的参数保存到指定路径下。

        Args:
            path (str): 保存模型的路径。
            name (str): 模型的名称，用于生成保存文件的名称。

        Returns:
            None
        """
        torch.save(
            self.policy_network.state_dict(), f"{path}/ppo_model_{name}.pt"
        )

    def load(self, path, name):
        """
        从指定路径下加载模型的参数，并更新当前模型。

        Args:
            path (str): 保存模型的路径。
            name (str): 模型的名称，用于生成保存文件的名称。
        
        Returns:
            None
        """
        self.policy_network.load_state_dict(
            torch.load(f"{path}/ppo_model_{name}.pt")
        )

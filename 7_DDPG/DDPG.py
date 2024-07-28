import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化函数，用于创建神经网络。
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_size)

    def forward(self, x):
        """
        执行网络前向传播。
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc_out(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化神经网络结构，包括两个全连接层和一个输出层。
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, state, action):
        """
        根据当前状态（state）和动作（action）进行前向传播。
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
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        将样本添加到队列中。
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """
        从经验回放池队列中随机抽取一批经验用于训练。
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

class DDPGAgent:
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
    ):
        """
        初始化方法，用于初始化智能体的各种参数和组件。
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.update_targets()

    def update_targets(self):
        """
        更新目标网络参数。
        """
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, local_model, target_model):
        """
        使用soft update规则更新target_model的参数。
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def select_action(self, state, noise_scale=0.1):
        """
        根据当前状态选择一个动作。
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise_scale * np.random.randn(self.action_size)  # 添加噪声
        action = np.clip(action, -self.action_range, self.action_range)
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        将经验（state, action, reward, next_state, done）存入记忆库，并在记忆库大小超过批次大小时进行学习。
        """
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        根据给定的经验列表来更新模型的参数。
        """
        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            q_target_next = self.target_critic(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * q_target_next

        q_expected = self.critic(states, actions)
        critic_loss = self.loss_fn(q_expected, q_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.update_targets()

    def save(self, path, name):
        """
        将当前模型的参数保存到指定路径下。
        """
        torch.save(self.actor.state_dict(), f"{path}/ddpg_actor_{name}.pt")
        torch.save(self.critic.state_dict(), f"{path}/ddpg_critic_{name}.pt")

    def load(self, path, name):
        """
        从指定路径下加载模型的参数，并更新当前模型。
        """
        self.actor.load_state_dict(torch.load(f"{path}/ddpg_actor_{name}.pt"))
        self.critic.load_state_dict(torch.load(f"{path}/ddpg_critic_{name}.pt"))
        self.update_targets()

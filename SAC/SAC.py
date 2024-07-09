import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium
import matplotlib.pyplot as plt
from torch.distributions import Normal

# SAC Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_size)
        self.fc_log_std = nn.Linear(256, action_size)
        self.action_range = action_range

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x)) * self.action_range
        log_std = torch.tanh(self.fc_log_std(x))
        std = torch.exp(log_std)
        return mean, std

# SAC Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

# SAC Agent
class SACAgent:
    def __init__(self, state_size, action_size, action_range, buffer_size, batch_size, gamma, tau, lr, alpha):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha

        self.actor = Actor(state_size, action_size, action_range)
        self.critic1 = Critic(state_size, action_size)
        self.critic2 = Critic(state_size, action_size)
        self.target_critic1 = Critic(state_size, action_size)
        self.target_critic2 = Critic(state_size, action_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.update_targets()

    def update_targets(self):
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.actor(state)
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        return action.clamp(-self.action_range, self.action_range).detach().numpy()[0]

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(q1, q_target)
        critic2_loss = nn.MSELoss()(q2, q_target)

        self.optimizer_critic1.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        critic2_loss.backward()
        self.optimizer_critic2.step()

        actions_pred, log_probs = self.actor(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.update_targets()

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

def train():
    env = gymnasium.make("Pendulum-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = env.action_space.high[0]
    agent = SACAgent(
        state_size,
        action_size,
        action_range,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr=0.0003,
        alpha=0.2
    )

    n_episodes = 1000
    max_t = 200
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
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        episode_rewards.append(total_rewards)
        if i_episode % print_interval == 0:
            print(f"Episode {i_episode}/{n_episodes} - Score: {total_rewards}")
            torch.save(agent.actor.state_dict(), f"checkpoints/sac_actor_{i_episode}.pt")
            torch.save(agent.critic1.state_dict(), f"checkpoints/sac_critic1_{i_episode}.pt")
            torch.save(agent.critic2.state_dict(), f"checkpoints/sac_critic2_{i_episode}.pt")

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('sac_rewards.png')


def test():
    env = gymnasium.make("Pendulum-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = env.action_space.high[0]
    agent = SACAgent(
        state_size,
        action_size,
        action_range,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr=0.0003,
        alpha=0.2
    )
    
    last_episode = 1000
    agent.actor.load_state_dict(torch.load(f"checkpoints/sac_actor_{last_episode}.pt"))
    agent.critic1.load_state_dict(torch.load(f"checkpoints/sac_critic1_{last_episode}.pt"))
    agent.critic2.load_state_dict(torch.load(f"checkpoints/sac_critic2_{last_episode}.pt"))

    for _ in range(5):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            env.render()
            action = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            if terminated or truncated:
                break
            state = next_state
        print(f"Test Completed! Total Rewards: {total_rewards}")
    env.close()

if __name__ == "__main__":
    # train()
    test()

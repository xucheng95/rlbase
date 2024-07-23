# RLBase

RLBase 是一个用于深度强化学习的开源项目，包含了常见的深度强化学习算法，每个算法都可以单独运行，适用于新手学习理解算法。

## 目录

- [RLBase](#rlbase)
  - [目录](#目录)
  - [简介](#简介)
  - [安装](#安装)
  - [使用方法](#使用方法)
  - [实现的算法](#实现的算法)
  - [许可证](#许可证)

## 简介

RLBase 项目旨在为研究人员和开发人员提供一个简单易用的框架，以便快速实验和测试各种深度强化学习算法。项目包含以下特性：

- 常见深度强化学习算法的实现
- 每个算法可以单独运行

## 安装

1. 克隆此仓库：
    ```bash
    git clone https://github.com/xucheng95/rlbase.git
    cd rlbase
    ```

2. 创建并激活虚拟环境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`
    ```

3. 安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1. 选择算法并运行主脚本：
    ```bash
    python main.py
    ```

## 实现的算法

以下是已实现的算法列表，可以通过复选框选择并运行：

- [X] DQN (Deep Q-Network)
- [X] Double DQN
- [X] Dueling DQN
- [X] Noisy DQN
- [X] Prioritized DQN
- [X] Rainbow
- [ ] A2C (Advantage Actor-Critic)
- [ ] PPO (Proximal Policy Optimization)
- [ ] SAC (Soft Actor-Critic)
- [ ] DDPG (Deep Deterministic Policy Gradient)
- [ ] TD3 (Twin Delayed DDPG)

## 许可证

此项目基于 [MIT 许可证](LICENSE) 开源。

---



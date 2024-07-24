import os
import argparse
import gymnasium
import matplotlib.pyplot as plt
from PPO import PPOAgent


def train(args):
    """
    训练并在环境中执行以获取奖励。

    Args:
        args (argparse.Namespace): 包含所有命令行参数的命名空间对象。

    Returns:
        None
    """
    env = gymnasium.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(
        state_size,
        action_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        update_every=args.update_every,
        k_epochs=args.k_epochs,
        entropy_coeff=args.entropy_coeff
    )

    episode_rewards = []
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    for i_episode in range(1, args.n_episodes + 1):
        state, _ = env.reset()
        total_rewards = 0
        for _ in range(args.max_t):
            action, log_prob = agent.select_action(state, sample=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            done = terminated or truncated
            agent.remember(state, action, reward, log_prob, done)
            state = next_state
            if done:
                break
        episode_rewards.append(total_rewards)
        if i_episode % agent.update_every == 0:
            agent.learn()
        if i_episode % args.print_interval == 0:
            print(f"Episode {i_episode}/{args.n_episodes} - Score: {total_rewards}")
        if i_episode % args.checkpoint_interval == 0:
            agent.save("checkpoints", i_episode)
    env.close()

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes")
    plt.savefig("checkpoints/rewards.png")


def test(args):
    """
    对训练好的agent进行5次测试，并打印每次的总奖励。

    Args:
        args: argparse.Namespace类型的对象，包含以下属性：
            - env_name (str): 环境名称。
            - render_mode (str): 渲染模式，可选值为'human'或'rgb_array'。
            - n_episodes (int): 从检查点中加载的代理模型的序号。

    Returns:
        None
    """
    env = gymnasium.make(args.env_name, render_mode=args.render_mode)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(state_size, action_size)
    agent.load("checkpoints", args.n_episodes)

    for _ in range(5):
        state, _ = env.reset()
        total_rewards = 0
        while True:
            env.render()
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            if terminated or truncated:
                break
            state = next_state
        print(f"Test Completed! Total Rewards: {total_rewards}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_mode", action="store_true", help="是否为测试模式，默认为False"
    )
    parser.add_argument(
        "--env_name", default="CartPole-v1", help="环境名称，默认为CartPole-v1"
    )
    parser.add_argument("--render_mode", default="human", help="渲染模式，默认为human")
    parser.add_argument(
        "--gamma", default=0.99, type=float, help="折扣因子，默认为0.99"
    )
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="学习率，默认为0.001"
    )
    parser.add_argument(
        "--clip_epsilon", default=0.2, type=float, help="裁剪epsilon，默认为0.2"
    )
    parser.add_argument(
        "--update_every", default=4, type=int, help="更新间隔，默认为4"
    )
    parser.add_argument(
        "--k_epochs", default=4, type=int, help="K步优化，默认为4"
    )
    parser.add_argument(
        "--entropy_coeff", default=0.01, type=float, help="熵系数，默认为0.01"
    )
    parser.add_argument(
        "--n_episodes", default=1000, type=int, help="训练回合数，默认为1000"
    )
    parser.add_argument(
        "--max_t", default=1000, type=int, help="最大时间步数，默认为1000"
    )
    parser.add_argument(
        "--print_interval", default=10, type=int, help="打印间隔，默认为10"
    )
    parser.add_argument(
        "--checkpoint_interval", default=100, type=int, help="检查点间隔，默认100"
    )
    args = parser.parse_args()

    if args.test_mode:
        test(args)
    else:
        train(args)
        test(args)
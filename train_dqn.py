import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from itertools import count
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    # Environment setup
    env = gym.make(args.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Agent initialization
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device,
                    gamma=0.99, lr=args.lr,
                    epsilon_start=0.9, epsilon_end=0.05,
                    epsilon_decay=1000, target_update=500,
                    buffer_capacity=10000, batch_size=args.batch_size)

    # Training loop
    episode_rewards = []
    for i_episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in count():
            # Select and perform action
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, observation, done)
            
            # Move to next state
            state = observation
            episode_reward += reward
            
            # Perform one step of optimization
            agent.learn()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f'Episode {i_episode}\tAverage Reward: {avg_reward:.2f}\tEpsilon: {agent.epsilon:.2f}')
            
            # Check if solved
            if avg_reward >= 475.0:
                print(f'Solved at episode {i_episode}!')
                break

    # Plot results
    plt.plot(episode_rewards)
    plt.title(f'DQN Training - {args.env}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # Save model
    torch.save(agent.policy_net.state_dict(), 'dqn_cartpole.pth')

if __name__ == '__main__':
    main()
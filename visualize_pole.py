import gymnasium as gym
import torch
from dqn_agent import DQNAgent

def visualize_trained_model(model_path='dqn_cartpole.pth', num_episodes=3):
    # Initialize environment with rendering
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create agent (with minimal initialization)
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
        lr=0.001,  # Doesn't matter for visualization
        batch_size=64  # Doesn't matter for visualization
    )
    
    # Load the trained model
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()  # Set to evaluation mode
    
    # Run episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action from policy (no exploration)
            action = agent.choose_action(state, evaluate=True)
            
            # Take action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    
    env.close()

if __name__ == '__main__':
    visualize_trained_model(model_path='dqn_cartpole.pth', num_episodes=3)
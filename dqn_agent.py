from models import QNetwork
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import random
import math
import numpy as np

class DQNAgent:
    def __init__(self, obs_dim, action_dim, device, 
                 gamma=0.99, lr=1e-4, 
                 epsilon_start=0.9, epsilon_end=0.05, 
                 epsilon_decay=1000, target_update=500,
                 buffer_capacity=10000, batch_size=128):
        
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        
        # Networks
        self.policy_net = QNetwork(obs_dim, action_dim).to(device)
        self.target_net = QNetwork(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
    
    def choose_action(self, state, evaluate=False):
        """Select an action using epsilon-greedy policy"""
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, 
                                          device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)
    
    def learn(self):
        """Perform one step of optimization"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update epsilon
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
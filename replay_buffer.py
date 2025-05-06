from collections import namedtuple, deque
import random
import torch

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) 
                               for s in batch.state])
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.cat([torch.tensor(ns, dtype=torch.float32).unsqueeze(0) 
                                    for ns in batch.next_state])
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.memory)
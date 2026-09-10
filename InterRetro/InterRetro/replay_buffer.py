
from tqdm import tqdm
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity, agent):
        self.agent = agent
        self.capacity = capacity
        self.buffer = {
            'obs': [[] for _ in range(int(capacity))],
            'actions': [[] for _ in range(int(capacity))],
            'next_obs': [[] for _ in range(int(capacity))],
            'rewards': [[] for _ in range(int(capacity))],
            'dones': [[] for _ in range(int(capacity))],
            'graphs': [[] for _ in range(int(capacity))],
        }
        self.size = 0
        self.pos = 0
        
    def is_full(self):
        return self.size == self.capacity
    
    def add(self, obs, actions, next_obs, rewards, dones, remember_graph):
        idx = int(self.pos % self.capacity)
        self.buffer['obs'][idx] = obs
        self.buffer['actions'][idx] = actions
        self.buffer['next_obs'][idx] = next_obs
        self.buffer['rewards'][idx] = rewards
        self.buffer['dones'][idx] = dones
        
        if remember_graph:
            reactant_string = '.'.join(next_obs)
            graph_seq = self.agent.build_graph_seq(obs, reactant_string, actions)
            self.buffer['graphs'][idx] = graph_seq
        
        self.pos += 1
        self.size = int(min(self.size + 1, self.capacity))
        bound = min(self.pos, self.size)
        diversity = len(set(self.buffer['obs'][:bound])) / bound
        return {'buffer/space': self.size/self.capacity,
                'buffer/size': self.size,
                'buffer/diversity': diversity}

    def update_graphs(self):
        for i, (obs, actions, next_obs) in tqdm(enumerate(zip(self.buffer['obs'], 
                                               self.buffer['actions'], 
                                               self.buffer['next_obs'])), total=self.capacity):
            reactant_string = '.'.join(next_obs)
            graph_seq = self.agent.build_graph_seq(obs, reactant_string, actions)
            self.buffer['graphs'][i] = graph_seq
            if i == self.capacity:
                break

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return {
            'obs': [self.buffer['obs'][i] for i in indices],
            'actions': [self.buffer['actions'][i] for i in indices],
            'next_obs': [self.buffer['next_obs'][i] for i in indices],
            'rewards': [self.buffer['rewards'][i] for i in indices],
            'dones': [self.buffer['dones'][i] for i in indices],
            'graphs': [self.buffer['graphs'][i] for i in indices],
        }

    def sample_without_replacement(self, batch_size):
        indices = list(range(self.size))
        random.shuffle(indices)  # Shuffle the indices
    
        for start_idx in range(0, self.size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield {
                'obs': [self.buffer['obs'][i] for i in batch_indices],
                'actions': [self.buffer['actions'][i] for i in batch_indices],
                'next_obs': [self.buffer['next_obs'][i] for i in batch_indices],
                'rewards': [self.buffer['rewards'][i] for i in batch_indices],
                'dones': [self.buffer['dones'][i] for i in batch_indices],
                'graphs': [self.buffer['graphs'][i] for i in batch_indices],
            }


    def __len__(self):
        return self.size
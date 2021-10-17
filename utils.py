import random
import torch
import gym
from gridworld import GridworldEnv
import re

def set_seed(seed, env):
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

def get_env(env_name):
    if 'gridworld' in env_name.lower():
        grid_size = re.findall(r'([0-9]+)x([0-9]+)', env_name)
        if grid_size is None:
            gird_size = [16,16]
        else:
            grid_size = [int(grid_size[0][0]), int(grid_size[0][1])]

        env = GridworldEnv(shape=grid_size)
        state_dim, action_dim = 1, env.action_space.n
    else:
        env = gym.envs.make(env_name)
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    return env, state_dim, action_dim

def get_epsilon(it, eps_start=1.0, eps_end=0.05, decay_iters=1000):
    epsilon = eps_start - (it*((eps_start-eps_end)/decay_iters)) if it < decay_iters else eps_end
    return epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    # Calculate q values
    q_values = Q(states)
    # Pick the q values of the selected actions for the whole batch
    selected_q_values = torch.gather(q_values, dim=1, index=actions)
    return selected_q_values
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of rewards. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    # Compute q values, pick the maximum ones and reshape
    q_values = Q(next_states)
    q_max = torch.max(q_values, dim=1)[0].unsqueeze(1)
    
    # Calculate the non terminal rewards and terminal rewards seperately
    # In this case, it's actually not necessary, because it is a continuing task (not episodic)
    targets_nonterminal = rewards + discount_factor*q_max
    targets_terminal = rewards
    
    # Calculate the final targets
    targets = torch.where(dones, targets_terminal, targets_nonterminal)
    return targets

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
        # Added a pointer here 
        self.pointer = 0

    def push(self, transition):
        # If memory is not yet at capacity, keep appending transitions
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        # Otherwise, we want to use a pointer and overwrite the memory starting from the beginning
        else:
            if self.pointer >= self.capacity:
                self.pointer = int((self.pointer) % self.capacity)

            self.memory[self.pointer] = transition
        
        # Increment the pointer
        self.pointer += 1

    def sample(self, batch_size):
        transition_batch = random.sample(self.memory, k=batch_size)
        return transition_batch

    def __len__(self):
        return len(self.memory)
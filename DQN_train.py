import random
import torch
import torch.nn.functional as F
import gym
import time
from Q_network import QNetwork
from utils import *
from gridworld import GridworldEnv
import re

class DQL_policy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, device):
        self.Q = Q
        self.epsilon = epsilon
        self.device = device
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        with torch.no_grad():
            # First turn observation into a float tensor, and reshape to be (1 x obs_dim)
            # so that it can fit in the network
            state = torch.tensor(state).unsqueeze(0).to(torch.float).to(self.device)
            if len(state.shape) < 2:
                state = state.unsqueeze(0)
            # Pass obs through networks to get Q values
            actions = self.Q(state)
            
            # Pick greedy action from q-values with probability (1-epsilon) 
            if random.random() > self.epsilon:
                action = torch.argmax(actions)
                return action.item()
            # Otherwise, pick randomly
            else:
                action = random.choice(list(range(actions.shape[1])))
                return action
                
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def dql_train_step(Q, memory, optimizer, batch_size, discount_factor, device):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float).to(device)
    if len(state.shape) < 2:
        state = state.unsqueeze(1)
    action = torch.tensor(action, dtype=torch.int64)[:, None].to(device)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    if len(next_state.shape) < 2:
        next_state = next_state.unsqueeze(1)
    reward = torch.tensor(reward, dtype=torch.float)[:, None].to(device)
    done = torch.tensor(done, dtype=torch.uint8)[:, None].to(device)  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def train_dqn(env_name, num_eps=10000, batch_size=64, hidden_dim=128, lr=1e-3, 
                gamma=0.8, eps_start=1.0, eps_end=0.05, eps_decay_iters=1000,
                mem_cap=10000, seed=42, render=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'gridworld' in env_name.lower():
        env = GridworldEnv(shape=[16,16])
        state_dim, action_dim = 1, env.action_space.n
    else:
        env = gym.envs.make(env_name)
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    
    memory = ReplayMemory(mem_cap)
    set_seed(seed, env)

    Q = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy = DQL_policy(Q, eps_start, device)
    
    optimizer = torch.optim.Adam(Q.parameters(), lr)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    returns = []

    start_time = time.time()
    for i in range(num_eps):
        # First initialize the current epsilon value based on global step, and set it in current policy
        curr_eps = get_epsilon(global_steps, eps_start, eps_end, eps_decay_iters)
        policy.set_epsilon(curr_eps)

        state = env.reset()
        
        steps = 0
        G = 0
        while True:
            
            # Sample action from policy
            action = policy.sample_action(state)
            
            # Get next state and reward
            state_, reward, done, _ = env.step(action)

            # Calulcate G for statistics
            G += (gamma**steps) * reward

            # Render screen if render=True
            if render:
                env.render()
                time.sleep(0.1)
            
            # Push transition to memory
            memory.push((state, action, reward, state_, done))
            
            # Train on a batch of transitions and optimize
            dql_train_step(Q, memory, optimizer, batch_size, gamma, device)
            
            # Increment step counters
            global_steps += 1
            steps += 1
            
            # Set current state to next state
            state = state_
            
            if done:
                if i % 10 == 0:
                    print(f'Episode {i} finished after {steps} steps with cumulative return {G}')

                episode_durations.append(steps)
                returns.append(G)
                break

        if render:
            env.close()
    
    end_time = time.time()

    print(f'DQN ran for {end_time-start_time} seconds on {env_name}')

    return episode_durations, returns
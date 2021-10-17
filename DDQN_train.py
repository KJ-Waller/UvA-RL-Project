import random
import torch
import torch.nn.functional as F
import time
from Q_network import QNetwork
from utils import *

class DDQL_policy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q1, Q2, epsilon, device):
        self.Q1, self.Q2 = Q1, Q2
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
        with torch.no_grad():
            # First turn observation into a float tensor, and reshape to be (1 x obs_dim)
            # so that it can fit in the network
            state = torch.tensor(state).unsqueeze(0).to(torch.float).to(self.device)
            if len(state.shape) < 2:
                state = state.unsqueeze(0)
            # Pass obs through networks to get Q values
            actions_Q1 = self.Q1(state)
            actions_Q2 = self.Q2(state)
            # Average the Q-values
            actions = (actions_Q1 + actions_Q2 ) / 2
            
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


def ddql_train_step(Q1, Q2, memory, optimizer, batch_size, discount_factor, device):
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
    q_val = compute_q_vals(Q1, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q2, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())
        
def train_ddqn(env_name, num_eps=10000, batch_size=64, hidden_dim=128, lr=1e-3, 
                gamma=0.8, eps_start=1.0, eps_end=0.05, eps_decay_iters=1000,
                mem_cap=10000, seed=42, render=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env, state_dim, action_dim = get_env(env_name)

    memory = ReplayMemory(mem_cap)
    set_seed(seed, env)

    Q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    Q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy = DDQL_policy(Q1, Q2, eps_start, device)
    
    optimizer_Q1 = torch.optim.Adam(Q1.parameters(), lr)
    optimizer_Q2 = torch.optim.Adam(Q2.parameters(), lr)
    
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
            
            # update Q1 or Q2, using the other network
            if random.random() > 0.5:
                ddql_train_step(Q1, Q2, memory, optimizer_Q1, batch_size, gamma, device)
            else:
                ddql_train_step(Q2, Q1, memory, optimizer_Q2, batch_size, gamma, device)
            
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
                #plot_durations()
                break

        if render:
            env.close()
    
    end_time = time.time()

    print(f'DDQN ran for {end_time-start_time} on {env_name}')

    return episode_durations, returns

# train_ddqn('GridWorld-16x16', render=False)
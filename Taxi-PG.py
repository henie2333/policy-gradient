import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

env = gym.make('Taxi-v3')
# observattion spave: 500(1), action space:6
device = torch.device("cpu")

class Agent:
    def __init__(self, n_states, n_hidden, n_actions, lr=0.003):
        
        # Define the agent's network
        self.net = nn.Sequential(nn.Linear(n_states, n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, n_actions),
                                 nn.Softmax(dim=0)).to(device)
        
        # How we're optimizing the network
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        
    def predict(self, observation):
        """ Given an observation, a state, return a probability distribution
            of actions
        """
        state = torch.tensor([observation], dtype=torch.float32).to(device)
        actions = agent.net(state)
        
        return actions

    def update(self, loss):
        """ Update the agent's network given a loss """
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def discount_rewards(rewards, gamma=0.99):
    discounted = []
    R = 0
    for r in rewards[::-1]: # cauculate the R in reverse order
        R = r + gamma * R
        discounted.insert(0, R) # stored the R started by every action(or t)
        
    # Now normalize
    discounted = np.array(discounted)
    normed = (discounted - discounted.mean())/discounted.std()
    
    return normed

agent = Agent(1, 32, 6, lr=0.003)
total_episodes = 5000
max_steps = 500
solved_reward = 100

print_every = 5    # print and update every 10/5 episode
update_every = 3
replay = {'actions':[], 'rewards':[]}    # replay memory
render = True       # render the scene
reward_log = []

cur_episode = 0
while cur_episode < total_episodes:
    state = env.reset()     # gain current state
    rewards, actions = [], []  

    for t in range(max_steps):
        action_ps = agent.predict(state)    # gain predict state based on current model
        action = torch.multinomial(action_ps, 1).item() # gain action randomly based on probability
        actions.append(action_ps[action].unsqueeze(0))  # stord the chosen action's probability
        
        # get the next state and the reward based on the chosen action
        state, reward, done, _ = env.step(action)  

        rewards.append(reward)       
            
        if done or t == (max_steps - 1):
            # Record experiences
            reward_log.append(sum(rewards))
            if cur_episode % print_every == 0:
                print(sum(rewards))
            
            losses = []
            # count the R based on discounter gamma
            rewards = discount_rewards(rewards) 
            
            replay['actions'].extend(actions)
            replay['rewards'].extend(rewards)
            
            # Update our agent with the experiences
            # policy gradient:由于是改进策略的梯度下降，采取了loss = -log(prob_action)*(R)
            # 使得R大的action有更大的policy prob
            if cur_episode % update_every == 0:
                for a, r in zip(*replay.values()):
                    losses.append(-torch.log(a)*r)
                loss = torch.cat(losses).sum()  # let the sum of each action's loss be a new loss
                agent.update(loss)
                
                replay['actions'], replay['rewards'] = [], []
            
            break
            
    if sum(reward_log[-100:])/100 > solved_reward:
        print(f"Environment solved in {cur_episode-100} episodes with a reward of {np.mean(reward_log[-100:])}")
        break
        
    state = env.reset()
    cur_episode += 1

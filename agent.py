import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self,hyperparameter_set):
        with open("hyperparameters.yml", 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

     # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min'] 
        self.learning_rate_a     = hyperparameters['learning_rate_a']        # learning rate for the Adam optimizer
        self.discount_factor_g   = hyperparameters['discount_factor_g']      # discount factor for future rewards

        self.loss_fn = nn.MSELoss()
        self.optimizer = None           # minimum epsilon value


    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0",render_mode="human",use_lidar=False)
        env = gymnasium.make("CartPole-v1",render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []   
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states,num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(),lr = self.learning_rate_a)

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state,dtype=torch.float32).to(device)

            terminated = False
            episode_reward = 0.0

            while not terminated:

                if is_training and random.random()<epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action,dtype=torch.int64).to(device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item( ))
                
                episode_reward += reward

                new_state = torch.tensor(new_state,dtype=torch.float32).to(device)
                reward = torch.tensor(reward,dtype=torch.float32).to(device)
                if is_training:
                    memory.append((state, action, reward, terminated))

                    step_count += 1
                state = new_state

            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon*self.epsilon_decay,self.epsilon_min) 
            epsilon_history.append(epsilon)

            if len(memory)>self.mini_batch_size:

                mini_batch = memory.sample(self.mini_batch_size)

                self.optimizer(mini_batch,policy_dqn,target_dqn)

                if step_count>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        for state, action,new_state, reward, terminated in mini_batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.discount_factor_g*target_dqn(new_state).max()
            current_q = policy_dqn(state)

            loss = self.loss_fn(current_q,target)

            self.optimize 
            loss.backward()
            self.optimizer.step()
if __name__ == '__main__':
    agent = Agent('cartpole1')
    agent.run(is_training=True, render=True)
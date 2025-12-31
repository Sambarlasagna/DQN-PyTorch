# 🧠 Deep Q-Network (DQN) – Reinforcement Learning Project

This project is my hands-on introduction to Reinforcement Learning, where I implemented a Deep Q-Network (DQN) agent from scratch using PyTorch.

The agent learns by interacting with an environment, receiving rewards, and improving its actions over time through trial and error, rather than relying on predefined rules or paths.

# 📌 What this project demonstrates

Core Reinforcement Learning concepts

How an agent learns from experience

Exploration vs exploitation using epsilon-greedy strategy

Q-learning with neural networks (DQN)

Experience replay and target networks for stable learning

Applying the same DQN framework to multiple environments

# 🌍 Environments Used
### ✅ CartPole-v1 (Primary)

Fully trained and evaluated

Continuous state space

Discrete action space

Used to validate and debug the DQN implementation

### 🐦 Flappy Bird (Gymnasium)

Integrated using flappy_bird_gymnasium

Used to experiment with DQN in a more challenging, stochastic environment

Helped explore reward design, exploration behavior, and training instability

CartPole was used as a baseline environment, while Flappy Bird was used to understand how the same RL agent behaves in a harder setting.

# 🧠 Algorithm Used
## Deep Q-Network (DQN)

The agent approximates the action-value function:

𝑄
(
𝑠
,
𝑎
)
=
𝐸
[
future rewards
]
Q(s,a)=E[future rewards]

Key components:

Neural network as a Q-function approximator

Experience Replay buffer

Target Network for stable learning

Epsilon-greedy exploration strategy

Optional support for Double DQN

# 🔁 Training Workflow

1. Agent observes the current state

2. Selects an action (exploration or exploitation)

3. Environment returns next state and reward

4. Experience is stored in replay memory

5. Mini-batches are sampled for training

6. Network weights are updated using Bellman targets

7. Target network is periodically synced

8. Exploration rate decays over time

The objective is to maximize long-term cumulative reward.

# 📊 Monitoring & Logging

Episode rewards tracked

Epsilon decay tracked

Training graphs saved automatically

Best-performing models checkpointed

# 🚀 How to Run

### Install dependencies

`pip install gymnasium torch numpy matplotlib pyyaml flappy-bird-gymnasium`

### Train the agent

`python agent.py cartpole1 --train`

Run the trained agent (evaluation)

`python agent.py cartpole1`

(Flappy Bird environment can be enabled by switching the environment inside agent.py.)

# 📁 Project Structure

DQN-PyTorch/
│
├── agent.py                # Agent logic (training + evaluation)

├── dqn.py                  # Q-network definition

├── experience_replay.py    # Replay memory

├── hyperparameters.yml     # Training configs

├── runs/                   # Saved models, logs, plots

└── README.md

# 🎯 Learning Outcomes

Understood how reinforcement learning differs from supervised learning

Learned how agents improve behavior purely through reward feedback

Implemented exploration vs exploitation in practice

Experienced real RL challenges like instability and reward sparsity

Applied the same DQN agent to both simple and harder environments

# 🔜 Future Work

Improve Flappy Bird performance with better reward shaping

Dueling DQN

Prioritized Experience Replay

Vision-based (pixel input) RL

PPO for comparison with value-based methods

# 🧠 Key Takeaway

Reinforcement learning is about learning from consequences.
This project helped solidify that idea by building and experimenting with a DQN agent across different environments, from simple control tasks to more challenging game dynamics.


# OANDA NEAT Trading Algorithm

This repository contains a trading algorithm that leverages **NeuroEvolution of Augmenting Topologies (NEAT)** for decision-making in the forex market using the OANDA trading platform.

## Overview

The trading agent is designed to learn from historical market data and make informed trading decisions. The algorithm employs NEAT, which evolves neural network architectures through genetic algorithms, optimizing performance over generations.

## Prerequisites

To run this project, you will need to have your own OANDA account. The necessary authentication credentials have been removed from the code to protect sensitive information. You can obtain your account's authentication details from the OANDA platform.

### Dependencies

You can install the required packages using the following command:

```bash
pip install -r requirements.txt


Reward Function

The reward function plays a crucial role in training the trading agent. It is designed to encourage profitable trading behavior while penalizing losses. The key components of the reward function include:

    Profit and Loss Calculation: The reward is based on the net profit or loss from trades executed by the agent. A positive reward is given for profitable trades, while losses incur a penalty.
    Risk Management: To promote sustainable trading practices, the reward function also considers risk exposure, rewarding the agent for maintaining a balanced portfolio.
    Trade Frequency: The function may include incentives for maintaining an optimal trade frequency, preventing overtrading or underutilization of capital.

This multi-faceted reward system encourages the agent to develop strategies that balance profit generation and risk management.
Genetic Algorithm and Augmenting Topologies

The NEAT algorithm employs a genetic algorithm to evolve neural network topologies, allowing for dynamic adjustments to the network's architecture based on performance. Key aspects of this approach include:

    Population Evolution: A population of neural networks is created, and their performance is evaluated based on the reward function.
    Crossover and Mutation: The best-performing networks undergo crossover and mutation to create new offspring networks, promoting diversity in strategies.
    Augmenting Topologies: NEAT allows for the addition of nodes and connections during the evolution process, enabling the architecture to adapt and optimize itself for changing market conditions.

This genetic approach facilitates the discovery of innovative trading strategies that may not be possible with static neural network designs.
Initial Results

Initial results from the trading algorithm can be found in the graphs and trades directories. Here you will find:

    Graphs: Visual representations of the performance of various genome agents over time.
    Trade Lists: Logs of trades executed by the agents, detailing timestamps, trade directions, and profit/loss outcomes.

Directory Structure:
.
├── Dockerfile
├── main.py
├── multi-test.py
├── requirements.txt
├── sine.py
├── README.md
├── docker-commands.txt
├── print_love.bat
├── test_genome.py
├── original_main.py
├── new_main.py
├── live_trading_loop.py
├── AUD_JPY_M1.pkl
├── checkpoints
│   ├── neat-checkpoint-0
│   ├── neat-checkpoint-1
│   ├── neat-checkpoint-2
│   ├── neat-checkpoint-3
│   ├── neat-checkpoint-4
│   ├── neat-checkpoint-5
│   └── neat-checkpoint-6
├── config
│   ├── neat_config.txt
│   ├── experiment_config.py
│   └── acct_config.py
│   └── __pycache__
│       ├── acct_config.cpython-310.pyc
│       └── experiment_config.cpython-310.pyc
├── data
│   └── (Numerous .pkl files for training data)
├── models
│   └── (Various model checkpoints)
├── src
│   ├── trading_environment.py
│   ├── neat_agent.py
│   ├── functions.py
│   ├── __init__.py
│   ├── xor.py
│   ├── visualize.py
│   ├── position.py
│   ├── config-feedforward
│   └── evaluate_genome_new.py
├── trades
│   └── (Trade lists recorded during simulations)
└── graphs
    └── (Graphical representations of agent performance)

Getting Started

    Clone the repository:

    bash

git clone https://github.com/your-repo-url.git
cd your-repo-name

Install the dependencies:

bash

pip install -r requirements.txt

Set up your OANDA account and configure acct_config.py with your authentication details.

Run the trading simulation:

bash

    python main.py

Conclusion

This trading algorithm demonstrates the power of NEAT in creating adaptive trading strategies in the forex market. The combination of dynamic neural network topologies and a robust reward function aims to optimize trading performance over time.

For further exploration and modifications, feel free to dive into the code and experiment with different configurations and parameters.

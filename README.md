Statistical Reinforcement Learning Agent for Rare Earth Element Exchange
📌 Project Overview
This project implements an autonomous Statistical Analyst Agent designed to navigate the highly volatile and complex market of Rare Earth Elements (REE). By integrating traditional time-series forecasting with modern unsupervised anomaly detection and Reinforcement Learning (Q-Learning), the system simulates intelligent decision-making in a multi-agent environment.
The architecture focuses on two primary objectives:
1. Risk Mitigation: Identifying market manipulation or flash crashes using isolation forests.
2. Strategic Optimization: Learning optimal entry and exit points through iterative state-reward cycles.
🏗️ System Architecture
1. The Communication Hub
To simulate a realistic exchange environment, the system utilizes a Publish-Subscribe (Pub-Sub) communication model.
* Channels: Agents subscribe to specific asset classes (e.g., Neodymium, Dysprosium).
* Decoupling: The hub ensures that the sender and receiver of market signals remain independent, allowing for scalable multi-agent simulations.
2. Statistical Analysis & Trend Identification
The agent processes raw market data using Holt-Winters Exponential Smoothing via the statsmodels library.
* Trend Component: The model utilizes an additive trend to generate a Smooth Moving Average (SMA), filtering out high-frequency noise from the underlying commodity price movement.
3. Anomaly Detection (The Risk Engine)
A critical feature for commodity exchange is the detection of non-standard market behavior.
* Algorithm: IsolationForest.
* Functionality: By standardizing price features and measuring "isolation scores," the agent can identify outliers (anomalies) with a configured contamination factor (default 5%).
* Actionable Intelligence: When an anomaly is detected, the agent is programmed to generate 'Sell' or 'Hold' signals to protect the portfolio.
4. Reinforcement Learning (Q-Learning)
The agent features a custom implementation of a Q-Learning algorithm to optimize its trading policy over time.
* Q-Table: Maps state-action pairs to expected future rewards.
* Bellman Equation: Utilizes a learning rate ($\alpha$) and discount factor ($\gamma$) to update values based on market feedback.
* Exploration vs. Exploitation: Implements an $\epsilon$-greedy strategy to ensure the agent continues to discover new market patterns while capitalizing on known profitable states.
⚙️ Technical Stack
* Data Handling: pandas, numpy
* Modeling: scikit-learn (Isolation Forest), statsmodels (Exponential Smoothing)
* Simulation: Custom Python class-based event loop.
🚀 Usage Simulation
The provided script includes a simulation bootstrap using synthetic Rare Earth Element data.
Python

# Initialize the exchange hub and agent
hub = CommunicationHub()
agent = StatisticalAnalystAgent(asset_class="Rare_Earth_Elements", communication_hub=hub)

# Simulate a 10-day trading window
agent.simulate_trading(rare_earth_elements_data)
⚠️ Disclaimer
This repository is a simulation framework for academic and research purposes. It is intended to showcase the integration of reinforcement learning and statistical anomaly detection in commodity market contexts and does not constitute financial advice.




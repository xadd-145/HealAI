# HealAI
🏥 HealAI - Healthcare Management using Reinforcement Learning
🚀 Optimizing Healthcare Policies with AI

📌 Project Overview

This project builds a Reinforcement Learning (RL) environment that simulates a government managing healthcare policies. 
The goal is to train an AI agent that optimizes healthcare levels, manages budgets, and minimizes health risks while navigating elections every 5 years. 
The agent must make strategic decisions to balance investments in healthcare, education, and financial reserves while ensuring long-term sustainability.

🎯 Key Objectives

    ✔ Improve healthcare quality by making optimal decisions.
    ✔ Minimize health risks to prevent pandemics.
    ✔ Manage a limited budget efficiently.
    ✔ Survive elections by keeping citizens happy.

📦 Project Components
🔹 Custom Reinforcement Learning Environment (Health_Env.py)

    State Space:
    budget 💰 (resources available for spending)
    health_level 🏥 (quality of healthcare in the system)
    risk_level ⚠️ (probability of a pandemic)

    Action Space:
    0 → Invest in healthcare (improves health, reduces risk, costs money).
    1 → Invest in education & prevention (small health boost, lowers risk).
    2 → Do nothing (saves budget, but increases health risk).

    Rewards & Penalties:
    Positive reward for increasing health and lowering risk.
    Election bonus every 5 years if the healthcare level is higher than the risk level.
    Severe penalties for pandemics if risks exceed a threshold.

🔹 Reinforcement Learning Agents (Algorithms.py)

    🤖 1. Q-learning with Value Function Approximation (VFA)
    Uses linear approximation to learn the best policies.
    Epsilon-greedy action selection (chooses best action most of the time but explores occasionally).
    Updates Q-values based on rewards received.
    🧠 2. Deep Q-Learning (DQL)
    Uses a Neural Network to approximate Q-values.
    Learns long-term strategies using experience replay.
    Trains on past experiences to improve decision-making over time.

📊 Results & Performance

    -The agent learns over time to invest efficiently in healthcare.
    -Election survival rate improves as the model optimizes health-risk balance.
    -The Deep Q-Learning agent outperforms basic Q-learning in complex scenarios.

📜 License

    This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing

    We welcome contributions! Feel free to submit a pull request or open an issue for discussion.

👨‍💻 Authors

    Aditi Patil Linkedin - https://www.linkedin.com/in/aditi-amitpatil/
    Parag Patel Linkedin - https://www.linkedin.com/in/parag1912/
    Isha Parab Linkedin - https://www.linkedin.com/in/ishaparab/

🚀 Let's build smarter healthcare policies with AI! 💡✨

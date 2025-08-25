# **DroneNavRL — PPO vs DQN for Obstacle-Aware Drone Delivery**

This project implements a **custom reinforcement learning environment** to simulate **autonomous drone navigation for delivery systems**.
The drone navigates a **10×10 grid** to reach the goal while avoiding **dynamic obstacles (birds)** and a **static no-fly zone**.
Two popular RL algorithms, **Proximal Policy Optimization (PPO)** and **Deep Q-Network (DQN)**, are implemented and compared.

---

## **🎯 Aim**

* Develop and evaluate **custom RL algorithms** for dynamic grid-world navigation.
* Compare the **performance, stability, and convergence** of PPO and DQN.
* Analyze training behavior with techniques like **curriculum learning** and **reward shaping**.
* Demonstrate practical application for **real-world drone delivery optimization**.

---

## **⚙️ Environment**

* **Grid Size:** 10×10 (configurable)
* **Start → Goal:** (0,0) → (9,9)
* **Obstacles:**

  * **Dynamic:** 3 randomly moving birds
  * **Static:** No-fly zone rectangle `(3,3)` to `(5,5)`
* **Actions:** `0=up, 1=right, 2=down, 3=left`
* **Rewards:**

  * `+100` for successful delivery
  * `-100` for collision with bird or no-fly violation
  * `-1` per time step (to encourage faster routes)
* **Episode Ends:**

  * Reaching goal
  * Collision or violation
  * Reaching max steps (default `300`)

---

## **📊 Results**

| Algorithm | Success Rate | Average Reward | Stability | Notes                                                   |
| --------- | ------------ | -------------- | --------- | ------------------------------------------------------- |
| **PPO**   | **88.0%**    | High           | High      | Smooth learning curve, better generalization            |
| **DQN**   | **58.5%**    | Moderate       | Medium    | Slower convergence, sensitive to exploration parameters |

* **PPO** outperformed DQN in both success rate and convergence speed.
* **Curriculum Learning:** Gradually scaling grid size (6×6 → 8×8 → 10×10) improved PPO stability and training efficiency.
* **Reward Shaping:** Helped reduce random wandering and optimized route discovery.

---

## **🚀 Quickstart**

### **1. Setup**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### **2. Train PPO**

```bash
python scripts/train_ppo.py --timesteps 200000
```

### **3. Evaluate and Save Replay**

```bash
python scripts/eval.py --algo ppo --model_path checkpoints/ppo_best.zip --episodes 3 --gif_path media/ppo.gif
```

### **4. Train and Evaluate DQN**

```bash
python scripts/train_dqn.py --timesteps 200000
python scripts/eval.py --algo dqn --model_path checkpoints/dqn_best.zip --episodes 3 --gif_path media/dqn.gif
```

---

## **📂 Project Structure**

```
drone-delivery-rl/
├─ notebooks/DroneDelivery_PPO_vs_DQN.ipynb
├─ src/
│  ├─ envs/grid_world.py
│  └─ algos/{ppo.py, dqn.py}
├─ scripts/
│  ├─ train_ppo.py
│  ├─ train_dqn.py
│  └─ eval.py
├─ checkpoints/
├─ media/
├─ configs/default.yml
├─ requirements.txt
└─ README.md
```

---

## **🧠 Techniques Implemented**

* **Custom Environment:** Compatible with `gymnasium` and stable-baselines3.
* **Deep Q-Network (DQN):** Value-based learning with experience replay.
* **Proximal Policy Optimization (PPO):** Policy gradient method for stable updates.
* **Curriculum Learning:** Scaling environment complexity to enhance training performance.
* **Reward Shaping:** Faster convergence by penalizing wandering and optimizing path efficiency.

---

## **🔮 Future Enhancements**

* Integrate **multi-agent coordination** for drone swarms.
* Add **realistic wind and energy constraints**.
* Deploy visualizations in **Streamlit** for real-time replay demos.
* Experiment with **A2C** or **SAC** for further performance gains.

---

## **🛠️ Tech Stack**

* **RL Frameworks:** Stable-Baselines3 (PPO, DQN)
* **Environment:** Gymnasium
* **Language:** Python
* **Visualization:** Matplotlib, ImageIO (for GIF replays)
* **Hardware:** CPU (basic) / GPU for faster training

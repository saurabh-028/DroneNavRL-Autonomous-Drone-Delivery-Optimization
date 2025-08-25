
# DroneNavRL — PPO vs DQN for Obstacle-Aware Drone Delivery

Custom **Gymnasium** environment for a **10×10 grid** drone delivery task with dynamic birds and a static no-fly zone.  
Includes training scripts for **PPO** and **DQN** (Stable-Baselines3) and an evaluation script that exports a **GIF** replay.

## Environment
- Grid: `10×10` (configurable), Start `(0,0)` → Goal `(9,9)`
- No-fly zone: rectangle `(3,3)` → `(5,5)`
- Dynamic obstacles: `num_birds=3` random movers (including "stay")
- Actions: `0=up, 1=right, 2=down, 3=left`
- Rewards: `+100 goal`, `-100` collision/no-fly, `-1` per-step
- Ends on **goal**, **collision/violation**, or **max_steps**

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### PPO
```bash
python scripts/train_ppo.py --timesteps 200000
python scripts/eval.py --algo ppo --model_path checkpoints/ppo_best.zip --episodes 3 --gif_path media/ppo.gif
```

### DQN
```bash
python scripts/train_dqn.py --timesteps 200000
python scripts/eval.py --algo dqn --model_path checkpoints/dqn_best.zip --episodes 3 --gif_path media/dqn.gif
```

## Structure
```
drone-delivery-rl/
├─ notebooks/DroneDelivery_PPO_vs_DQN.ipynb
├─ src/
│  ├─ envs/grid_world.py
│  └─ algos/{ppo.py, dqn.py}
├─ scripts/{train_ppo.py, train_dqn.py, eval.py}
├─ checkpoints/
├─ media/
├─ configs/default.yml  (optional)
├─ requirements.txt
└─ README.md
```

## Notes
- Uses **Stable-Baselines3** PPO & DQN.
- `eval.py` renders frames to a GIF under `media/` (no GUI dependency).
- Set seeds per run for stricter reproducibility.

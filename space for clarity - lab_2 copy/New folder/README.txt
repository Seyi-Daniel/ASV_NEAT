# Two-Boat Sectors RL (feature-based)

## Files
- `new_environment.py` — Environment with 2 boats, 100-d observations, HUD/trails, goals & rewards.
- `train_dqn.py` — Double-DQN trainer with per-step action CSV logging and periodic checkpoints.
- `requirements.txt` — Python deps.

## Install
```bash
python -m venv venv
source venv/bin/activate  # (on Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

## Quick visual sanity check (no learning)
```bash
python new_environment.py
```
Hotkeys in the window: H (HUD), G (grid), S (sectors), T (trails), ESC to quit.

## Train
```bash
python train_dqn.py --episodes 300 --render True
```
Outputs go to `results/<timestamp>/`:
- `actions_all.csv` — every action for every step, both boats
- `ckpt_epXX.pth`, `buffer_epXX.pkl`, `state_epXX.pt` — periodic checkpoints
- `returns.npy`, `training_curves_epXX.png` — learning curves

## Resume from a checkpoint
```bash
python train_dqn.py --resume_model results/<ts>/ckpt_ep150.pth \
                    --resume_buffer results/<ts>/buffer_ep150.pkl \
                    --resume_state results/<ts>/state_ep150.pt
```

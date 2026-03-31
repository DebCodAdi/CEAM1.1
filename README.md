# CEAM1.1

# Pitchers V6: Intent & Trajectory Prediction

Trajectory prediction for pedestrians and cyclists on nuScenes, built around the problem statement requirement:

- Input: 2 seconds of past `(x, y)` motion
- Output: 3 seconds of future `(x, y)` trajectory
- Modeling goals: temporal reasoning, social context, and 3 likely future paths
- Primary metrics: ADE and FDE

Current pipeline notes:

- Phase 1 keeps the true `t=0` point as the final history point, so plots and predictions originate from the same anchor with no break.
- Phase 2.1 keeps dynamic neighbor futures for training, burns parked/static obstacles into the forbidden channel, and uses target-aware semantic map grouping.
- Training predicts exactly 3 modes and calibrates confidence so trajectories closer to ground truth are encouraged to rank higher.

## Project Layout

```text
v6_traj_proj/
├── train.py
├── data/
│   ├── sets/nuscenes/
│   └── processed/
├── src/
│   ├── phase1.py
│   ├── phase21.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   └── visualize.py
├── weights/
├── README.md
└── requirements.txt
```

## Pipeline

1. `python src/phase1.py`
   Extracts pedestrian and cyclist scenes from nuScenes and normalizes each scene into an agent-centric frame.
   The final history point is the true `t=0` origin at `(0, 0)`.

2. `python src/phase21.py`
   Builds the raster map tensor and social graph features, then writes `data/processed/v6_feature_set.pkl`.

3. `python train.py`
   Trains the joint transformer and saves the best checkpoint in `weights/v6_best_model.pth`.

4. `python src/visualize.py --index 0`
   Renders the top-3 predicted paths with arrows, step markers, history, and ground truth into `outputs/visualizations/`.

## Model Notes

- `model.py` uses self-attention over motion history and attention-based social context fusion.
- The decoder predicts exactly 3 trajectory modes, matching the problem statement.
- Confidence is trained against trajectory quality so modes closer to ground truth are encouraged to receive higher probability.
- Neighbor trajectories are modeled for the nearest dynamic agents to improve social consistency.

## Setup

Create or activate your virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

The nuScenes dataroot defaults to `data/sets/nuscenes`. You can override it with `NUSCENES_DATAROOT`.

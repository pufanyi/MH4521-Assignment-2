# MH4521 Assignment 2 - Question 3: REINFORCE Algorithm

## Overview

This project implements the REINFORCE algorithm on a corridor environment with switched actions (Chapter 5, Slide 5 problem). Two different models are compared:

- **Model a)**: Feature-based representation with compact feature vectors
- **Model b)**: Tabular representation with full state-action space

Both models are tested with and without a learned baseline (state-value function).

## Environment

The **Corridor Switched Environment** has:
- 3 non-terminal states: 0, 1, 2
- 1 terminal state: 3
- 2 actions: 0 (left) and 1 (right)
- **Special property**: Actions are swapped in state 1 (the middle state)
- Reward: -1 per step
- Optimal path: right → left → right (total reward: -3)

## Implementation

### Core Classes

#### Tabular Representation (Model b)
- `TabularStateValueFunction`: Lookup table for state values
- `TabularPolicy`: Softmax policy with one preference per state-action pair

#### Feature-Based Representation (Model a)
- `FeatureStateValueFunction`: Linear value approximation with features:
  - Normalized state position: `state / 2.0`
  - Middle state indicator: `1.0 if state == 1 else 0.0`
- `FeaturePolicy`: Linear policy with state-action features:
  - State position
  - Middle state indicator
  - Action value

### REINFORCE Variants

The code implements four configurations:
1. **Tabular with baseline**: Uses advantage $G_t - \hat{v}(S_t)$
2. **Tabular without baseline**: Uses raw return $G_t$
3. **Feature-based with baseline**: Uses advantage with feature representation
4. **Feature-based without baseline**: Uses raw return with feature representation

## Requirements

- Python >= 3.13
- numpy >= 1.24.0
- matplotlib >= 3.10.7

## Installation

Using `uv` (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install numpy matplotlib
```

## Running the Code

Execute the main script:

```bash
uv run python Reinforce_partial.py
```

Or if using pip:

```bash
python3 Reinforce_partial.py
```

## Results

The script will:
1. Train all four model configurations for 1,000 episodes each
2. Display progress messages and final performance metrics
3. Generate comparison plots saved as `reinforce_comparison.png`
4. Print a performance summary

### Expected Output

```
============================================================
PERFORMANCE SUMMARY (Average of last 100 episodes)
============================================================
Tabular with baseline:        -3.61 ± 1.03
Tabular without baseline:     -6.29 ± 3.58
Feature-based with baseline:  -4.13 ± 1.46
Feature-based without baseline: -8.57 ± 6.23

Optimal reward: -3 (shortest path: right -> left -> right)
```

## Key Findings

1. **Baseline is critical**: Both representations improve 40-50% with a baseline
2. **Tabular excels for small problems**: Achieves near-optimal performance (-3.61 vs -3)
3. **Feature-based has approximation error**: Slightly suboptimal (-4.13) due to limited features
4. **Variance reduction**: Baseline substantially reduces learning variance

See `REPORT.md` for detailed analysis and comparison.

## File Structure

- `Reinforce_partial.py`: Main implementation with all four model configurations
- `REPORT.md`: Detailed analysis, comparison, and conclusions
- `reinforce_comparison.png`: Generated comparison plots
- `pyproject.toml`: Project dependencies

## Hyperparameters

- Discount factor: $\gamma = 1.0$
- Policy learning rate (with baseline): $\alpha_\theta = 2^{-9}$
- Policy learning rate (without baseline): $\alpha_\theta = 2^{-13}$
- Value function learning rate: $\alpha_\psi = 2^{-6}$
- Episodes: 1,000
- Repeats: 1 (can be increased for averaging)

## Author

Assignment submission for MH4521

## License

See LICENSE file


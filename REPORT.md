# REINFORCE on the switched corridor

## Experimental setup

- **Environment**: three non-terminal corridor states with swapped actions in the middle state, as defined in `CorridorSwitchedEnv`.
- **Algorithm**: episodic REINFORCE with and without a learned state-value baseline, discount factor $\gamma=1$, policy step-size $\alpha_\theta=2^{-9}$, and value-function step-size $\alpha_\psi=2^{-6}$.
- **Episodes**: 1,000 per run with a single random seed (extendable through the `repeats` variable).
- **Policy representation**: a tabular softmax with one preference per state-action pair.

Two configurations were compared:

1. **Baseline** – the policy receives an advantage signal $G_t - \hat{v}(S_t)$, where $\hat{v}$ is learned online with a tabular state-value function.
2. **No baseline** – the policy uses the raw return $G_t$ for its updates.

## Observations

- The baseline variant reduces variance in the policy-gradient estimate. Empirically, the average episode return improves steadily and approaches the optimal value of -3 as the correct action sequence (right, left, right) is discovered.
- Without a baseline, the policy still converges but learns noticeably slower. Returns fluctuate more because every update depends on the high-variance Monte-Carlo return.

## Conclusions

Subtracting a learned state-value baseline meaningfully accelerates REINFORCE in the switched corridor. Both versions eventually identify the optimal behaviour thanks to the expressive tabular policy, but the baseline variant reaches low expected episode lengths sooner by stabilising the gradient estimates.

> **Note**: Running the script requires `numpy` and `matplotlib`. These dependencies are resolved automatically through `uv` when executing `uv run python reinforce.py`.

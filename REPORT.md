# REINFORCE on the Switched Corridor - Question 3

## Experimental Setup

- **Environment**: Three non-terminal corridor states (0, 1, 2) with swapped actions in the middle state (state 1), as defined in `CorridorSwitchedEnv`. Terminal state is 3.
- **Algorithm**: Episodic REINFORCE with and without a learned state-value baseline
  - Discount factor: $\gamma=1$
  - Policy step-size: $\alpha_\theta=2^{-9}$ (with baseline), $\alpha_\theta=2^{-13}$ (without baseline)
  - Value-function step-size: $\alpha_\psi=2^{-6}$
- **Episodes**: 1,000 per run with a single random seed
- **Optimal solution**: The shortest path is right → left → right, giving a total reward of -3

## Model Implementations

Two different feature representations were implemented and compared:

### Model a) Feature-Based Representation

Uses compact feature vectors to represent state-action pairs:

- **State features**: 
  - Normalized state position: `state / 2.0`
  - Middle state indicator: `1.0 if state == 1 else 0.0`
  
- **State-action features**: 
  - State position
  - Middle state indicator
  - Action value

**Implementation details:**
- `FeatureStateValueFunction`: Linear approximation $\hat{v}(s) = \mathbf{w}^T \phi(s)$
- `FeaturePolicy`: Softmax policy with linear preference $h(s,a) = \boldsymbol{\theta}_a^T \phi(s,a)$

### Model b) Tabular Representation

Uses full tabular representation with separate parameters for each state-action pair:

- **State values**: One parameter per state (3 states)
- **Policy preferences**: One parameter per state-action pair (3 states × 2 actions = 6 parameters)

**Implementation details:**
- `TabularStateValueFunction`: Direct lookup table for $\hat{v}(s)$
- `TabularPolicy`: Softmax over action preferences $\boldsymbol{\theta}(s)$

## Results

### Performance Summary (Average of last 100 episodes)

| Configuration | Mean Reward | Std Dev |
|--------------|-------------|---------|
| **Tabular with baseline** | **-3.76** | 1.39 |
| Tabular without baseline | -6.71 | 4.58 |
| Feature-based with baseline | -4.25 | 2.07 |
| Feature-based without baseline | -7.92 | 4.67 |

**Optimal reward**: -3 (shortest path: right → left → right)

## Analysis and Comparison

### 1. Effect of Baseline

Both feature representations show significant improvement when using a baseline:

- **Tabular**: Improved from -6.71 to -3.76 (43% better)
- **Feature-based**: Improved from -7.92 to -4.25 (46% better)

The baseline reduces variance in policy gradient estimates by subtracting the state value $\hat{v}(S_t)$ from the return $G_t$, yielding the advantage function. This stabilizes learning and leads to faster convergence.

### 2. Tabular vs Feature-Based Representation

**Tabular representation performs better** in this small problem:

- With baseline: Tabular (-3.76) vs Feature-based (-4.25)
- Without baseline: Tabular (-6.71) vs Feature-based (-7.92)

**Reasons for tabular superiority:**

1. **Expressive power**: With only 3 states, tabular representation can represent any state-specific behavior without approximation error.

2. **No generalization needed**: The corridor problem doesn't benefit from feature-based generalization since each state has unique optimal actions (especially state 1 with swapped actions).

3. **Direct parameter updates**: Tabular methods update each state independently, avoiding interference from feature sharing.

**Why feature-based performs worse:**

1. **Underfitting**: The chosen features (position, middle-state indicator, action) may not capture all the nuances needed to distinguish optimal behavior in each state.

2. **Feature interference**: Updates to one state-action pair affect others through shared features, potentially slowing learning.

3. **Limited capacity**: Only 3 features vs 6 independent parameters in tabular case.

### 3. Variance Analysis

Standard deviations reveal learning stability:

- **With baseline**: Lower variance (1.39-2.07) indicates stable learning
- **Without baseline**: Higher variance (4.58-4.67) shows unstable, high-variance updates

The baseline substantially reduces gradient variance, making learning more reliable.

### 4. Convergence Behavior

From the learning curves (see `reinforce_comparison.png`):

- **Tabular with baseline**: Converges fastest to near-optimal performance
- **Feature-based with baseline**: Converges but remains slightly suboptimal
- **Without baseline**: Both show slower, noisier convergence

## Conclusions

1. **Baseline is essential**: Both representations benefit dramatically from using a learned baseline to reduce variance.

2. **Tabular excels for small problems**: In the 3-state corridor, tabular representation achieves near-optimal performance (-3.76 vs optimal -3), outperforming the feature-based approach (-4.25).

3. **Representation matters**: While feature-based representations enable generalization in large state spaces, they introduce approximation error that hurts performance in small, discrete problems where exact representation is feasible.

4. **Trade-off consideration**: For larger problems, feature-based representations would be necessary due to memory constraints and sample efficiency through generalization. However, in small problems like this corridor, the extra expressiveness of tabular methods justifies their use.

5. **Learning rate sensitivity**: The no-baseline variants required a much lower learning rate ($2^{-13}$ vs $2^{-9}$) to remain stable, illustrating the variance-reduction benefit of baselines.

## Recommendations

For this switched corridor problem:
- **Best choice**: Tabular REINFORCE with baseline
- Achieves near-optimal performance with stable learning
- Simple to implement and computationally efficient for 3 states

For larger problems:
- Feature-based or neural network representations would become necessary
- Baseline remains critical for variance reduction
- More sophisticated features or learned representations may be needed

> **Note**: Run the script with `uv run python Reinforce_partial.py` to reproduce results and generate comparison plots.

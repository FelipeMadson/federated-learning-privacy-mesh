# ADR-001: Gaussian Mechanism and Rényi Differential Privacy Accounting

## Context
In distributed multi-institutional research data spaces (such as RNP academic hospitals and genomic registries), models trained on private patient data can leak sensitive identity information through membership inference and gradient inversion attacks.

## Decision
We enforce provable $(\epsilon, \delta)$-Differential Privacy directly at the edge boundary. Before submitting parameter updates to the central coordinator:
1. Vectors are clipped to maximum $L_2$ norm $C$:
   $$v_{\text{clip}} = v \cdot \min\left(1, \frac{C}{\|v\|_2}\right)$$
2. Calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 I)$ is added with $\sigma = \text{noise\_multiplier} \times C$.
3. A cumulative privacy accountant computes cumulative $\epsilon$ spend across rounds using the Strong Composition Theorem.

## Consequences
- **Positive**: Mathematical upper bound on information leakage; immune to auxiliary data re-identification.
- **Negative**: Adds controlled perturbation noise to parameters, requiring slight increase in round iterations to reach identical accuracy.

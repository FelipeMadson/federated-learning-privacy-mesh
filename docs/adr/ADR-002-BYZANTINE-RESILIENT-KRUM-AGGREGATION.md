# ADR-002: Byzantine-Resilient Multi-Krum Secure Aggregation

## Context
Decentralized federated systems are vulnerable to Byzantine attackers or compromised research nodes submitting arbitrarily large or inverted weights (data/model poisoning attacks). Standard FedAvg computes a linear mean, allowing a single attacker to corrupt the entire global model.

## Decision
We implement Blanchard et al.'s Multi-Krum algorithm. For $n$ clients with up to $f$ Byzantine adversaries, Krum computes pairwise Euclidean distances and sums the distances to the $n - f - 2$ closest neighbors. The updates with the minimum scores are selected and averaged.

## Consequences
- **Positive**: Provable convergence guarantee even in the presence of malicious nodes as long as $n \ge 2f + 3$.
- **Positive**: Completely neutralizes outlier gradient injection attacks.
- **Negative**: $\mathcal{O}(n^2 \cdot d)$ computation per round, well suited for dozens of institutional research nodes.

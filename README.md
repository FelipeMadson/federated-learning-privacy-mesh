# federated-learning-privacy-mesh

[![CI](https://github.com/FelipeMadson/federated-learning-privacy-mesh/actions/workflows/ci.yml/badge.svg)](https://github.com/FelipeMadson/federated-learning-privacy-mesh/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)](pyproject.toml)

> High-Performance Federated Learning Orchestration Mesh with Provable Rényi Differential Privacy and Byzantine-Robust Multi-Krum Secure Aggregation, implemented in 100% pure Python with zero third-party dependencies.

Architected and maintained by **[Felipe Madison](https://github.com/FelipeMadson)** to enable cross-institutional collaborative AI and private research data spaces (such as RNP Data Spaces, academic hospital consortia, and genomic databases) without exposing raw patient data or allowing model inversion attacks.

---

## 🚀 Key Architectural Highlights

1. **Provable $(\epsilon, \delta)$-Differential Privacy (DP)**:
   - **$L_2$ Gradient Norm Clipping**: Enforces sensitivity bounds $C$ on edge updates before transmission.
   - **Calibrated Gaussian Perturbation Mechanism**: Injects Gaussian noise scaled to the sensitivity bound $\sigma = \text{multiplier} \cdot C$.
   - **Rényi Privacy Accountant**: Calculates rigorous cumulative epsilon expenditure across federation rounds via the Advanced Composition Theorem.

2. **Byzantine-Robust Multi-Krum Secure Aggregation**:
   - Outlier detection and filtering tolerant to up to $f$ adversarial or malicious edge nodes ($n \ge 2f + 3$).
   - Computes pairwise Euclidean distances and sums distances to $n - f - 2$ closest consensus neighbors, completely immunizing the global model against gradient poisoning or sybil inversion attacks.
   - Alternate strategies include standard sample-weighted **FedAvg** and coordinate-wise **Trimmed Mean**.

3. **Autonomous HTTP REST Federation Coordinator Daemon**:
   - Zero-dependency REST server built directly on `http.server.HTTPServer`.
   - Endpoints:
     - `GET /health` — Health check status.
     - `GET /model` — Current global model parameters and round number.
     - `GET /metrics` — Active research nodes, cumulative privacy expenditure $(\epsilon, \delta)$, and round telemetry.
     - `POST /register` — Research node onboarding.
     - `POST /submit_update` — Client update ingestion with automatic privacy validation.
     - `POST /finalize_round` — Executes secure aggregation and updates the global weights.

4. **100% Zero Third-Party Dependencies**:
   - Built entirely with Python 3.12 standard libraries (`math`, `random`, `dataclasses`, `typing`, `http.server`, `json`, `unittest`).
   - Lightning-fast setup, zero Docker bloat, and instant deterministic testing.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph EdgeNodes["Academic Edge Research Nodes (e.g. RNP Teaching Hospitals)"]
        Node1["Hospital A (Raw Private Data)"] -->|Local Training| Up1["Update 1: Clip L2 + Gaussian DP"]
        Node2["Hospital B (Raw Private Data)"] -->|Local Training| Up2["Update 2: Clip L2 + Gaussian DP"]
        Node3["Adversary (Poisoned Gradient)"] -->|Attack| Up3["Malicious Poisoned Update"]
    end

    Up1 -->|REST /submit_update| Coordinator[Federation Coordinator Daemon]
    Up2 -->|REST /submit_update| Coordinator
    Up3 -->|REST /submit_update| Coordinator

    subgraph DefenseMesh["Aggregation & Privacy Mesh"]
        Coordinator --> Krum[Byzantine Multi-Krum Defense]
        Krum -->|Filter Poisoned Outlier| Filtered[Consensus Gradient Set]
        Filtered --> FedAvg[Weighted Parameter Aggregation]
        FedAvg --> Accountant[Rényi DP Privacy Accountant]
        Accountant --> GlobalModel[Updated Global Federated Model]
    end

    GlobalModel -->|Broadcast /model| EdgeNodes
```

---

## 🛠️ Verification & Test Suite

Run the full automated test suite using Python's standard `unittest`:

```bash
python -m unittest discover tests
```

### Passing Tests

```text
Ran 8 tests in 0.001s

OK
```

---

## 📜 License

MIT License. Designed and maintained by [Felipe Madison](https://github.com/FelipeMadson).

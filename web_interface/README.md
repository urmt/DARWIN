# DARWIN: SFH Evolution Simulator - Markov Extension

## Overview
DARWIN is a web-based simulator for exploring concepts from the Sentience-Field Hypothesis (SFH), as detailed in the book "The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality" by Mark Rowe Traver (2025). This repository extension focuses on Markov Chain Monte Carlo (MCMC) simulations for qualic partitioning, a core mathematical component of SFH.

### What is Qualic Partitioning in SFH?
In SFH, the universe is viewed as a sentient field that "weaves" reality by optimizing two key qualities:
- **Coherence (C)**: Structural stability, allowing consistent physical laws and complex systems.
- **Fertility (F)**: Generative potential for novelty, complexity, and emergence (e.g., life, galaxies).

Qualic energy (Q) represents the total "sentient energy" distributed via discrete partitions, modeled using the Hardy-Ramanujan partition function p(Q). This discrete approach resolves cosmological fine-tuning puzzles by showing our universe as a high-probability "sweet spot" in partition space, rather than a random accident.

### Why Use This Simulator?
- **Scientific Rigor and Testability**: MCMC allows researchers to simulate stochastic transitions in qualic configurations, testing SFH's predictions (e.g., clustering at optimal Q levels for fine-tuned constants like the fine-structure constant α). It provides statistical evidence against null hypotheses of random universes.
- **Exploration for Researchers**: Visualize how qualic energy evolves under stochastic dynamics, mimicking quantum-sentient integration (e.g., via Nelson's stochastic mechanics). This helps in fields like quantum physics, neuroscience (neural synchrony thresholds C ≥ 1.3), and cosmology.
- **Educational Value**: Demonstrates how simple probabilistic rules (Markov chains) can model profound concepts like consciousness as a fundamental field.
- **Peer-Review Readiness**: Outputs include convergence metrics (e.g., R-hat for chain reliability), mean Q, variance, and plots—reproducible for validating SFH claims.
- **What It Shows**: Chains converging to stable distributions represent "optimal universes." High mean Q indicates fertile configurations; deviations test falsifiability (e.g., if no clustering at observed constants, SFH is challenged).

If you're a physicist, neuroscientist, or philosopher, use this to experiment with parameters and generate data for papers or further simulations.

## Installation and Setup
- **Dependencies**: None for the web interface (runs in browser). For backend Python scripts (in /markov_simulations/): NumPy (pip install numpy).
- **Running Locally**: Open `darwin_interface.html` in a modern browser (e.g., Chrome). No server needed.
- **GitHub Integration**: This is in the 'markov-extension' branch of https://github.com/urmt/SFH. Merge to main for full access.
- **Testing**: Click "Run Simulation" with defaults—expect a plot and stats in seconds.

## User Interface Guide
The interface (darwin_interface.html) includes:
- **MCMC Qualic Simulator**: Interactive form to run simulations.
  - Explanations and tooltips provided for each field.
  - Outputs: Line plot of the chain, sample values, mean Q, variance, and simple convergence check.
- **Controls**: Adjustable variables for research flexibility (e.g., add burn-in to discard initial unstable steps).
- Navigate via sections; results update dynamically.

## Python Scripts in /markov_simulations/
These mirror the web simulator for offline/advanced use:
- **basic_markov.py**: Simulates simple Markov chains; computes stationary distributions.
- **mcmc_qualic.py**: Core MCMC for qualic sampling; includes Gelman-Rubin (R-hat) for convergence.
- **quantum_markov.py**: Models entangled states; computes correlations.
- **weavelang_markov_extension.py**: Integrates with WeaveLang primitives.

Example: `python mcmc_qualic.py` runs a sample simulation and prints stats.

## Limitations and Future Work
- Current MCMC uses approximations (e.g., asymptotic p(Q)); exact partitions via SymPy for small Q in future.
- No real-time collaboration; add if needed.
- Extend to multi-chain R-hat in web for full diagnostics.

## References
- Traver, M. R. (2025). *The Sentience-Field Hypothesis*. [Book details].
- Markov, A. A. (1906). Extension of the Law of Large Numbers.
- See book appendices for full citations.

For issues: Open a GitHub issue or contact urmt.

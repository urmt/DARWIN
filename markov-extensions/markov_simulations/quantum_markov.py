import numpy as np

def entangled_markov(init_Q1, init_Q2, steps, coupling, noise=0.1):
    """
    Simulate entangled qualic states via coupled Markov processes.
    
    Parameters:
    - init_Q1, init_Q2: float, initial states.
    - steps: int.
    - coupling: float, entanglement strength.
    - noise: float, added diffusion.
    
    Returns:
    - chain: list of (Q1, Q2) tuples.
    """
    Q1, Q2 = init_Q1, init_Q2
    chain = [(Q1, Q2)]
    for _ in range(steps):
        delta = np.random.normal(0, 1) * coupling
        noise1 = np.random.normal(0, noise)
        noise2 = np.random.normal(0, noise)
        Q1 += delta + noise1
        Q2 -= delta + noise2  # Anti-correlation for entanglement analogy
        chain.append((Q1, Q2))
    return chain

# Analysis for rigor
def compute_correlation(chain):
    Q1s = [pair[0] for pair in chain]
    Q2s = [pair[1] for pair in chain]
    return np.corrcoef(Q1s, Q2s)[0, 1]

if __name__ == "__main__":
    chain = entangled_markov(0.0, 0.0, 1000, 0.5)
    print("Sample chain (first 5):", chain[:5])
    corr = compute_correlation(chain)
    print("Correlation (expected ~ -1 for strong entanglement):", corr)

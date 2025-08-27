import numpy as np

def hardy_ramanujan_p(Q, exact=False):
    """
    Hardy-Ramanujan asymptotic partition function p(Q).
    For exact, use sympy if needed (but approximate for large Q).
    """
    if exact:
        from sympy import partition
        return partition(int(Q))
    else:
        if Q <= 0:
            return 0
        return (1 / (4 * Q * np.sqrt(3))) * np.exp(np.pi * np.sqrt(2 * Q / 3))

def mcmc_qualic(steps, init_Q, sigma, T_qualic=1.0, burn_in=0):
    """
    MCMC simulation for qualic energy partitions.
    
    Parameters:
    - steps: int, total MCMC steps.
    - init_Q: float, initial qualic energy.
    - sigma: float, proposal standard deviation.
    - T_qualic: float, qualic temperature.
    - burn_in: int, steps to discard for convergence.
    
    Returns:
    - chain: list of sampled Q values.
    - energies: list of p(Q) values.
    """
    chain = [init_Q]
    energies = [hardy_ramanujan_p(init_Q)]
    for _ in range(steps):
        proposal = chain[-1] + np.random.normal(0, sigma)
        if proposal <= 0:
            continue
        pi_proposal = hardy_ramanujan_p(proposal) * np.exp(-proposal / T_qualic)
        pi_current = hardy_ramanujan_p(chain[-1]) * np.exp(-chain[-1] / T_qualic)
        accept_prob = min(1, pi_proposal / pi_current)
        if np.random.rand() < accept_prob:
            chain.append(proposal)
            energies.append(hardy_ramanujan_p(proposal))
        else:
            chain.append(chain[-1])
            energies.append(energies[-1])
    # Apply burn-in
    chain = chain[burn_in:]
    energies = energies[burn_in:]
    return chain, energies

# Validation and stats
if __name__ == "__main__":
    chain, energies = mcmc_qualic(10000, 10.0, 1.0, burn_in=1000)
    print("Mean Q:", np.mean(chain))
    print("Variance:", np.var(chain))
    # For peer review: Gelman-Rubin (simplified for single chain)
    split1, split2 = chain[:len(chain)//2], chain[len(chain)//2:]
    W = (np.var(split1) + np.var(split2)) / 2
    B = len(chain)//2 * (np.mean(split1) - np.mean(split2))**2
    R_hat = np.sqrt((W + B / len(chain)) / W)
    print("R-hat (convergence):", R_hat)  # Should be ~1 for converged

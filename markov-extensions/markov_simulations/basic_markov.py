import numpy as np

def markov_chain_simulation(transition_matrix, init_state, steps):
    """
    Simulate a Markov chain given a transition matrix.
    
    Parameters:
    - transition_matrix: np.array, square matrix of transition probabilities.
    - init_state: int, starting state index.
    - steps: int, number of simulation steps.
    
    Returns:
    - list: Sequence of states.
    """
    state = init_state
    chain = [state]
    for _ in range(steps):
        state = np.random.choice(len(transition_matrix), p=transition_matrix[state])
        chain.append(state)
    return chain

# Example usage
if __name__ == "__main__":
    trans_mat = np.array([[0.7, 0.3], [0.4, 0.6]])  # 2-state example
    example_chain = markov_chain_simulation(trans_mat, 0, 1000)
    print("Sample chain (first 10):", example_chain[:10])
    # For rigor: Compute stationary distribution
    eigvals, eigvecs = np.linalg.eig(trans_mat.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)].flatten().real
    stationary /= stationary.sum()
    print("Stationary distribution:", stationary)

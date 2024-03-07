#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="The seaborn styles shipped by Matplotlib are deprecated")


# In[2]:


def metropolis_hastings_hypergeom(K, N, n, iterations=10000):
    """
    Metropolis-Hastings algorithm for simulating the hypergeometric distribution.

    Args:
        K (int): Number of successes in the population.
        N (int): Total population size.
        n (int): Number of draws (sample size).
        iterations (int, optional): Number of iterations. Defaults to 10000.

    Returns:
        numpy.ndarray: Array of sampled states from the Markov chain.
        int: Lower bound of the support
        int: Upper bound of the support
    """

    # Boundary values for the hypergeometric distribution's support
    support_min = max(0, n + K - N)
    support_max = min(n, K)

    # Initialize the Markov chain
    X = np.zeros(iterations)
    X[0] = support_min  # Start at the lower limit

    # Metropolis-Hastings loop
    for t in range(1, iterations):
        current_state = X[t - 1]

        # Propose a new state (add or subtract 1)
        proposal = current_state + np.random.choice([-1, 1])

        # Handle boundary conditions
        if proposal < support_min or proposal > support_max:
            X[t] = current_state  # Reject proposal
            continue

        # Calculate acceptance ratio
        acceptance_ratio = min(1, (hypergeom.pmf(proposal, N, K, n) * 0.5) /
                                  (hypergeom.pmf(current_state, N, K, n) * 0.5))

        # Accept or reject the proposal
        if np.random.rand() < acceptance_ratio:
            X[t] = proposal
        else:
            X[t] = current_state

    return X, support_min, support_max 


# In[3]:


# Example usage
K = 25  
N = 80  
n = 15  
iterations = 20000

samples, support_min, support_max = metropolis_hastings_hypergeom(K, N, n, iterations)


# In[4]:


# Visualize the results
plt.style.use('seaborn-darkgrid') 
plt.figure(figsize=(10, 5))  # Wider figure
plt.plot(samples[1:500], 
         linestyle='-', 
         color='teal', 
         marker='s',       # Square markers
         markersize=6      # Larger marker size
         ) 
plt.xlabel('MCMC step', fontsize=12)  # Adjust font sizes
plt.ylabel('X_t', fontsize=12)
plt.title('Metropolis-Hastings Simulation: First 600 Steps', fontsize=14) 
plt.show()


# In[5]:


# Histogram and the true PMF
plt.figure(figsize=(10, 3.5))
plt.hist(samples, bins=range(support_min, support_max + 2), density=True, label='MCMC Estimate')

# Calculate and overlay the true PMF
true_pmf = hypergeom.pmf(range(support_min, support_max + 1), N, K, n)
plt.plot(range(support_min, support_max + 1), true_pmf, color='red', label='True PMF')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('MCMC Simulation vs. True Hypergeometric Distribution')
plt.legend()
plt.show()


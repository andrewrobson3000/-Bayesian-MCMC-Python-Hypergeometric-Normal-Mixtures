#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def f(x, mu1, mu2, S1i, S2i, p1=0.5):
    """Mixture of normals density (up to a constant factor)"""
    x = np.array(x)  # Ensure x is a NumPy array
    c1 = np.exp(-np.dot(x - mu1, np.dot(S1i, x - mu1)))
    c2 = np.exp(-np.dot(x - mu2, np.dot(S2i, x - mu2)))
    return p1 * c1 + (1 - p1) * c2 


# In[3]:


def mcmc_binorm(n, a, x0, mu1, mu2, S1i, S2i):
    X = np.zeros((n, 2))  # Initialize array to store states
    X[0, :] = x0 

    for t in range(1, n):
        y = x0 + (2 * np.random.rand(2) - 1) * a
        MHR = f(y, mu1, mu2, S1i, S2i) / f(x0, mu1, mu2, S1i, S2i)
        if np.random.rand() < MHR:
            x0 = y  # Update the state
        X[t, :] = x0

    return X


# In[4]:


# Parameters
mu1 = np.array([1, 1])
mu2 = np.array([4, 4])
S = np.diag([2, 2]) 
S1i = S2i = np.linalg.inv(S) 

# Simulation
X = mcmc_binorm(a=4, n=10000, x0=mu1, mu1=mu1, mu2=mu2, S1i=S1i, S2i=S2i)

# Plotting (Requires matplotlib)
import matplotlib.pyplot as plt

plt.plot(X[1:2000, 0], label='X[1]_t')  
plt.xlabel('MCMC step')
plt.ylabel('MCMC state X[1]_t')
plt.legend()
plt.show()

plt.scatter(X[:, 0], X[:, 1])  
plt.show()

# Contour Plot
m = 30
u = np.linspace(-2, 7, m)
v = np.linspace(-2, 7, m) 
z = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        z[i, j] = f(np.array([u[i], v[j]]), mu1, mu2, S1i, S2i)

plt.contour(u, v, z, levels=7, colors='black') 
plt.show()


# In[ ]:





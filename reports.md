# Sampling the Hypergeometric Distribution Using the Metropolis-Hastings Algorithm (see discrete_sampling.py): 

## Introduction to MCMC

My fascination with the Metropolis-Hastings algorithm stemmed from my degree in Statistics, when I realized the challenge of sampling from complex probability distributions, especially in Bayesian inference. The core of Markov Chain Monte Carlo (MCMC) methods, including the Metropolis-Hastings algorithm, lies in their ability to navigate these complexities. By generating a vast number of samples, I aimed to not only approximate the shape of the hypergeometric distribution but also to calculate expectations, estimate probabilities of events, and assess variability.

## The Mathematical Basis of MCMC (Discrete)

The mathematical underpinnings of the Metropolis-Hastings algorithm are both elegant and robust. Suppose I need samples from a probability mass function $\pi$, indexed by $i$ in a finite set $\Omega$. The beauty of this algorithm is in simulating the transitions $X_{t+1}$ given $X_t$, effectively determining the transition probabilities $P(X_{t+1} = j | X_t = i)$ and thereby constructing a transition matrix $P$ that targets $\pi$.

By simulating a random walk $X_0, X_1, X_2, ...$ in $\Omega$ through proposals from a chosen, simple, irreducible transition matrix $Q_{i,j}$, I employed a rejection step based on a rule that "corrects" these proposals. This process yields a new effective transition matrix $P$ that satisfies detailed balance for $\pi$. Assuming the Markov Chain is irreducible and aperiodic, it adheres to the ergodic theorem, targeting $\pi$ effectively.

In this code example, consider the challenge of selecting balls from a bucket containing $N = 80$ balls, of which $K = 25$ are marked as successes. The goal is to sample the number of successes $k$ in $n$ draws without replacement, a process governed by the hypergeometric distribution. The algorithm excels by navigating the state space $\Omega$, which spans from the minimum to the maximum number of successes possible within the constraints of the draw.


### The Metropolis-Hastings MCMC Algorithm

The Metropolis-Hastings algorithm, as I applied it, involves a proposal probability distribution $q(j|i) = Q_{i,j}$ with transition probability matrix $Q$ ensuring $q(j|i) > 0$ if and only if $q(i|j) > 0$. Starting from an initial state $X_0 = i_0$ with $\pi_{i_0} > 0$, I iterated through the following steps:

- **Draw:** $j$ from $q(\cdot|i)$ and $u$ from a uniform distribution $U[0, 1]$.
- **Accept or Reject:** If $u \leq \alpha(j|i)$ where 
  $$ \alpha(j|i) = \min \left( 1, \frac{p_j q(i|j)}{p_i q(j|i)} \right) $$
  then set $X_{t+1} = j$, otherwise set $X_{t+1} = i$.


In this code example, I tackle the challenge of drawing a specific number of successes from a finite population without replacement, modeled by the hypergeometric distribution. The given Python function `metropolis_hastings_hypergeom` is designed to simulate this scenario, where:

- $N = 80$ represents the total number of balls in the bucket.
- $K = 25$ specifies the number of these balls that are marked as successes.
- $n = 15$ is the number of draws we aim to make from the bucket.

The goal is to estimate the distribution of the number of successes $k$ within these $n$ draws, utilizing the Metropolis-Hastings algorithm. The algorithm's execution involves navigating the state space, denoted as $\Omega$, which is bounded by the calculated `support_min` and `support_max`. These bounds effectively encapsulate the feasible range of successes that can be observed, given the constraints of the draw.

### Implementation Overview

The iterative process set for $t = 1, 2, 3, \ldots, T$ iterations simulates the drawing process. Key components of the algorithm include:

- **Proposal Distribution**: A simple, symmetric proposal that either increments or decrements the current state by 1, akin to flipping a coin to decide the direction of the next proposed move.
- **Acceptance Ratio**: Calculated as $$\alpha(j|i) = \min(1, \frac{p_j \cdot 0.5}{p_i \cdot 0.5})$$, where $p_j$ and $p_i$ are the probabilities of the proposed and current states, respectively, derived from the hypergeometric PMF. This ratio determines whether to accept the new state or remain in the current one.
- **Boundary Conditions**: Ensures the proposed moves remain within the valid state space, defined by `support_min` and `support_max`, rejecting any proposals that fall outside this range.

### Metropolis-Hastings for the Hypergeometric Distribution

The careful design of this algorithm allows for an effective exploration of $\Omega$, from `support_min` to `support_max`, accurately reflecting the distribution's characteristics. By simulating a series of states and evaluating their distribution, I approximate the hypergeometric distribution's behavior under the specified parameters.

This tailored narrative now accurately reflects the implementation and aims of your Metropolis-Hastings simulation for the hypergeometric distribution, grounded in the specifics of your Python code.



## Code Explanation

In my code, the heart of the Metropolis-Hastings algorithm is expressed succinctly. For each iteration, I propose a new state by either incrementing or decrementing the current state, ensuring that the proposal lies within the distribution's valid range. The acceptance ratio $\alpha(j|i)$ dictates whether the proposal is accepted or rejected, ensuring that the chain converges to the target distribution.

## Output Analysis

By examining the trace plot and histogram of the samples generated, I sought to understand how well the algorithm approximated the hypergeometric distribution. Ideally, the trace plot should show good mixing, indicating the algorithm's efficiency in exploring the state space. Similarly, the histogram closely aligns with the theoretical Probability Mass Function (PMF) of the hypergeometric distribution, showing successful convergence.

# Sampling from a Continuous Distribution using Metropolis-Hastings Algorithm (see continuous_sampling.py)


This is an application of the Metropolis-Hastings (MH) algorithm, a Monte Carlo Markov Chain (MCMC) method, to sample from a complex continuous distribution. Specifically, I focus on a mixture of two bivariate normal distributions. The MH algorithm is instrumental in Bayesian inference and computational statistics for sampling from distributions where direct sampling is not feasible. I detail the methodology, implementation, and findings from an experiment that illustrates the algorithm's effectiveness and the impact of parameter choices on its efficiency.

## Introduction

Sampling from complex continuous distributions is a cornerstone task in statistical inference, especially within the Bayesian framework. The Metropolis-Hastings (MH) algorithm is a widely used method to generate a sequence of sample values from a probability distribution from which direct sampling is challenging. This report demonstrates the MH algorithm's application in sampling from a specific continuous distribution: an equal mixture of bivariate normals.

## Theoretical Background

The target density function for our example is defined as:

$$
\pi(\theta) = \frac{1}{2\pi} \left( 0.5e^{-\frac{1}{2}(\theta - \mu_1)^T \Sigma^{-1}_1 (\theta - \mu_1)} + 0.5e^{-\frac{1}{2}(\theta - \mu_2)^T \Sigma^{-1}_2 (\theta - \mu_2)} \right)
$$

where $\theta$ represents the parameters (or state) being sampled, $\mu_1$ and $\mu_2$ are the means of the bivariate normals, and $\Sigma_1$ and $\Sigma_2$ are their respective covariance matrices. This density combines two bivariate normal distributions, offering a model with multiple modes to challenge the sampling algorithm.

## Methodology

The MH algorithm follows a straightforward but powerful procedure:

### Step 1: Proposal Distribution

We choose a simple proposal distribution $q$ that is easy to sample from. Specifically, for a state $\theta$, we propose a new state $\theta'$ using a uniform distribution:

$$
\theta'_i \sim U(\theta_i - a, \theta_i + a)
$$

where $a$ is a positive constant determining the "jump size" or the breadth of the proposal distribution. The conditional density of the proposal is symmetric:

$$
q(\theta'|\theta) = q(\theta|\theta') = \frac{1}{4a^2}
$$

### Step 2: Acceptance Probability

Given the current state $\theta^{(n)}$, we simulate a new candidate state $\theta'$ as described. The candidate is accepted with probability:

$$
\alpha(\theta'|\theta) = \min \left( 1, \frac{\pi(\theta')}{\pi(\theta)} \right)
$$

This step ensures that the chain gradually moves towards higher-density regions of the target distribution.

### Step 3: Ergodicity and Efficiency

The MH algorithm is theoretically ergodic for any $a > 0$, guaranteeing convergence to the target distribution over time. However, the choice of $a$ significantly influences the algorithm's efficiency. A value too small leads to slow exploration (high acceptance but minimal movement), whereas a value too large results in frequent rejections (quick movement but inefficient sampling).

## Implementation and Experiment

We implemented the MH algorithm in Python, targeting the aforementioned mixture of bivariate normals with $\mu_1 = (1, 1)^T$, $\mu_2 = (4, 4)^T$, and identical diagonal covariance matrices for simplicity. We experimented with different values of $a$ to observe its impact on the efficiency of the sampling process.

## Results

Visual inspection of the sampled states via trace plots and scatter plots revealed the algorithm's ability to explore the state space effectively. With $a$ chosen close to the suggested value (around 3), the algorithm demonstrated a good balance between acceptance rate and movement across the distribution's modes.

## Conclusion

The Metropolis-Hastings algorithm proves to be a versatile and effective tool for sampling from complex continuous distributions. Through careful choice of the proposal distribution and tuning of parameters like the jump size $a$, one can achieve efficient sampling even in multimodal or otherwise challenging distributions. This experiment underscores the importance of understanding both the theoretical underpinnings and practical considerations of MCMC methods in statistical computing.

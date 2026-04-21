# Metropolis–Hastings Sampler (MHsampler)

## Overview

This repository contains a minimal implementation of a Metropolis–Hastings (MH) sampler designed for Bayesian parameter inference.

The sampler evolves multiple **independent walkers**, each corresponding to a separate Markov chain. These chains explore the target distribution using a Gaussian random-walk proposal. It is ready to be parallelized but frankly I do not have time :)

---

## Algorithm

For each walker, the sampler performs:

1. **Proposal step**:
   [
   x_{\text{trial}} = x + \text{step} \cdot \mathcal{N}(0, I)
   ]

2. **Acceptance rule**:
   [
   \alpha = \min\left(1, \exp(\log p(x_{\text{trial}}) - \log p(x))\right)
   ]

3. **Update**:

   * Accept with probability ( \alpha )
   * Otherwise retain current state

---

## Installation

 The code depends only on:

* `numpy`


---

## Usage

### 1. Define a log-probability function

```python
import numpy as np

def logprob(x):
    return -0.5 * np.sum(x**2)  # example: standard Gaussian
```

---

### 2. Initialize the sampler

```python
sampler = MHsampler(
    nwalkers=32,
    logprob=logprob,
    prior=[(-5, 5), (-5, 5)],  # bounds for each parameter
    ndim=2
)
```

---

### 3. Run the sampler

```python
sampler.run(
    nsteps=5000,
    mode="random"  # initialize from prior
)
```

Initialization modes:

* `"input"`: use user-provided starting positions
* `"resume"`: continue from previous chain
* `"random"`: sample initial positions from prior

---

### 4. Extract samples

```python
chain = sampler.get_flat_chain(burnin=1000)
```

This returns a flattened array of shape:

```
(nsteps * nwalkers, ndim)
```

---

### 5. Diagnostics

Check acceptance rate:

```python
print(sampler.acceptance_fraction())
```

Typical desirable range:

```
0.2 – 0.4
```

---

## Key Parameters

### `step`

Controls proposal scale:

* Too small → high acceptance, strong autocorrelation
* Too large → low acceptance, poor exploration




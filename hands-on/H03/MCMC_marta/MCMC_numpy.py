import numpy as np
import random

class MHsampler:
    """
    Basic implementation of a Metropolis–Hastings (MH) sampler with multiple
    independent walkers (i.e. independent Markov chains).

    Each walker evolves separately using a Gaussian proposal:
        x_trial = x + step * N(0, I)

    The acceptance rule follows standard MH:
        accept with probability min(1, exp(logp_trial - logp_current))

    Notes
    -----
    - logprob must return the log of the target distribution (up to a constant).
    - the step 
    """

    step = 0.1  # proposal scale (standard deviation of Gaussian proposal)

    def __init__(self, nwalkers, logprob, prior, ndim):
        """
        Parameters
        ----------
        nwalkers : int
            Number of independent chains.
        logprob : callable
            Function computing log-probability: logprob(x, *args, **kwargs).
        prior : list of tuples
            List of (min, max) bounds for each parameter dimension.
            Used only for initialization.
        ndim : int
            Dimensionality of parameter space.
        """
        self.nwalkers = nwalkers
        self.logprob = logprob
        self.prior = prior
        self.ndim = ndim

        # Storage for results
        self.chain = None

        # Counters for diagnostics
        self.accepted = 0
        self.nproposals = 0
        
    
        np.random.seed(2)  # seed for reproducibility (only affects Python's random, not NumPy)


    def _initialize_walkers(self, xstart, mode):
        """
        Initialize the starting positions of all walkers.

        Parameters
        ----------
        xstart : array-like or None
            Initial positions with shape (nwalkers, ndim), used only if mode="input".
        mode : str
            Initialization strategy:
            - "input"  : use provided `xstart`
            - "resume" : continue from the last stored position in `self.chain`
            - "random" : draw initial positions uniformly from the prior bounds

        Returns
        -------
        ndarray
            Array of shape (nwalkers, ndim) containing initial positions.

        Notes
        -----
        - In "resume" mode, the previous chain must exist.
        - In "random" mode, samples are drawn independently for each parameter
        from uniform distributions defined by `self.prior`.
        - Acceptance counters are reset at initialization.
        """

        self.accepted = 0
        self.nproposals = 0
 
        if xstart is not None and mode == "input":
            if xstart.shape != (self.nwalkers, self.ndim):
                raise ValueError(f"xstart must have shape ({self.nwalkers}, {self.ndim})")
            self.chain = None
            return xstart

        elif self.chain is not None and mode == "resume":
           # last step: shape (nwalkers, ndim)
           return self.chain[-1].copy()

        elif mode == "random":
            self.chain = None
            return self._starting_point()
        
        else:
            raise ValueError("Invalid mode. Choose 'input', 'resume', or 'random'.") 

    def _starting_point(self):
        """
        Generate initial walker positions by sampling uniformly within the prior bounds.

        Returns
        -------
        ndarray
            Array of shape (nwalkers, ndim) with initial positions.

        Notes
        -----
        - Each parameter is sampled independently within its (min, max) interval.
        - This does not account for correlations or structure in the target distribution,
        so walkers may start far from high-probability regions.
        """
        xstart = np.ndarray((self.nwalkers,self.ndim))
        for i in range(self.nwalkers):
            for j in range(self.ndim):
                xstart[i,j] = (np.random.uniform(self.prior[j][0], self.prior[j][1]))
        return xstart

    def _step(self, x, *args, **kwargs):
        """
        Perform a single Metropolis–Hastings update for one walker.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            Current position of the walker.

        Returns
        -------
        ndarray
            Updated position after the MH step.

        Notes
        -----
        - A Gaussian proposal is generated around the current position.
        - The move is accepted or rejected according to the MH acceptance rule.
        - The method updates global counters:
            * self.accepted   : number of accepted proposals
            * self.nproposals : total number of proposals
        """
        x_trial = x + self.step * np.random.randn(self.ndim)
        logp = self.logprob(x, *args, **kwargs)
        logp_trial = self.logprob(x_trial, *args, **kwargs)
        if np.log(np.random.rand()) < (logp_trial - logp):
            x = x_trial
            logp = logp_trial
            self.accepted += 1
        self.nproposals += 1
        return x

    def run(self, nsteps, xstart=None, mode='input', *args, **kwargs):
        """
        Run the Metropolis–Hastings sampler for all walkers.

        Parameters
        ----------
        nsteps : int
            Number of steps per walker.
        xstart : ndarray or None
            Initial positions (used only if mode="input").
        mode : str
            Initialization mode ("input", "resume", or "random").
        *args, **kwargs :
            Additional arguments passed to the log-probability function.

        Notes
        -----
        - Each walker is evolved independently for `nsteps`.
        - The resulting chain has shape (nsteps, nwalkers, ndim).
        - If mode="resume", the new chain is appended to the existing one.

        - No thinning or burn-in removal is performed here; this must be handled externally.
        """
        x0 = self._initialize_walkers(xstart, mode)
        chains = np.zeros((nsteps, self.nwalkers, self.ndim))
        for w in range(self.nwalkers):
            x = x0[w]
            chains[0, w] = x
            for i in range(1, nsteps):
                x = self._step(x, *args, **kwargs)
                chains[i, w] = x
                print(f"Step {i+1}, Walker {w+1}: {x}, Acceptance fraction: {self.acceptance_fraction():.3f}\t", end='\r')
        if mode == 'resume' and self.chain is not None:
            self.chain = np.concatenate((self.chain, chains[1:]), axis=0)
            print('resuming previously runned chain')
        else:
            self.chain = chains

    def acceptance_fraction(self):
        return self.accepted / self.nproposals if self.nproposals > 0 else 0.0

    def get_flat_chain(self, burnin = 0):
        """
        Return a flattened version of the chain after discarding burn-in.

        Parameters
        ----------
        burnin : int
            Number of initial steps to discard.

        Returns
        -------
        ndarray
            Array of shape (nsteps * nwalkers, ndim).

        Notes
        -----
        - Walkers are concatenated into a single chain.
        - No thinning is applied.
        - Assumes that burn-in is the same for all walkers.
        """
        return self.chain[burnin:].reshape(-1, self.ndim)




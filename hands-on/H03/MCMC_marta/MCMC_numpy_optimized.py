import numpy as np # type: ignore

class MHsampler:
    """
    Basic Metropolis-Hastings MCMC sampler.

    Characteristics
    ---------------
    - Uses a Gaussian random-walk proposal:
          x_new = x_old + step * N(0, I)
    - Supports multiple independent walkers.
    - Walkers are NOT coupled (this is not an ensemble sampler).
    - Chains are stored with shape:
          (nsteps, nwalkers, ndim)
    - Can start from user-specified positions or random prior draws.

    Parameters
    ----------
    nwalkers : int
        Number of independent chains.

    logprob : callable
        Function returning the log-posterior (or log-target density).

    prior : list-like
        Prior bounds used only for random initialization.
        Expected format:
            [[xmin1, xmax1],
             [xmin2, xmax2],
             ...]

    ndim : int
        Number of model parameters.

    step : float
        Standard deviation of the Gaussian proposal distribution.
    """

    def __init__(self, nwalkers, logprob, prior, ndim, step):

        self.nwalkers = nwalkers
        self.logprob = logprob
        self.prior = prior
        self.ndim = ndim
        self.step = step

        # Storage for results
        self.chain = None

        # Random seed for reproducibility
        self.seed = 2

        # Counters for diagnostics
        self.accepted = 0
        self.nproposals = 0
        


    def _initialize_walkers(self, xstart, mode):
        
        """
        Generate initial positions for all walkers.

        Parameters
        ----------
        xstart : ndarray or None
            Initial walker positions.
            Shape must be (nwalkers, ndim).

        mode : str
            'input'
                Use xstart provided by the user.

            'resume'
                Continue from the last stored sample.

            'random'
                Draw initial positions uniformly from prior bounds.

        Returns
        -------
        ndarray
            Initial walker positions with shape
            (nwalkers, ndim).
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
        Draw initial walker positions uniformly
        within the prior bounds.

        Returns
        -------
        ndarray
            Shape (nwalkers, ndim)
        """
        rng = np.random.default_rng(self.seed)  

        xstart = np.ndarray((self.nwalkers,self.ndim))
        for i in range(self.nwalkers):
            for j in range(self.ndim):
                xstart[i,j] = (rng.uniform(self.prior[j][0], self.prior[j][1]))
        return xstart



    def _run_single_walker(self, args):
        """
        Run a single Metropolis-Hastings chain.

        Parameters
        ----------
        args : tuple
            Contains:
            - logprob function
            - proposal scale
            - parameter dimension
            - starting point
            - number of steps
            - positional arguments for logprob
            - keyword arguments for logprob

        Returns
        -------
        chain : ndarray
            Chain with shape (nsteps, ndim)

        accepted : int
            Number of accepted proposals.

        nsteps-1 : int
            Total number of proposals.
        """

        logprob, step, ndim, x0, nsteps, args_lp, kwargs_lp = args

        rng = np.random.default_rng(self.seed) 

        chain = np.zeros((nsteps, ndim))
        x = x0.copy()
        logp = logprob(x, *args_lp, **kwargs_lp)

        accepted = 0
        chain[0] = x

        for i in range(1, nsteps):
            # Gaussian random-walk proposal
            x_trial = x + step * rng.normal(size=ndim)
            logp_trial = logprob(x_trial, *args_lp, **kwargs_lp)
            
            # Metropolis acceptance test
            if np.log(rng.random()) < (logp_trial - logp):
                x = x_trial
                logp = logp_trial
                accepted += 1

            chain[i] = x

        return chain, accepted, nsteps - 1

    def run(self, nsteps, xstart=None, mode='input', *args, **kwargs):
        """
        Execute all walkers.

        Parameters
        ----------
        nsteps : int
            Number of MCMC steps per walker.

        xstart : ndarray or None
            Initial walker positions.
            Required if mode='input'.

        mode : str
            Initialization mode:
            - 'input'
            - 'resume'
            - 'random'

        *args, **kwargs
            Additional arguments passed directly
            to logprob().

        Notes
        -----
        Current implementation runs walkers
        sequentially, although the structure is
        compatible with possible multiprocessing.
        """

        x0 = self._initialize_walkers(xstart, mode)

        worker_args = [
            (self.logprob, self.step, self.ndim, x0[w], nsteps, args, kwargs)
            for w in range(self.nwalkers)
        ]

        results = [self._run_single_walker(arg) for arg in worker_args]

        chains = np.zeros((nsteps, self.nwalkers, self.ndim))

        total_accepted = 0
        total_proposals = 0

        for w, (chain_w, acc_w, prop_w) in enumerate(results):
            chains[:, w, :] = chain_w
            total_accepted += acc_w
            total_proposals += prop_w

        self.accepted = total_accepted
        self.nproposals = total_proposals

        if mode == 'resume' and self.chain is not None:
            self.chain = np.concatenate((self.chain, chains[1:]), axis=0)
        else:
            self.chain = chains

        def acceptance_fraction(self):
            return self.accepted / self.nproposals if self.nproposals > 0 else 0.0

    def get_flat_chain(self, burnin = 0 ):
        """
        Return flattened samples.

        Parameters
        ----------
        burnin : int, optional
            Number of initial samples to discard
            along the chain axis.

        Returns
        -------
        ndarray
            Shape:
                ((nsteps-burnin)*nwalkers, ndim)
        """ 

        return self.chain[burnin:].reshape(-1, self.ndim)


    def seed_rng(self, seed):
        """
        Set random seed.

        Parameters
        ----------
        seed : int
            Seed used for reproducible chains.
        """
        self.seed = seed
        np.random.seed(seed)

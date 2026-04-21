import multiprocessing as mp
import numpy as np # type: ignore

class MHsampler:


    step = 0.15  # proposal scale (standard deviation of Gaussian proposal)

    def __init__(self, nwalkers, logprob, prior, ndim):

        self.nwalkers = nwalkers
        self.logprob = logprob
        self.prior = prior
        self.ndim = ndim

        # Storage for results
        self.chain = None

        # Random seed for reproducibility
        self.seed = 2

        # Counters for diagnostics
        self.accepted = 0
        self.nproposals = 0
        


    def _initialize_walkers(self, xstart, mode):

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

        rng = np.random.default_rng(self.seed)  

        xstart = np.ndarray((self.nwalkers,self.ndim))
        for i in range(self.nwalkers):
            for j in range(self.ndim):
                xstart[i,j] = (rng.uniform(self.prior[j][0], self.prior[j][1]))
        return xstart



    def _run_single_walker(self, args):
        logprob, step, ndim, x0, nsteps, args_lp, kwargs_lp, seed = args

        rng = np.random.default_rng(seed) 

        chain = np.zeros((nsteps, ndim))
        x = x0.copy()
        logp = logprob(x, *args_lp, **kwargs_lp)

        accepted = 0
        chain[0] = x

        for i in range(1, nsteps):
            x_trial = x + step * rng.normal(size=ndim)
            logp_trial = logprob(x_trial, *args_lp, **kwargs_lp)

            if np.log(rng.random()) < (logp_trial - logp):
                x = x_trial
                logp = logp_trial
                accepted += 1

            chain[i] = x

        return chain, accepted, nsteps - 1

    def run(self, nsteps, xstart=None, mode='input', verbose=False, *args, **kwargs):
        """
        Sequential execution using the same worker used for parallel runs.
        """

        x0 = self._initialize_walkers(xstart, mode)

        worker_args = [
            (self.logprob, self.step, self.ndim, x0[w], nsteps, args, kwargs, self.seed + w)
             for w in range(self.nwalkers)
        ]

        # --- Run in parallel ---
        nproc = min(self.nwalkers, mp.cpu_count())
        with mp.Pool(nproc) as pool:
            results = pool.map(self._run_single_walker, worker_args)

        chains = np.zeros((nsteps, self.nwalkers, self.ndim))

        total_accepted = 0
        total_proposals = 0

        for w, (chain_w, acc_w, prop_w) in enumerate(results):
            chains[:, w, :] = chain_w
            total_accepted += acc_w
            total_proposals += prop_w
            
            if verbose:
                acc_frac = acc_w / prop_w if prop_w > 0 else 0.0
                print(f"Walker {w+1}/{self.nwalkers} done | acc frac = {acc_frac:.3f}")

        self.accepted = total_accepted
        self.nproposals = total_proposals

        if mode == 'resume' and self.chain is not None:
            self.chain = np.concatenate((self.chain, chains[1:]), axis=0)
        else:
            self.chain = chains

        def acceptance_fraction(self):
            return self.accepted / self.nproposals if self.nproposals > 0 else 0.0
            
        if verbose:
            global_acc = self.accepted / self.nproposals if self.nproposals > 0 else 0.0
            print(f"Done. Global acceptance fraction = {global_acc:.3f}")

    def get_flat_chain(self, burnin = 0):

        return self.chain[burnin:].reshape(-1, self.ndim)


    def seed_rng(self, seed):
        """
        Set the random seed for reproducibility.

        Parameters
        ----------
        seed : int
        """
        self.seed = seed

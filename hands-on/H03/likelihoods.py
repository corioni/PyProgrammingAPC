import numpy #  type: ignore

prior = [
    [-2.0,0.0],
    [2.0,6.0],
    [-4.0, 4.0]
]

def log_likelihood(theta, xx, yy, ee):
    ''' The log-likelihood function for the model.'''
    pass
    if theta.shape != (3,):
        raise ValueError('theta must be a 3-element array: (m, b, log_f)')
    m, b, log_f = theta
    model = m * xx + b
    sigma2 = ee**2 + model**2 * numpy.exp(2 * log_f)
    return -0.5 * numpy.sum((yy - model) ** 2 / sigma2 + numpy.log(sigma2))

def log_prior(theta, prior):
    ''' The log-prior function for the model.'''
    if theta.shape != (3,):
        raise ValueError('theta must be a 3-element array: (m, b, log_f)')
    m, b, log_f = theta
    mlim, blim, flim = prior
    if mlim[0] < m < mlim[1] and blim[0] < b < blim[1] and flim[0] < log_f < flim[1]:
        return 0.0
    return -numpy.inf

def log_posterior(theta, xx, yy, ee, prior):
    ''' The log-posterior function'''
    # require a prior argument
    if prior is None:
        raise ValueError('prior must be supplied')
    lp = log_prior(theta, prior)
    if not numpy.isfinite(lp): # why not just implement it in "else" in prior?
        return -numpy.inf
    return lp + log_likelihood(theta, xx, yy, ee)
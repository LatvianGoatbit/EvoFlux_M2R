from scipy import stats, linalg
import numpy as np
import pandas as pd
import os
from scipy.special import logsumexp, gammaln, logit, softmax
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns   
from scipy.stats import invweibull, lognorm, beta, halfnorm
from scipy.integrate import quad
from cmdstanpy import CmdStanModel
import arviz as az

def generate_next_timepoint(m, k, w, mu, gamma, nu, zeta, S, dt, rng=None):
    """
    Simulates the transitions between the homozygous demtheylated, heterozygous
    and homozygous methylated states in a time step dt in a pool of S cells.

    Arguments:
        m: number of homozygous methylated cells - array of the ints
        k: number of heterozygous methylated cells - array of the ints
        w: number of homozygous demethylated cells - array of the ints
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        S: total number of cells all(m + k + w == S) - int
        dt: time step - float > 0
        rng: np.random.default_rng() object, Optional
    Returns:
        Updated m, k, w after transitions have occurred
    """

    if rng is None:
        rng = np.random.default_rng()

    NSIM = len(m)

    # Use sequential rounds of binomial sampling to calculate how many cells
    # transition between each state
    m_to_k, k_out, w_to_k = rng.binomial(
                                    n = (m, k, w), 
                                    p = np.tile([2*gamma*dt, 
                                        (nu + zeta)*dt, 2*mu*dt], [NSIM, 1]).T)

    k_to_m = rng.binomial(n=k_out, p = np.repeat(nu / (nu + zeta), NSIM))

    m = m - m_to_k + k_to_m
    k = k - k_out + m_to_k + w_to_k
    w = S - m - k

    return (m, k, w)

def multinomial_rvs(counts, p, rng=None):
    """
    Simulate multinomial sampling of D dimensional probability distribution

    Arguments:
        counts: number of draws from distribution - int or array of the 
                ints (N)
        p: probability  - array of the floats (D, N)
        rng: np.random.default_rng() object, Optional
    Returns:
        Multinomial sample
    """

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(counts, (np.ndarray)):
        counts = np.full(p[0, ...].shape, counts)

    out = np.zeros(np.shape(p), dtype=int)
    ps = np.cumsum(p[::-1, ...], axis=0)[::-1, ...]
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0

    for i in range(p.shape[0]-1):
        binsample = rng.binomial(counts, condp[i, ...])
        out[i, ...] = binsample
        counts -= binsample

    out[-1, ...] = counts

    return out

def initialise_cancer(tau, mu, gamma, nu, zeta, NSIM, rng=None, init = None):
    """
    Initialise a cancer, assigning fCpG states assuming fCpGs are homozygous 
    at t=0

    Arguments:
        tau: age when population began expanding exponentially - float
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        NSIM: number of fCpG loci to simulate - int
        rng: np.random.default_rng() object, Optional
        init: allowed values 0, 1, 2 or None. If None, initialise assuming
                time cancer began at time time tau, otherwise initailise in 
                0: w, 1: k, or 2: m. 
    Returns:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
    """

    if rng is None:
        rng = np.random.default_rng()

    if init is None:
        # assume fCpG's are homozygous methylated at t=0
        mkw = np.zeros((3, NSIM), dtype = int)
        idx = np.arange(NSIM)
        np.random.shuffle(idx)
        mkw[0, idx[:NSIM//2]] = 1
        mkw[2, idx[NSIM//2:]] = 1

        # generate distribution of fCpG loci when population begins growing 
        # at t=tau
        RateMatrix = np.array([[-2*gamma, nu, 0], 
                                [2*gamma, -(nu+zeta), 2*mu], 
                                [0, zeta, -2*mu]])

        ProbStates = linalg.expm(RateMatrix * tau) @ mkw

        m_cancer, k_cancer, w_cancer = multinomial_rvs(1, ProbStates, rng)
    
    elif init in [0, 1, 2]:
        wkm = np.zeros((3, NSIM), dtype = int)
        wkm[init, :] = 1

        w_cancer, k_cancer, m_cancer = wkm

    else:
        raise ValueError('init must be None or 0, 1 or 2')

    return m_cancer, k_cancer, w_cancer

def grow_cancer(m_cancer, k_cancer, w_cancer, S_cancer_i, S_cancer_iPlus1, rng):
    """
    Grow a cancer, assigning fCpG states according to a multinomial ditribution

    Arguments:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
        S_cancer_i: number of cells at time t - int = m_cancer + k_cancer + w_cancer
        S_cancer_iPlus1: number of cells at time t+dt - int >= S_cancer_i
        rng: np.random.default_rng() object, Optional
    Returns:
        Updated m_cancer, k_cancer, w_cancer
    """

    if rng is None:
        rng = np.random.default_rng()

    if S_cancer_iPlus1 - S_cancer_i > 0:
        prob_matrix = np.stack((m_cancer, k_cancer, w_cancer)) / S_cancer_i
        growth = multinomial_rvs(S_cancer_iPlus1 - S_cancer_i, prob_matrix, rng)

        m_cancer += growth[0, :]
        k_cancer += growth[1, :]
        w_cancer += growth[2, :]

    return m_cancer, k_cancer, w_cancer


def stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, NSIM, init = None):
    """
    Simulate the methylation distribution of fCpG loci for an exponentially 
    growing well-mixed population evolving neutrally

    Arguments:
        theta: exponential growth rate of population - float
        tau: age when population began expanding exponentially - float < T
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        T: patient's age - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        betaCancer: fCpG methylation fraction distribution - np.array[float]
    """

    # calculate the time step so all transition probabilities are <= 10%
    dt_max = 0.01 / np.max((
        2*gamma, 
        2*mu,
        2*nu,
        2*zeta,
        theta)
    )
    
    # calculate deterministic exponential growth population size
    n = int((T-tau) / dt_max) + 2  # Number of time steps.
    t = np.linspace(tau, T, n) 
    dt = t[1] - t[0]
    S_cancer = np.exp(theta * (t-tau)).astype(int)

    if np.any(S_cancer < 0):
        raise(OverflowError('overflow encountered for S_cancer'))

    rng = np.random.default_rng()

    # generate distribution of fCpG loci depending on init param
    m_cancer, k_cancer, w_cancer = initialise_cancer(tau, mu, gamma, nu, zeta, 
                                                     NSIM, rng, init)

    # simulate changes to methylation distribution by splitting the process 
    # into 2 phases, an exponential growth phase and a methylation transition 
    # phase
    for i in range(len(t)-1):
        m_cancer, k_cancer, w_cancer = grow_cancer(m_cancer, k_cancer,
                                                    w_cancer, S_cancer[i], 
                                                    S_cancer[i+1], rng)

        m_cancer, k_cancer, w_cancer = generate_next_timepoint(m_cancer, 
                                                    k_cancer, w_cancer, 
                                                    mu, gamma, nu, zeta,
                                                    S_cancer[i+1], dt, rng)

    with np.errstate(divide='raise', over='raise'):
        betaCancer = (k_cancer + 2*m_cancer) / (2*S_cancer[-1])

    return betaCancer

# We were given these values for each parameter by our supervisor.
T = 50
tau = 45
theta = 2.4
mu = 0.01
gamma = 0.01
nu = mu
zeta = gamma
N  = 10000
init = 0

# Each different simulated data set is relevant for each necessary graph for each peak- the balanced data set would be used if a mixture model was constructed.
betaCancer = stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, N, init)
betaCancer_balanced = stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, N)
betaCancer_meth = stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, N, 2)
betaCancer_hetero = stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, N, 1)
x = np.linspace(0, 1, 1001)

# Begin code for actual Bayesian Inference
model = CmdStanModel(stan_file = "frechet_lpdfs_progress_copy.stan")
data_dict = {
    'N': N,
    'y': betaCancer,
    'T': T,
}



# Need to set inits for model.sample to not throw error due to definition of frechet_lpdf. Sampling limits were up to 4000 for iter_warmup and iter_sampling each.
fit = model.sample(data = data_dict, inits = {
    "tau_rel": 0.9,
    "mu": 0.001,
    "gamma": 0.001,
    "nu_rel": 0.001,
    "zeta_rel": 0.001,
    "theta": np.exp(3)})

df = fit.draws_pd()
df.to_csv('fit_draws.csv')

# Extraction of the relevant samples for each parameter, plotting and calculating R_hat
theta_x = np.linspace(0, 20, 1001)
theta_samples = fit.stan_variable("theta")
n_draws = theta_samples.shape[0] // 4
draws_by_chain_theta = theta_samples.reshape((4, n_draws))
m , n = draws_by_chain_theta.shape[:2]
theta_chain_means = np.mean(draws_by_chain_theta, axis = 1)
theta_overall_mean = np.mean(theta_chain_means)
B_theta = n * np.var(theta_chain_means, ddof = 1)
W_theta = np.mean(np.var(draws_by_chain_theta, axis=1, ddof=1))
var_hat_theta = ((n-1)/n)*W_theta + (1/n)*B_theta
R_hat_theta = np.sqrt(var_hat_theta / W_theta)

plt.hist(theta_samples, bins = np.linspace(0, 20, 1001),
         alpha = 0.4, density = True)
theta_prior = lognorm.pdf(theta_x, s=np.sqrt(2), scale = 3)
plt.plot(theta_x, theta_prior)
plt.xlabel('Theta')
plt.ylabel('Probability Density')
plt.tight_layout()
sns.despine()
plt.savefig(f"{"Theta"}.png")
plt.close()

tau_rel_samples = fit.stan_variable('tau_rel')
draws_by_chain_tau_rel = tau_rel_samples.reshape((4, n_draws))
m , n = draws_by_chain_tau_rel.shape[:2]
tau_rel_chain_means = np.mean(draws_by_chain_tau_rel, axis = 1)
tau_rel_overall_mean = np.mean(tau_rel_chain_means)
B_tau_rel = n * np.var(tau_rel_chain_means, ddof = 1)
W_tau_rel = np.mean(np.var(draws_by_chain_tau_rel, axis=1, ddof=1))
var_hat_tau_rel = ((n-1)/n)*W_tau_rel + (1/n)*B_tau_rel
R_hat_tau_rel = np.sqrt(var_hat_tau_rel / W_tau_rel)
plt.hist(tau_rel_samples, bins = np.linspace(0,1,1001), alpha = 0.4, density = True)
tau_rel_prior = beta.pdf(x, 2, 2)
plt.plot(x, tau_rel_prior)
plt.xlabel('Tau_rel')
plt.ylabel('Probability Density')
plt.tight_layout()
sns.despine()
plt.savefig(f"{"Tau_rel"}.png")
plt.close()

mu_samples = fit.stan_variable("mu")
draws_by_chain_mu = mu_samples.reshape((4, n_draws))
m , n = draws_by_chain_mu.shape[:2]
mu_chain_means = np.mean(draws_by_chain_mu, axis = 1)
mu_overall_mean = np.mean(mu_chain_means)
B_mu = n * np.var(mu_chain_means, ddof = 1)
W_mu = np.mean(np.var(draws_by_chain_mu, axis=1, ddof=1))
var_hat_mu = ((n-1)/n)*W_mu + (1/n)*B_mu
R_hat_mu = np.sqrt(var_hat_mu / W_mu)
mu_x = np.linspace(-0.05, 0.25, 1001)
plt.hist(mu_samples, bins = np.linspace(0,0.2,1001), alpha = 0.4, density = True)
mu_prior = halfnorm.pdf(mu_x, scale = 0.05)
plt.plot(mu_x, mu_prior)
plt.xlabel('Mu')
plt.ylabel('Probability Density')
plt.tight_layout()
sns.despine()
plt.savefig(f"{"Mu"}.png")
plt.close()

# Posterior plots in ArviZ
az.plot_posterior(fit, var_names=['mu', 'tau_rel', 'theta'])
plt.tight_layout()
sns.despine
plt.savefig(f"{"Posterior_plots"}.png")
plt.close()

# Convert to Inference data for use in ArviZ
idata = az.from_cmdstanpy(fit)
az.plot_trace(idata, var_names = ['mu', 'tau_rel', 'theta'])
plt.tight_layout()
sns.despine()
plt.savefig(f"{"Trace_Plots"}.png")

az.plot_pair(idata, var_names = ['mu', 'tau_rel', 'theta', 'tau'])
plt.tight_layout()
sns.despine()
plt.savefig(f"{"Pair_Plots"}.png")

# Assess relevant MCMC diagnostics
R_hats = {
    "theta": R_hat_theta,
    "tau_rel": R_hat_tau_rel,
    "mu": R_hat_mu
}

for param, rhat in R_hats.items():
    print(f"{param:10s} : R_hat =  {rhat:.3f}")

ess = az.ess(idata, var_names = ['mu', 'tau_rel', 'theta'])
print("ESS of each parameter:")
for var in ess.data_vars:
    ess_value = ess[var].values
    print(f"{var}: {ess_value}")

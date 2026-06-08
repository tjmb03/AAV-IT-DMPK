"""
Real-time data integration.

Models earn their keep when they are living objects that update as study data
read out. This module closes the loop: given new observations from an NHP study,
it re-estimates a small number of identifiable parameters and returns the updated
parameter set, so the cross-species dose projection can be refreshed automatically.

Two complementary data streams are used, which is what makes the update
*identifiable*:

  * terminal biodistribution -- internalized/transduced vector per CNS region at
    one or more necropsy cohorts, and
  * CSF pharmacokinetics -- vector concentration in cisternal CSF at early
    sampling times.

We fit two multipliers: CSF->tissue uptake and CSF clearance. The choice of
which parameters to fit is itself a mechanistic decision. Transduction of vector
already delivered to tissue is fast relative to tissue clearance in this regime,
so the internalized amount is *uptake-limited* -- nearly insensitive to the
transduction rate constant. Trying to fit a transduction multiplier from
biodistribution data is therefore ill-posed; the quantity the tissue data
actually inform is the rate at which vector leaves CSF and enters tissue. The CSF
stream, in turn, sees clearance but not uptake-versus-transduction, so it pins
CSF clearance. Together they identify (uptake, clearance); fit against tissue
alone, the two confound.

This is the disciplined choice when data are sparse: estimate only what the data
can inform, keep the rest fixed at mechanistic priors, and report the change. In
production, `update_from_observations` is what you would wire to a LIMS / data
lake so each new necropsy or biofluid readout re-tunes the model and the dose
projection without manual rework.
"""

import numpy as np
from scipy.optimize import least_squares

from .model import simulate


def predict_region_transduced(phys, bio, dose, site, day):
    """Internalized vector per region (vg) at a given day."""
    t, states = simulate(phys, bio, dose=dose, site=site, t_end=day)
    return np.array([states["Vcell"][r][-1] for r in range(states["Vcell"].shape[0])])


def _stack_days(phys, bio, dose, site, days):
    """Stack per-region transduced vector across one or more days -> 1D array
    (day-major: [d0_r0, d0_r1, d0_r2, d1_r0, ...])."""
    days = np.atleast_1d(days).astype(float)
    return np.concatenate([predict_region_transduced(phys, bio, dose, site, d) for d in days])


def predict_csf_conc(phys, bio, dose, site, days, region=0):
    """Vector concentration (vg/mL) in a CSF segment at one or more days.

    A small positive floor guards against integrator undershoot once the segment
    has cleared by many orders of magnitude (the value is then below numerical
    resolution); CSF PK should in practice be sampled early, where the signal is
    well above this floor.
    """
    days = np.atleast_1d(days).astype(float)
    out = []
    for d in days:
        t, states = simulate(phys, bio, dose=dose, site=site, t_end=d)
        out.append(states["Vcsf"][region][-1] / phys["csf_seg_mL"][region])
    return np.maximum(np.array(out), 1.0)


def _bio_with(bio_prior, f_uptake, f_csf_clear):
    bio = dict(bio_prior)
    bio["k_uptake"] = bio_prior["k_uptake"] * f_uptake
    bio["k_csf_deg"] = bio_prior["k_csf_deg"] * f_csf_clear
    return bio


def make_synthetic_observation(phys, bio_prior, dose, site,
                               necropsy_days, csf_days,
                               true_f_uptake, true_f_csf_clear,
                               cv_tissue=0.20, cv_csf=0.15, seed=0, csf_region=0):
    """Generate a noisy 'observed' dataset as if from a real NHP study.

    Returns a dict with both data streams and the schedules that produced them.
    The true biology differs from the prior by the supplied multipliers; the
    fitter should recover them. Lognormal noise mimics assay variability.
    """
    rng = np.random.default_rng(seed)
    bio_true = _bio_with(bio_prior, true_f_uptake, true_f_csf_clear)

    tissue = _stack_days(phys, bio_true, dose, site, necropsy_days)
    tissue = tissue * rng.lognormal(0.0, cv_tissue, size=tissue.shape)

    csf = predict_csf_conc(phys, bio_true, dose, site, csf_days, region=csf_region)
    csf = csf * rng.lognormal(0.0, cv_csf, size=csf.shape)

    return {
        "tissue": tissue, "csf": csf,
        "necropsy_days": list(np.atleast_1d(necropsy_days).astype(float)),
        "csf_days": list(np.atleast_1d(csf_days).astype(float)),
        "csf_region": csf_region,
    }


def update_from_observations(phys, bio_prior, dose, site, observed,
                             bounds_log=(-2.5, 2.5)):
    """Re-estimate (f_uptake, f_csf_clear) from a two-stream observation dict.

    Fits in log space, with the two streams balanced by their assay CVs so that
    neither dominates purely because of unit scale. Returns updated biology and
    the fitted multipliers.
    """
    tissue_obs = np.asarray(observed["tissue"], dtype=float)
    csf_obs = np.asarray(observed["csf"], dtype=float)
    necropsy_days = observed["necropsy_days"]
    csf_days = observed["csf_days"]
    csf_region = observed.get("csf_region", 0)
    w_tissue, w_csf = 1.0 / 0.20, 1.0 / 0.15      # inverse-CV weights

    def residual(theta):
        f_up, f_cl = np.exp(theta)
        bio = _bio_with(bio_prior, f_up, f_cl)
        pred_t = _stack_days(phys, bio, dose, site, necropsy_days)
        pred_c = predict_csf_conc(phys, bio, dose, site, csf_days, region=csf_region)
        r_t = w_tissue * (np.log(pred_t) - np.log(tissue_obs))
        r_c = w_csf * (np.log(pred_c) - np.log(csf_obs))
        return np.concatenate([r_t, r_c])

    res = least_squares(
        residual, x0=np.array([0.0, 0.0]), method="trf",
        bounds=(np.array([bounds_log[0]] * 2), np.array([bounds_log[1]] * 2)),
    )
    f_up, f_cl = np.exp(res.x)
    return {
        "bio_updated": _bio_with(bio_prior, f_up, f_cl),
        "f_uptake": float(f_up),
        "f_csf_clear": float(f_cl),
        "cost": float(res.cost),
        "success": bool(res.success),
    }


# ---------------------------------------------------------------------------
# Bayesian updating.
#
# The least-squares fit above returns a point estimate. With sparse study data
# the more honest -- and more decision-useful -- object is a posterior: how well
# do the data actually constrain (uptake, clearance), and what is the resulting
# uncertainty on the projected human dose? We sample the posterior with a small
# self-contained random-walk Metropolis sampler (no extra dependencies), using
# weakly-informative lognormal priors on the two multipliers and a log-space
# Gaussian likelihood whose scale is the assay CV. The posterior is then
# propagated through the dose projection to yield a credible interval on the
# recommended human dose -- the quantity a sponsor actually has to defend.
# ---------------------------------------------------------------------------

def _predict_streams_fast(phys, bio, dose, site, necropsy_days, csf_days, csf_region=0):
    """Both data streams from the EXACT linear solution (fast inner loop).

    The baseline (non-saturable) model is dy/dt = A y, so y(t) = expm(A t) y0.
    Building A by probing the rhs and exponentiating at the sampling days is both
    faster than a stiff ODE solve and exact, which matters because this is called
    once per likelihood evaluation inside the sampler. (Linear model only; the
    data-integration fit is defined on that baseline, as in the point-estimate
    fit above.)
    """
    from scipy.linalg import expm
    from . import model as _m
    rhs = _m._build_rhs(phys, bio)
    n = _m.NSTATE
    A = np.empty((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        A[:, j] = rhs(0.0, e)
        e[j] = 0.0
    y0 = np.zeros(n)
    y0[_m._i("Vcsf", _m.SITE_TO_REGION[site])] = float(dose)

    nd = np.atleast_1d(necropsy_days).astype(float)
    cd = np.atleast_1d(csf_days).astype(float)
    prop = {d: expm(A * d) @ y0 for d in np.unique(np.concatenate([nd, cd]))}
    tissue = np.array([prop[d][_m._i("Vcell", r)] for d in nd for r in range(_m.N)])
    Vol = phys["csf_seg_mL"][csf_region]
    csf = np.maximum(np.array([prop[d][_m._i("Vcsf", csf_region)] / Vol for d in cd]), 1.0)
    return tissue, csf


def bayesian_update_from_observations(phys, bio_prior, dose, site, observed,
                                      prior_log_sd=0.5, n_samples=4000,
                                      burn=1000, thin=4, proposal_sd=0.09, seed=0):
    """Posterior over (f_uptake, f_csf_clear) by random-walk Metropolis.

    Returns posterior samples (shape (M, 2): columns f_uptake, f_csf_clear),
    medians, 95% credible intervals, the acceptance rate, and a posterior-median
    biology dict for convenience.
    """
    tissue_obs = np.asarray(observed["tissue"], dtype=float)
    csf_obs = np.asarray(observed["csf"], dtype=float)
    nd = observed["necropsy_days"]
    cd = observed["csf_days"]
    csf_region = observed.get("csf_region", 0)
    sig_t, sig_c = 0.20, 0.15                     # log-space noise sd = assay CV
    log_tobs, log_cobs = np.log(tissue_obs), np.log(csf_obs)

    def log_post(theta):
        lp = -0.5 * np.sum((theta / prior_log_sd) ** 2)     # lognormal prior, median 1
        f_up, f_cl = np.exp(theta)
        bio = _bio_with(bio_prior, f_up, f_cl)
        pt, pc = _predict_streams_fast(phys, bio, dose, site, nd, cd, csf_region)
        ll = (-0.5 * np.sum(((np.log(pt) - log_tobs) / sig_t) ** 2)
              - 0.5 * np.sum(((np.log(pc) - log_cobs) / sig_c) ** 2))
        return lp + ll

    rng = np.random.default_rng(seed)
    theta = np.zeros(2)
    lp = log_post(theta)
    chain, n_acc, total = [], 0, burn + n_samples
    for i in range(total):
        prop = theta + rng.normal(0.0, proposal_sd, size=2)
        lp_prop = log_post(prop)
        if np.log(rng.random()) < lp_prop - lp:
            theta, lp, n_acc = prop, lp_prop, n_acc + 1
        if i >= burn and (i - burn) % thin == 0:
            chain.append(theta.copy())
    samples = np.exp(np.array(chain))             # -> (f_uptake, f_csf_clear)

    def ci(x):
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

    f_up_med = float(np.median(samples[:, 0]))
    f_cl_med = float(np.median(samples[:, 1]))
    return {
        "samples": samples,
        "f_uptake_median": f_up_med,
        "f_uptake_ci": ci(samples[:, 0]),
        "f_csf_clear_median": f_cl_med,
        "f_csf_clear_ci": ci(samples[:, 1]),
        "acceptance": n_acc / total,
        "bio_posterior_median": _bio_with(bio_prior, f_up_med, f_cl_med),
    }


def propagate_dose_posterior(phys_target, bio_prior, site, metric_key,
                             target_value, samples, max_draws=400, seed=1):
    """Push posterior (uptake, clearance) draws through the dose projection.

    For each draw, recompute the target-species dose needed to hit the fixed
    efficacy target `target_value` on `metric_key`. Returns the dose draws plus
    median and 95% credible interval -- uncertainty on the recommended dose.
    """
    from .translation import Arm, dose_for_target
    samples = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    if len(samples) > max_draws:
        samples = samples[rng.choice(len(samples), size=max_draws, replace=False)]
    doses = []
    for f_up, f_cl in samples:
        bio = _bio_with(bio_prior, f_up, f_cl)
        arm = Arm(phys_target, site=site, bio=bio)
        doses.append(dose_for_target(metric_key, arm, target_value))
    doses = np.array(doses, dtype=float)
    return {
        "doses": doses,
        "median": float(np.median(doses)),
        "ci": (float(np.percentile(doses, 2.5)), float(np.percentile(doses, 97.5))),
    }

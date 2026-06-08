"""
Visual predictive checks (VPC) for the IT-AAV translational DMPK pipeline.

A VPC asks whether the model's predictive distribution is consistent with the
observed data: simulate many replicate studies at the observed design, summarize
each by its 5th/50th/95th percentiles per time bin, and check that the *observed*
percentiles fall inside the simulated percentile bands.

Honest scope note. A textbook VPC is built for population (NLME) models with
between-subject random effects, where the bands test whether the model captures
the spread *across individuals*. The model here is a mechanistic typical-subject
ODE with a posterior over two biological multipliers plus a residual (assay)
error model. So what is produced is a **posterior-predictive VPC**: the bands
reflect parameter (posterior) uncertainty + residual error, not fitted
between-subject variability. That is a legitimate, reviewer-legible check; it is
simply labelled for what it is. `make_synthetic_study` can inject optional
between-animal variability (`bsv_cv`) so the check can be stress-tested.

Functions
  make_synthetic_study .. generate a replicate study (cohort) at a design
  study_to_observation .. collapse a study to per-bin summaries for fitting
  vpc_csf ............... posterior-predictive VPC for the CSF PK stream (time axis)
  vpc_tissue ............ posterior-predictive VPC for biodistribution, by region
"""

import numpy as np

from .model import REGIONS, N
from . import data_integration as di


def _geomean(x, axis):
    return np.exp(np.mean(np.log(np.maximum(x, 1e-300)), axis=axis))


# ---------------------------------------------------------------------------
# Synthetic study (the "observed" data)
# ---------------------------------------------------------------------------
def make_synthetic_study(phys, bio, dose, site, csf_days, necropsy_days,
                         csf_region=0, n_csf=8, n_tissue=6,
                         residual_cv_csf=0.20, residual_cv_tissue=0.25,
                         bsv_cv=0.0, true_f_uptake=1.0, true_f_csf_clear=1.0,
                         seed=0):
    """A replicate study: n_csf serially-sampled CSF animals and n_tissue
    terminal-necropsy animals per design point.

    Each observation is a typical-subject prediction times lognormal residual
    (assay) error; if ``bsv_cv`` > 0 each animal also gets a lognormal multiplier
    on the biological parameters (between-animal variability).
    """
    rng = np.random.default_rng(seed)
    nd, cd = list(necropsy_days), list(csf_days)
    base = di._bio_with(bio, true_f_uptake, true_f_csf_clear)
    tiss_mean, csf_mean = di._predict_streams_fast(phys, base, dose, site, nd, cd, csf_region)
    tiss_mean = tiss_mean.reshape(len(nd), N)

    def animal_means():
        if bsv_cv <= 0:
            return tiss_mean, csf_mean
        bf = np.exp(rng.normal(0.0, bsv_cv, 2))
        b = di._bio_with(bio, true_f_uptake * bf[0], true_f_csf_clear * bf[1])
        tm, cm = di._predict_streams_fast(phys, b, dose, site, nd, cd, csf_region)
        return tm.reshape(len(nd), N), cm

    csf_obs = np.zeros((len(cd), n_csf))
    for a in range(n_csf):
        _, cm = animal_means()
        csf_obs[:, a] = cm * rng.lognormal(0.0, residual_cv_csf, size=len(cd))
    tissue_obs = np.zeros((len(nd), N, n_tissue))
    for a in range(n_tissue):
        tm, _ = animal_means()
        tissue_obs[:, :, a] = tm * rng.lognormal(0.0, residual_cv_tissue, size=(len(nd), N))

    return {"csf_days": cd, "necropsy_days": nd, "csf_region": csf_region,
            "csf_obs": csf_obs, "tissue_obs": tissue_obs,
            "n_csf": n_csf, "n_tissue": n_tissue,
            "residual_cv_csf": residual_cv_csf, "residual_cv_tissue": residual_cv_tissue}


def study_to_observation(study):
    """Collapse a study to per-bin geometric means, in the dict shape the
    calibration routines consume (so a posterior can be fit to the study)."""
    tissue = _geomean(study["tissue_obs"], axis=2)            # (n_day, N)
    csf = _geomean(study["csf_obs"], axis=1)                  # (n_csf_day,)
    return {"tissue": tissue.reshape(-1), "csf": csf,
            "necropsy_days": study["necropsy_days"],
            "csf_days": study["csf_days"], "csf_region": study["csf_region"]}


# ---------------------------------------------------------------------------
# VPC band construction
# ---------------------------------------------------------------------------
def _bands(sim, pi, ci):
    """sim: (n_rep, n_bin, n_subj). For each requested percentile, return the
    across-replicate median and CI of that simulated percentile, per bin."""
    a_lo, a_hi = (100 - ci) / 2.0, 100 - (100 - ci) / 2.0
    out = {}
    for p in pi:
        qp = np.percentile(sim, p, axis=2)                   # (n_rep, n_bin)
        out[int(p)] = {"median": np.median(qp, axis=0),
                       "lo": np.percentile(qp, a_lo, axis=0),
                       "hi": np.percentile(qp, a_hi, axis=0)}
    return out


def _simulate_csf(phys, bio_prior, dose, site, study, samples, n_sim, rng):
    cd, reg, n, rcv = (study["csf_days"], study["csf_region"],
                       study["n_csf"], study["residual_cv_csf"])
    idx = rng.integers(0, len(samples), size=n_sim)
    sim = np.zeros((n_sim, len(cd), n))
    for k in range(n_sim):
        f_up, f_cl = samples[idx[k]]
        _, m = di._predict_streams_fast(phys, di._bio_with(bio_prior, f_up, f_cl),
                                        dose, site, study["necropsy_days"], cd, reg)
        sim[k] = m[:, None] * rng.lognormal(0.0, rcv, size=(len(cd), n))
    return sim


def vpc_csf(phys, bio_prior, dose, site, study, posterior_samples,
            n_sim=800, pi=(5, 50, 95), ci=90, prediction_corrected=False, seed=1):
    """Posterior-predictive VPC for the CSF PK stream (independent variable: time).

    With ``prediction_corrected=True`` returns a pcVPC: observations and
    simulations are normalized by the typical (posterior-median) prediction in
    each bin, the standard correction when the typical prediction varies across
    the time axis (Bergstrand et al., 2011).
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(posterior_samples)
    sim = _simulate_csf(phys, bio_prior, dose, site, study, samples, n_sim, rng)
    obs = study["csf_obs"].copy()                            # (n_bin, n)

    if prediction_corrected:
        fmed = np.median(samples, axis=0)
        _, typ = di._predict_streams_fast(phys, di._bio_with(bio_prior, *fmed), dose,
                                          site, study["necropsy_days"], study["csf_days"],
                                          study["csf_region"])
        corr = typ / _geomean(typ[None, :], axis=1)[0]       # per-bin / overall
        obs = obs / corr[:, None]
        sim = sim / corr[None, :, None]

    return {"time": np.array(study["csf_days"], float),
            "bands": _bands(sim, pi, ci),
            "obs_pct": {int(p): np.percentile(obs, p, axis=1) for p in pi},
            "pi": tuple(int(p) for p in pi), "ci": ci,
            "prediction_corrected": prediction_corrected,
            "ylabel": "CSF vector conc (vg/mL)"
            + (", prediction-corrected" if prediction_corrected else "")}


def vpc_tissue(phys, bio_prior, dose, site, study, posterior_samples,
               n_sim=800, pi=(5, 50, 95), ci=90, seed=2):
    """Posterior-predictive VPC for biodistribution, stratified by CNS region
    (independent variable: necropsy day)."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(posterior_samples)
    nd, n, rcv = study["necropsy_days"], study["n_tissue"], study["residual_cv_tissue"]
    idx = rng.integers(0, len(samples), size=n_sim)
    sim = np.zeros((n_sim, len(nd), N, n))
    for k in range(n_sim):
        f_up, f_cl = samples[idx[k]]
        mt, _ = di._predict_streams_fast(phys, di._bio_with(bio_prior, f_up, f_cl),
                                         dose, site, nd, study["csf_days"], study["csf_region"])
        mt = mt.reshape(len(nd), N)
        sim[k] = mt[:, :, None] * rng.lognormal(0.0, rcv, size=(len(nd), N, n))
    obs = study["tissue_obs"]                                # (n_day, N, n)

    per_region = []
    for r in range(N):
        per_region.append({
            "region": REGIONS[r], "time": np.array(nd, float),
            "bands": _bands(sim[:, :, r, :], pi, ci),
            "obs_pct": {int(p): np.percentile(obs[:, r, :], p, axis=1) for p in pi},
        })
    return {"per_region": per_region, "pi": tuple(int(p) for p in pi), "ci": ci,
            "ylabel": "transduced vg / region"}

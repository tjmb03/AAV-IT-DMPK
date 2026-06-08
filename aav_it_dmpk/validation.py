"""
Validation utilities: turning the validation plan into runnable checks.

These are organized by the layers a reviewer distinguishes (see the README's
Validation section). The functions here are pure compute (no plotting) so they
can be unit-tested and wired into CI; `demo_validation.py` runs them and draws
figures.

  Verification (is the math solved correctly?)
    mass_balance ................ vector genomes are conserved (states + sinks = dose)
    verify_against_matrix_exponential .. LSODA vs the exact linear solution
    convergence ................. solver-tolerance independence
    limiting_cases .............. zero-uptake and dose-linearity sanity

  Calibration / predictive validation (does it predict held-out data?)
    posterior_predictive_check .. do replicate datasets cover the observations?
    leave_one_cohort_out ........ fit on the rest, predict the held-out necropsy cohort
    simulation_based_calibration  is the Bayesian posterior calibrated (rank uniformity)?

  Sensitivity / uncertainty (which parameters move the decision?)
    sobol_indices / pipeline_sobol .. global (Sobol) first-order and total indices

NOTE: with the package's illustrative parameters these validate the *method and
implementation*. Quantitative validation requires the program's own data.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.stats import qmc

from .model import (_build_rhs, simulate, exposure_metrics, default_bio,
                    REGIONS, N, NSTATE, I_BLOOD, SITE_TO_REGION, _i)
from . import data_integration as di


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def mass_balance(phys, bio, dose=3e13, site="ICM", t_end=120.0):
    """Vector-genome conservation: states-in-system + cumulative removed == dose.

    Internal transfers (transport, CSF->tissue uptake, transduction, CSF->blood
    absorption, CSF->DRG uptake) conserve vg; the true sinks are first-order
    degradation/clearance/loss. We integrate the cumulative removed amount as an
    extra state of the same system (so it shares the solver's discretization) and
    check the running total equals the dose. (Protein/substrate states are not vg
    and are excluded.)
    """
    rhs = _build_rhs(phys, bio)
    kcd, ktd, kvl = bio["k_csf_deg"], bio["k_tis_deg"], bio["k_vcell_loss"]
    kvld, kbc = bio.get("k_vdrg_loss", 0.01), bio["k_blood_clear"]

    def aug(t, ya):
        y = ya[:NSTATE]
        sink = (kcd * sum(y[_i("Vcsf", r)] for r in range(N))
                + ktd * sum(y[_i("Vtis", r)] for r in range(N))
                + kvl * sum(y[_i("Vcell", r)] for r in range(N))
                + kvld * sum(y[_i("Vdrg", r)] for r in range(N))
                + kbc * y[I_BLOOD])
        return np.concatenate([rhs(t, y), [sink]])

    y0 = np.zeros(NSTATE + 1)
    y0[_i("Vcsf", SITE_TO_REGION[site])] = float(dose)
    teval = np.linspace(0.0, t_end, 200)
    sol = solve_ivp(aug, (0.0, t_end), y0, method="LSODA",
                    rtol=1e-9, atol=max(1.0, 1e-10 * dose), t_eval=teval)
    y = sol.y
    in_system = (sum(y[_i(b, r)] for b in ("Vcsf", "Vtis", "Vcell", "Vdrg")
                     for r in range(N)) + y[I_BLOOD])
    removed = y[NSTATE]
    balance = in_system + removed
    return {
        "t": sol.t, "in_system": in_system, "cumulative_removed": removed,
        "balance": balance, "dose": float(dose),
        "max_rel_imbalance": float(np.max(np.abs(balance - dose)) / dose),
    }


def verify_against_matrix_exponential(phys, bio, dose=3e13, site="ICM",
                                      times=(1.0, 7.0, 28.0, 90.0)):
    """Compare the numerical (LSODA) solution to the exact linear solution.

    The baseline model is dy/dt = A y, so y(t) = expm(A t) y0 exactly. We
    integrate at the *exact* requested times (so the comparison isolates
    integrator error, not output-grid interpolation) and report the worst
    relative error. (Linear model only.)
    """
    if bio.get("saturable") or bio.get("saturable_uptake"):
        raise ValueError("matrix-exponential check applies to the linear model only")
    rhs = _build_rhs(phys, bio)
    A = np.empty((NSTATE, NSTATE)); e = np.zeros(NSTATE)
    for j in range(NSTATE):
        e[j] = 1.0; A[:, j] = rhs(0.0, e); e[j] = 0.0
    y0 = np.zeros(NSTATE); y0[_i("Vcsf", SITE_TO_REGION[site])] = 1.0  # unit; linear
    tt = sorted(float(x) for x in times)
    sol = solve_ivp(rhs, (0.0, max(tt)), y0, method="LSODA",
                    rtol=1e-9, atol=1e-12, t_eval=tt)
    errs = []
    for k, tval in enumerate(tt):
        num = sol.y[:, k] * dose
        ana = (expm(A * tval) @ y0) * dose
        mask = np.abs(ana) > 1e-6 * dose
        errs.extend(list(np.abs(num[mask] - ana[mask]) / np.abs(ana[mask])))
    return {"times": tt, "max_rel_error": float(max(errs))}


def convergence(phys, bio, dose=3e13, site="ICM", t_end=120.0):
    """Solver-tolerance independence: integrate the same system at a coarse and a
    fine tolerance and compare the decision metric (transduced vg)."""
    rhs = _build_rhs(phys, bio)
    y0 = np.zeros(NSTATE); y0[_i("Vcsf", SITE_TO_REGION[site])] = float(dose)

    def integ(rtol, atol):
        sol = solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                        rtol=rtol, atol=atol, t_eval=[t_end])
        return sum(sol.y[_i("Vcell", r), -1] for r in range(N))

    coarse = integ(1e-6, max(1e-3, 1e-6 * dose))
    fine = integ(1e-10, max(1e-6, 1e-10 * dose))
    return {"transduced_coarse": float(coarse), "transduced_fine": float(fine),
            "rel_diff": float(abs(coarse - fine) / abs(fine))}


def limiting_cases(phys, site="ICM"):
    """Behavioural sanity: no uptake -> no transduction; baseline is dose-linear."""
    base = default_bio()
    b0 = dict(base); b0["k_uptake"] = 0.0
    _, s0 = simulate(phys, b0, dose=1e13, site=site)
    zero_transduced = float(sum(s0["Vcell"][r][-1] for r in range(N)))

    _, s1 = simulate(phys, base, dose=1.0, site=site)
    _, sD = simulate(phys, base, dose=1e13, site=site)
    denom = s1["Vcell"] * 1e13
    lin_err = float(np.max(np.abs(sD["Vcell"] - denom) / (denom + 1e-30)))
    return {
        "zero_uptake_transduced_vg": zero_transduced,
        "zero_uptake_ok": zero_transduced < 1e-6 * 1e13,
        "linearity_max_rel_err": lin_err,
        "linearity_ok": lin_err < 1e-5,
    }


# ---------------------------------------------------------------------------
# Calibration / predictive validation
# ---------------------------------------------------------------------------
def posterior_predictive_check(phys, bio_prior, dose, site, observed, samples,
                               n_rep=300, cv_tissue=0.20, cv_csf=0.15, seed=3):
    """Do datasets simulated from the posterior cover the observations?

    For posterior draws of (uptake, clearance), simulate replicate datasets
    (parameter uncertainty + assay noise) and report the fraction of observed
    points inside the 95% posterior-predictive interval. Well-calibrated ~ 0.95.
    """
    rng = np.random.default_rng(seed)
    nd, cd = observed["necropsy_days"], observed["csf_days"]
    reg = observed.get("csf_region", 0)
    samples = np.asarray(samples)
    idx = rng.integers(0, len(samples), size=n_rep)
    tiss, csf = [], []
    for k in idx:
        bio = di._bio_with(bio_prior, samples[k, 0], samples[k, 1])
        pt, pc = di._predict_streams_fast(phys, bio, dose, site, nd, cd, reg)
        tiss.append(pt * rng.lognormal(0.0, cv_tissue, size=pt.shape))
        csf.append(pc * rng.lognormal(0.0, cv_csf, size=pc.shape))
    tiss, csf = np.array(tiss), np.array(csf)

    def cover(obs, reps):
        lo, hi = np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)
        return float(((obs >= lo) & (obs <= hi)).mean()), lo, hi, np.median(reps, 0)

    ct, lt, ht, mt = cover(np.asarray(observed["tissue"]), tiss)
    cc, lc, hc, mc = cover(np.asarray(observed["csf"]), csf)
    n_t, n_c = len(observed["tissue"]), len(observed["csf"])
    return {
        "coverage_tissue": ct, "coverage_csf": cc,
        "coverage_overall": (ct * n_t + cc * n_c) / (n_t + n_c),
        "tissue_lo": lt, "tissue_hi": ht, "tissue_med": mt,
        "csf_lo": lc, "csf_hi": hc, "csf_med": mc,
    }


def leave_one_cohort_out(phys, bio_prior, dose, site, observed):
    """Fit on all-but-one necropsy cohort, predict the held-out cohort.

    The honest predictive test: each cohort is dropped in turn, the model is
    re-fit on the remaining tissue + the CSF stream, and the held-out cohort is
    predicted. Returns the per-region fold-error for each held-out day.
    """
    nd = list(observed["necropsy_days"]); cd = observed["csf_days"]
    reg = observed.get("csf_region", 0)
    tissue = np.asarray(observed["tissue"]).reshape(len(nd), N)
    out = []
    for i in range(len(nd)):
        keep = [j for j in range(len(nd)) if j != i]
        reduced = {"tissue": tissue[keep].reshape(-1), "csf": observed["csf"],
                   "necropsy_days": [nd[j] for j in keep], "csf_days": cd,
                   "csf_region": reg}
        upd = di.update_from_observations(phys, bio_prior, dose, site, reduced)
        pred = di.predict_region_transduced(phys, upd["bio_updated"], dose, site, nd[i])
        obs_i = tissue[i]
        fold = np.maximum(pred / obs_i, obs_i / pred)
        out.append({"held_out_day": nd[i], "predicted": pred, "observed": obs_i,
                    "fold_error": fold, "max_fold": float(np.max(fold))})
    return out


def simulation_based_calibration(phys, bio_prior, dose, site, schedule=None,
                                 n_sims=60, n_samples=300, burn=150, thin=2,
                                 prior_log_sd=0.5, seed=0):
    """SBC: draw truth from the prior, simulate, refit, rank the truth in the
    posterior. Calibrated inference -> ranks uniform on [0, n_posterior]."""
    rng = np.random.default_rng(seed)
    if schedule is None:
        schedule = {"necropsy_days": [14.0, 42.0], "csf_days": [0.5, 1.0, 2.0, 3.0]}
    ru, rc, L = [], [], None
    for _ in range(n_sims):
        fu, fc = np.exp(rng.normal(0.0, prior_log_sd, size=2))
        obs = di.make_synthetic_observation(
            phys, bio_prior, dose, site,
            schedule["necropsy_days"], schedule["csf_days"],
            true_f_uptake=fu, true_f_csf_clear=fc, seed=int(rng.integers(1_000_000_000)))
        post = di.bayesian_update_from_observations(
            phys, bio_prior, dose, site, obs, n_samples=n_samples, burn=burn,
            thin=thin, prior_log_sd=prior_log_sd, seed=int(rng.integers(1_000_000_000)))
        sm = post["samples"]; L = len(sm)
        ru.append(int(np.sum(sm[:, 0] < fu)))
        rc.append(int(np.sum(sm[:, 1] < fc)))
    return {"ranks_uptake": np.array(ru), "ranks_clearance": np.array(rc),
            "n_posterior": L, "n_sims": n_sims}


# ---------------------------------------------------------------------------
# Global sensitivity (Sobol)
# ---------------------------------------------------------------------------
def sobol_indices(model_func, bounds, n_base=256, seed=0):
    """First-order (S1) and total (ST) Sobol indices for a scalar output.

    Saltelli sampling with a scrambled Sobol sequence; Saltelli (2010) estimator
    for S1 and Jansen (1999) for ST. `model_func` maps a length-D parameter
    vector to a scalar. Total model evaluations = (2**ceil(log2 n_base)) * (D + 2).
    """
    D = len(bounds)
    m = int(np.ceil(np.log2(n_base)))
    X = qmc.Sobol(d=2 * D, scramble=True, seed=seed).random_base2(m=m)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    scale = lambda U: lo + U * (hi - lo)
    A, B = scale(X[:, :D]), scale(X[:, D:])
    ev = lambda M: np.array([model_func(M[k]) for k in range(M.shape[0])])
    yA, yB = ev(A), ev(B)
    varY = np.var(np.concatenate([yA, yB]))
    S1, ST = np.zeros(D), np.zeros(D)
    for i in range(D):
        AB = A.copy(); AB[:, i] = B[:, i]
        yABi = ev(AB)
        S1[i] = np.mean(yB * (yABi - yA)) / varY
        ST[i] = 0.5 * np.mean((yA - yABi) ** 2) / varY
    return {"S1": S1, "ST": ST, "var": float(varY), "n_eval": int(X.shape[0] * (D + 2))}


# parameter set for the pipeline Sobol convenience: (name, lo, hi, kind)
SOBOL_PARAMS = [
    ("k_uptake", 0.5, 2.0, "mult"),
    ("k_transduce", 0.5, 2.0, "mult"),
    ("k_csf_deg", 0.5, 2.0, "mult"),
    ("k_tis_deg", 0.5, 2.0, "mult"),
    ("k_vcell_loss", 0.3, 3.0, "mult"),
    ("disp_frac", 0.1, 0.5, "abs"),
]

# outputs computable from the terminal state (so the linear model can be
# evaluated with one matrix exponential instead of a full ODE solve)
_TERMINAL_OUTPUTS = {"transduced_per_g_cns", "transduced_total_vg",
                     "drg_total_vg", "protein_terminal_AU"}


def pipeline_sobol(phys, base_bio=None, dose=3e13, site="ICM",
                   output="transduced_per_g_cns", n_base=256, seed=0, t_end=120.0):
    """Sobol indices for a decision output over the conserved biological params.

    Multiplier parameters scale the baseline rate constant; 'abs' parameters are
    set directly. For terminal-state outputs on the linear model, each evaluation
    uses the exact matrix exponential (fast and exact); otherwise it falls back to
    a full ODE solve. Returns the indices plus the parameter names.
    """
    base = dict(base_bio or default_bio())
    names = [p[0] for p in SOBOL_PARAMS]
    bounds = [(p[1], p[2]) for p in SOBOL_PARAMS]
    kinds = [p[3] for p in SOBOL_PARAMS]
    cns_mass = float(sum(phys["tissue_mass_g"]))
    r0 = SITE_TO_REGION[site]
    fast = (output in _TERMINAL_OUTPUTS
            and not (base.get("saturable") or base.get("saturable_uptake")))

    def make_bio(v):
        bio = dict(base)
        for name, val, kind in zip(names, v, kinds):
            bio[name] = base[name] * val if kind == "mult" else float(val)
        return bio

    def terminal_expm(bio):
        rhs = _build_rhs(phys, bio)
        A = np.empty((NSTATE, NSTATE)); e = np.zeros(NSTATE)
        for j in range(NSTATE):
            e[j] = 1.0; A[:, j] = rhs(0.0, e); e[j] = 0.0
        y0 = np.zeros(NSTATE); y0[_i("Vcsf", r0)] = float(dose)
        yT = expm(A * t_end) @ y0
        if output == "drg_total_vg":
            return float(sum(yT[_i("Vdrg", r)] for r in range(N)))
        if output == "protein_terminal_AU":
            return float(sum(yT[_i("P", r)] for r in range(N)))
        tot = float(sum(yT[_i("Vcell", r)] for r in range(N)))
        return tot / cns_mass if output == "transduced_per_g_cns" else tot

    def f(v):
        bio = make_bio(v)
        if fast:
            return terminal_expm(bio)
        t, s = simulate(phys, bio, dose=dose, site=site, t_end=t_end)
        return exposure_metrics(t, s, phys)[output]

    res = sobol_indices(f, bounds, n_base=n_base, seed=seed)
    res["names"] = names
    res["output"] = output
    return res

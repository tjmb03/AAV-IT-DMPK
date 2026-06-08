"""
DRG safety and pharmacodynamics (PD) readouts.

Two extensions live here because both sit "downstream" of the vector kinetics in
model.py and are cleaner to compute on top of a vector trajectory than to fold
into the core ODE state:

1. DRG safety -- dorsal root ganglion sensory-neuron toxicity is the
   dose-limiting finding in NHP studies of IT-delivered AAV, and it tracks the
   per-neuron vector load. `drg_load_table` converts the ganglionic vector state
   (model Vdrg, switched on by capsid serotype) into the qPCR-style readout a
   tox reviewer expects (vg per diploid genome) and reports the margin to an
   (illustrative) tolerability threshold.

2. PD / efficacy -- for a secreted therapeutic enzyme (the canonical CNS gene
   therapy paradigm, e.g. a lysosomal enzyme delivered by AAV9), the chain is:
   transduced cells express enzyme -> a fraction is secreted -> neighbouring
   cells take it up (cross-correction) -> intracellular + cross-corrected enzyme
   clears accumulated substrate. `pd_response` integrates this cascade and
   returns the fractional substrate reduction per region -- a mechanistic PD
   endpoint that can anchor dose selection (`dose_for_pd_target`).

Numerics: with the default (linear) vector model the internalized trajectory is
exactly (unit-dose trajectory) x dose. We exploit that: run the vector model
once at unit dose, then integrate the small, well-scaled enzyme/substrate system
with the actual dose entering only the (nonlinear) substrate-clearance term.
This avoids mixing vg-scale (~1e13) and substrate-scale (~1e2) states in one
stiff solve. (If saturable uptake is also engaged the vector kinetics are no
longer linear; combining the two is a documented future extension.)

ALL PD AND TOX PARAMETERS ARE ILLUSTRATIVE.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .model import simulate, REGIONS, N

# Illustrative DRG tolerability threshold. NHP IT-AAV studies associate sensory
# neuronopathy with high ganglionic transduction; the numeric level here is a
# placeholder for a program-specific, histopathology-anchored threshold.
DRG_TOX_THRESHOLD_VG_PER_DIPLOID = 1500.0

PG_PER_DIPLOID_GENOME = 6.6
UG_PER_DIPLOID_GENOME = 6.6e-6

GANGLIA = ["cranial_ganglia", "thoracic_DRG", "lumbar_DRG"]


# A secreted-enzyme transgene (illustrative). Rate units 1/day; enzyme tracked
# per unit dose so magnitudes stay O(1) (see module docstring).
DEFAULT_TRANSGENE = dict(
    label="Secreted lysosomal enzyme (AAV9, illustrative)",
    indication="CNS lysosomal storage disorder (illustrative)",
    k_express_pd=1.0,     # enzyme produced per internalized vg per day (per unit dose)
    k_pdeg_pd=0.10,       # intracellular enzyme turnover
    k_secrete=0.20,       # fraction-rate of enzyme secreted to ECF
    k_recapture=0.50,     # ECF enzyme taken up by bystander cells (cross-correction)
    k_eecf_deg=0.30,      # extracellular enzyme degradation
    S0=100.0,             # baseline (disease) substrate level, AU (= 100%)
    k_deg_S=0.05,         # enzyme-independent substrate turnover
    k_cat=1.0e-13,        # substrate cleared per (enzyme x dose) per day
)


def drg_load_table(phys, bio, dose, site, day,
                   threshold=DRG_TOX_THRESHOLD_VG_PER_DIPLOID):
    """Ganglionic vector load and safety margin at a necropsy day.

    Requires a serotype/biology with k_uptake_drg > 0 (otherwise the DRG state
    is identically zero). Returns a tidy DataFrame.
    """
    t, states = simulate(phys, bio, dose=dose, site=site, t_end=day)
    dna_per_g = phys["dna_ug_per_g"]
    drg_mass = phys["drg_mass_g"]
    rows = []
    for r, gname in enumerate(GANGLIA):
        vdrg = float(states["Vdrg"][r][-1])
        ug_dna = drg_mass[r] * dna_per_g
        vg_per_ug = vdrg / ug_dna if ug_dna > 0 else 0.0
        vg_per_dip = vg_per_ug * UG_PER_DIPLOID_GENOME
        margin = (threshold / vg_per_dip) if vg_per_dip > 0 else np.inf
        rows.append({
            "ganglion": gname,
            "vector_genomes_vg": vdrg,
            "vg_per_diploid_genome": vg_per_dip,
            "safety_margin_x": margin,
            "exceeds_threshold": vg_per_dip > threshold,
        })
    return pd.DataFrame(rows)


def _vcell_unit_interp(phys, bio, site, t_end):
    """Unit-dose internalized-vector trajectories per region, as interpolators.

    Forces linear vector kinetics (saturable modes off) so the result scales
    exactly with dose. DRG uptake, if on, is retained (still linear)."""
    bio_lin = dict(bio)
    bio_lin["saturable"] = False
    bio_lin["saturable_uptake"] = False
    t, states = simulate(phys, bio_lin, dose=1.0, site=site, t_end=t_end)
    vhat = [states["Vcell"][r] for r in range(N)]   # per unit dose
    return t, vhat


def pd_response(phys, bio, dose, site, transgene=None, t_end=120.0):
    """Integrate the enzyme -> secretion -> cross-correction -> substrate cascade.

    Returns a dict with time, substrate S[r](t), fractional reduction[r](t),
    terminal reduction per region, and the terminal brain (region 0) reduction.
    """
    tg = dict(DEFAULT_TRANSGENE if transgene is None else transgene)
    t_grid, vhat = _vcell_unit_interp(phys, bio, site, t_end)

    kexp, kpd = tg["k_express_pd"], tg["k_pdeg_pd"]
    ksec, krec, kedeg = tg["k_secrete"], tg["k_recapture"], tg["k_eecf_deg"]
    S0, kdegS, kcat = tg["S0"], tg["k_deg_S"], tg["k_cat"]
    dose = float(dose)

    def vh(r, t):
        return np.interp(t, t_grid, vhat[r])

    # state z = [E_hat(N), Eecf_hat(N), S(N)]; E,Eecf are per unit dose
    def rhs(t, z):
        E = z[0:N]; Eecf = z[N:2 * N]; S = z[2 * N:3 * N]
        dE = np.array([kexp * vh(r, t) - (kpd + ksec) * E[r] for r in range(N)])
        dEecf = ksec * E - (krec + kedeg) * Eecf
        # actual enzyme available to clear substrate = dose x (intra + extra)
        clear = kcat * dose * (E + Eecf)
        dS = kdegS * (S0 - S) - clear * S
        return np.concatenate([dE, dEecf, dS])

    z0 = np.concatenate([np.zeros(N), np.zeros(N), np.full(N, S0)])
    teval = np.unique(np.concatenate([
        np.linspace(0.0, min(20.0, t_end), 120),
        np.linspace(min(20.0, t_end), t_end, 160),
    ]))
    sol = solve_ivp(rhs, (0.0, t_end), z0, t_eval=teval, method="LSODA",
                    rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"PD integration failed: {sol.message}")
    S = sol.y[2 * N:3 * N, :]
    reduction = 1.0 - S / S0
    return {
        "t": sol.t,
        "S": S,
        "reduction": reduction,
        "terminal_reduction_per_region": [float(reduction[r][-1]) for r in range(N)],
        "brain_terminal_reduction": float(reduction[0][-1]),
        "transgene": tg,
    }


def brain_reduction_at_dose(phys, bio, site, dose, transgene=None, t_end=120.0):
    return pd_response(phys, bio, dose, site, transgene, t_end)["brain_terminal_reduction"]


def dose_for_pd_target(phys, bio, site, target_reduction_brain,
                       transgene=None, t_end=120.0, bracket=(1e10, 1e16),
                       tol=1e-3, maxit=60):
    """Smallest total dose whose terminal brain substrate reduction >= target.

    Terminal reduction is monotone increasing in dose, so we bisect.
    """
    lo, hi = bracket
    f = lambda d: brain_reduction_at_dose(phys, bio, site, d, transgene, t_end) \
        - target_reduction_brain
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return lo
    if fhi < 0:
        raise ValueError("target reduction not reachable within dose bracket; "
                         "raise the upper bound or check transgene potency.")
    for _ in range(maxit):
        mid = np.sqrt(lo * hi)          # geometric bisection (dose spans decades)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if fm < 0:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)

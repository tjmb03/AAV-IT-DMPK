"""
Mechanistic (not compartment-fitting) model of intrathecal AAV disposition.

State, per CSF/CNS region r in {cranial, thoracic, lumbar}:
    Vcsf[r]   free vector in the CSF segment            (vg)
    Vtis[r]   free vector in tissue extracellular space (vg)
    Vcell[r]  internalized / transduced vector          (vg)   <- biodistribution readout
    P[r]      expressed transgene product               (arbitrary protein units)
    Vdrg[r]   internalized vector in ganglia (DRG)       (vg)   <- safety readout
plus one systemic compartment:
    Vblood    free capsid that has effluxed to blood     (vg)

Processes (all first order unless a `saturable*` flag is set):

  CSF transport      bulk flow Q rostral->caudal carries vector between adjacent
                     segments; symmetric dispersion D allows limited rostral
                     spread (this is what makes a lumbar dose reach the brain
                     poorly relative to a cisternal/ICM dose).
  CSF clearance      degradation (k_csf_deg) + absorption to blood (distributed,
                     at the measured CSF turnover rate).
  Tissue uptake      CSF -> tissue ECF (k_uptake); optional receptor-limited
                     saturation at high vector concentration (saturable_uptake).
  Transduction       tissue ECF -> internalized (k_transduce); optional saturation.
  DRG uptake         CSF -> ganglionic neurons (k_uptake_drg), lumped uptake +
                     transduction. Dorsal root ganglia are bathed in CSF along
                     the nerve roots and take up vector avidly; their per-neuron
                     vector load is the driver of the sensory-neuron toxicity
                     that has been dose-limiting in NHP IT studies. Defaults OFF
                     (0); set by capsid serotype (see apply_serotype) or directly.
  Vector persistence slow loss of internalized vector (k_vcell_loss / k_vdrg_loss);
                     ~stable in post-mitotic neurons.
  Expression         internalized vector -> protein (k_express) with turnover
                     k_pdeg. (This P is a simple expression proxy; the full
                     enzyme / cross-correction / substrate PD cascade lives in
                     pd_safety.py, driven by Vcell.)

Because the default model is linear in dose, we integrate a UNIT dose and scale
the trajectories by the requested dose. This keeps the (stiff) system well
scaled and makes the dose->exposure relationship exact and transparent. DRG
uptake is first order and preserves linearity; the saturable modes do not, so
with `saturable` or `saturable_uptake` we integrate at the true dose instead.
"""

import numpy as np
from scipy.integrate import solve_ivp

REGIONS = ["cranial", "thoracic", "lumbar"]
TISSUES = ["brain", "thoracic_cord", "lumbar_cord"]
N = 3

# Block-structured state layout: each variable occupies a contiguous block of N,
# followed by the scalar blood compartment. Adding a state type = adding a block.
_BLOCKS = ("Vcsf", "Vtis", "Vcell", "P", "Vdrg")
_BASE = {name: i * N for i, name in enumerate(_BLOCKS)}
I_BLOOD = len(_BLOCKS) * N
NSTATE = len(_BLOCKS) * N + 1


def _i(block, r):
    return _BASE[block] + r


SITE_TO_REGION = {
    "ICM": 0, "cisternal": 0, "ICV": 0, "cervical": 0,
    "thoracic": 1,
    "lumbar": 2, "IT_lumbar": 2,
}


def default_bio():
    """Conserved (species-invariant) biological parameters. Units: 1/day.

    Note on transport: longitudinal movement of a viral vector along the
    neuraxis is dominated by dispersion / pulsatile mixing rather than net bulk
    flow, and CSF absorption to blood is distributed (arachnoid granulations,
    nerve-root sleeves). We therefore model neighbour exchange as dispersion D
    (a fraction of CSF flow Q) with a small caudal drift, and absorption as a
    per-segment first-order loss whose rate equals the measured CSF turnover
    (Q / CSF volume) -- a physiological, species-specific quantity. This
    reproduces the key qualitative behaviour: a cisternal/ICM dose gives a
    rostro-caudal *decreasing* gradient (strong brain), a lumbar dose the
    reverse (weak brain).

    The DRG and saturable-uptake parameters default to inert values so that the
    base model is unchanged; they are exercised by the extension analyses.
    """
    return dict(
        k_uptake=0.30,           # CSF free -> tissue ECF
        k_transduce=0.80,        # tissue ECF -> internalized / transduced
        k_tis_deg=0.20,          # degradation of free vector in tissue ECF
        k_csf_deg=0.50,          # degradation of free vector in CSF
        k_vcell_loss=0.01,       # loss of internalized vector (slow)
        k_express=1.0,           # protein produced per internalized vg per day (AU)
        k_pdeg=0.10,             # protein turnover (t1/2 ~ 7 d)
        k_blood_clear=5.0,       # systemic clearance of free capsid
        disp_frac=0.30,          # neighbour dispersion as a fraction of flow Q
        drift_frac=0.15,         # small net caudal drift as a fraction of flow Q
        # --- transduction saturation (pre-existing) ---
        saturable=False,
        Km_conc=1e9,             # vg/mL; only used when saturable=True
        # --- DRG / ganglionic uptake (extension; default OFF) ---
        k_uptake_drg=0.0,        # CSF -> DRG neuron uptake+transduction (1/day)
        k_vdrg_loss=0.01,        # loss of internalized vector in DRG (slow)
        # --- receptor-limited (saturable) parenchymal uptake (extension; OFF) ---
        saturable_uptake=False,
        Km_uptake_conc=1e11,     # vg/mL; CSF conc at half-maximal uptake rate
    )


def apply_serotype(bio, serotype):
    """Return a copy of `bio` with capsid-serotype tropism applied.

    `serotype` may be a name understood by physiology.get_serotype or a dict
    with keys f_uptake, f_transduce, k_uptake_drg. Serotype scales the conserved
    biology: parenchymal uptake/transduction efficiency and DRG avidity. This is
    what couples capsid choice to both efficacy and the DRG safety signal.
    """
    if isinstance(serotype, str):
        from .physiology import get_serotype
        serotype = get_serotype(serotype)
    out = dict(bio)
    out["k_uptake"] = bio["k_uptake"] * serotype.get("f_uptake", 1.0)
    out["k_transduce"] = bio["k_transduce"] * serotype.get("f_transduce", 1.0)
    out["k_uptake_drg"] = serotype.get("k_uptake_drg", bio.get("k_uptake_drg", 0.0))
    return out


def _build_rhs(phys, bio):
    Vol = np.asarray(phys["csf_seg_mL"], dtype=float)
    Q = float(phys["Q_csf_mL_per_day"])
    csf_total = float(phys["csf_total_mL"])
    D = bio["disp_frac"] * Q                  # dispersion (mL/day)
    v = bio["drift_frac"] * Q                  # caudal drift (mL/day)
    k_csf_abs = Q / csf_total                  # distributed absorption = CSF turnover (1/day)
    ku, ktr, ktd = bio["k_uptake"], bio["k_transduce"], bio["k_tis_deg"]
    kcd, kvl = bio["k_csf_deg"], bio["k_vcell_loss"]
    kex, kpd, kbc = bio["k_express"], bio["k_pdeg"], bio["k_blood_clear"]
    sat, Km = bio["saturable"], bio["Km_conc"]
    kdrg = bio.get("k_uptake_drg", 0.0)
    kvld = bio.get("k_vdrg_loss", 0.01)
    sat_up = bio.get("saturable_uptake", False)
    Km_up = bio.get("Km_uptake_conc", 1e11)

    def rhs(t, y):
        dy = np.zeros_like(y)
        conc = np.array([y[_i("Vcsf", r)] / Vol[r] for r in range(N)])

        # longitudinal transport: dispersion (symmetric) + small caudal drift
        for r in range(N - 1):
            J = D * (conc[r] - conc[r + 1]) + v * conc[r]
            dy[_i("Vcsf", r)] -= J
            dy[_i("Vcsf", r + 1)] += J

        # per-region kinetics
        for r in range(N):
            Vcsf = y[_i("Vcsf", r)]
            Vtis = y[_i("Vtis", r)]
            Vcell = y[_i("Vcell", r)]
            Vdrg = y[_i("Vdrg", r)]

            if sat_up:                              # receptor-limited uptake
                ccsf = Vcsf / Vol[r]
                uptake = ku * Vcsf / (1.0 + ccsf / Km_up)
            else:
                uptake = ku * Vcsf

            uptake_drg = kdrg * Vcsf                 # CSF -> DRG (parallel sink)
            absorb = k_csf_abs * Vcsf                 # CSF -> blood (distributed)
            dy[_i("Vcsf", r)] += -uptake - uptake_drg - kcd * Vcsf - absorb
            dy[I_BLOOD] += absorb

            if sat:                                  # optional transduction saturation
                ctis = Vtis / Vol[r]
                transduce = ktr * Vtis / (1.0 + ctis / Km)
            else:
                transduce = ktr * Vtis

            dy[_i("Vtis", r)] += uptake - transduce - ktd * Vtis
            dy[_i("Vcell", r)] += transduce - kvl * Vcell
            dy[_i("P", r)] += kex * Vcell - kpd * y[_i("P", r)]
            dy[_i("Vdrg", r)] += uptake_drg - kvld * Vdrg

        dy[I_BLOOD] += -kbc * y[I_BLOOD]
        return dy

    return rhs


def simulate(phys, bio, dose, site="ICM", t_end=120.0):
    """Integrate the model and return (t, states) with states already scaled to `dose`.

    states is a dict of arrays keyed 'Vcsf','Vtis','Vcell','P','Vdrg' each shape
    (N, len(t)), plus 'Vblood' shape (len(t),).
    """
    if site not in SITE_TO_REGION:
        raise KeyError(f"Unknown site '{site}'. Options: {list(SITE_TO_REGION)}")
    r0 = SITE_TO_REGION[site]
    rhs = _build_rhs(phys, bio)

    if t_end <= 10.0:
        t_eval = np.linspace(0.0, t_end, 200)
    else:
        t_eval = np.unique(np.concatenate([
            np.linspace(0.0, 10.0, 200),
            np.linspace(10.0, t_end, 220),
        ]))

    linear = not (bio["saturable"] or bio.get("saturable_uptake", False))
    y0 = np.zeros(NSTATE)
    y0[_i("Vcsf", r0)] = 1.0 if linear else float(dose)
    rtol = 1e-8 if linear else 1e-7
    atol = 1e-12 if linear else max(1e-3, 1e-9 * dose)
    sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=t_eval,
                    method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    scale = float(dose) if linear else 1.0
    states = {
        "Vcsf": sol.y[[_i("Vcsf", r) for r in range(N)], :] * scale,
        "Vtis": sol.y[[_i("Vtis", r) for r in range(N)], :] * scale,
        "Vcell": sol.y[[_i("Vcell", r) for r in range(N)], :] * scale,
        "P": sol.y[[_i("P", r) for r in range(N)], :] * scale,
        "Vdrg": sol.y[[_i("Vdrg", r) for r in range(N)], :] * scale,
        "Vblood": sol.y[I_BLOOD, :] * scale,
    }
    return sol.t, states


def exposure_metrics(t, states, phys):
    """Summarise exposure / transduction / expression / DRG load from a simulation."""
    Vol = np.asarray(phys["csf_seg_mL"], dtype=float)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    csf_auc = [float(_trapz(states["Vcsf"][r] / Vol[r], t)) for r in range(N)]
    transduced_total = float(sum(states["Vcell"][r][-1] for r in range(N)))
    drg_total = float(sum(states["Vdrg"][r][-1] for r in range(N)))
    cns_mass = float(sum(phys["tissue_mass_g"]))
    protein_total_t = sum(states["P"][r] for r in range(N))
    return {
        "csf_auc_vg_day_per_mL": csf_auc,          # per segment
        "csf_auc_cranial": csf_auc[0],
        "transduced_total_vg": transduced_total,
        "transduced_per_g_cns": transduced_total / cns_mass,
        "protein_peak_AU": float(np.max(protein_total_t)),
        "protein_terminal_AU": float(protein_total_t[-1]),
        "drg_total_vg": drg_total,
        "drg_terminal_per_region": [float(states["Vdrg"][r][-1]) for r in range(N)],
    }

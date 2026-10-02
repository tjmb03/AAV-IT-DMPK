"""
First-in-human dose bracketing, anchored on a tox-study NOAEL.

The translation module answers "what human dose reproduces an efficacious
animal exposure?" That is an *efficacious* dose, not a starting dose. This
module implements the FIH question instead: given a NOAEL observed in a tox
species, what human dose carries the same safety-tissue exposure, and is there
a window between that (divided by a safety factor) and the lowest dose with
plausible activity?

The sequence matters, and it is deliberately the inverse of the obvious one:

  1. The tox study gives a NOAEL *dose* in the tox species.
  2. The model converts that dose into the DRG exposure it produced
     (`threshold_from_noael`). THAT is the program's tolerability threshold --
     derived from the study's own histopathology, not chosen a priori. Setting
     a threshold by hand and then computing a dose from it is circular.
  3. The human dose carrying the same DRG exposure is the exposure-matched
     human-equivalent dose (`dose_for_drg_load`) -- which is generally *not*
     the body-weight-scaled dose.
  4. A safety factor is applied to that HED to give candidate starting doses.
  5. The PD model gives the efficacy floor. A window exists only if the
     safety-factored start clears it.

Everything here inherits the package's illustrative parameters: these are the
right quantities to compute, not numbers to act on. The NOAEL, the
histopathology behind it, and the choice of safety factor come from outside
the model.
"""

import numpy as np

from . import pd_safety as ps
from . import data_integration as di


DEFAULT_SAFETY_FACTORS = (3.0, 6.0, 10.0)


def worst_drg_load(phys, bio, dose, site="ICM", day=28.0):
    """Highest ganglionic load (vg per diploid genome) across ganglia."""
    tab = ps.drg_load_table(phys, bio, dose, site, day)
    return float(tab["vg_per_diploid_genome"].max())


def dose_for_drg_load(phys, bio, target_load, site="ICM", day=28.0,
                      bracket=(1e9, 1e16), maxit=60):
    """Invert `worst_drg_load`: the dose whose worst ganglion hits target_load.

    Load is monotone increasing in dose, so bisect geometrically.
    """
    lo, hi = bracket
    for _ in range(maxit):
        mid = np.sqrt(lo * hi)
        if worst_drg_load(phys, bio, mid, site, day) > target_load:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def threshold_from_noael(tox_phys, bio, noael_dose, site="ICM", day=28.0):
    """The DRG exposure produced by the NOAEL dose in the tox species.

    This is the program-specific tolerability threshold: the exposure the
    study's own histopathology found to be without adverse effect.
    """
    return worst_drg_load(tox_phys, bio, noael_dose, site, day)


def bodyweight_scaled_dose(tox_phys, human_phys, noael_dose):
    """Naive body-weight (mg/kg-style) scaling, for comparison only."""
    return float(noael_dose * human_phys["body_weight_kg"] / tox_phys["body_weight_kg"])


def _perturb_physiology(phys, csf_mult=1.0, drg_mult=1.0):
    """Scale the physiological quantities the DRG ceiling actually depends on.

    CSF volume is scaled (segments with it) holding CSF production Q fixed, so
    turnover moves accordingly -- volume and production rate are separately
    measured quantities, not a single parameter.
    """
    p = {k: (list(v) if isinstance(v, list) else v) for k, v in phys.items()}
    if csf_mult != 1.0:
        p["csf_total_mL"] = p["csf_total_mL"] * csf_mult
        p["csf_seg_mL"] = [v * csf_mult for v in p["csf_seg_mL"]]
    if drg_mult != 1.0:
        p["drg_mass_g"] = [v * drg_mult for v in p["drg_mass_g"]]
    return p


def propagate_fih_bracket(tox_phys, human_phys, bio_prior, noael_dose,
                          posterior_samples=None, site="ICM", day=28.0,
                          safety_factors=DEFAULT_SAFETY_FACTORS,
                          pd_floor_reduction=0.30, loael_dose=None,
                          noael_log10_sd=0.0, physiol_cv=0.0,
                          n_draws=200, transgene=None, t_end=120.0, seed=0):
    """Propagate uncertainty through the FIH bracket and report window survival.

    Three sources are propagated, because they do NOT act the same way:

      * **Biology** (posterior over the uptake / CSF-clearance multipliers).
        The DRG ceiling is a *ratio* anchor -- the NOAEL's exposure in the tox
        species matched in human -- so a global biology multiplier largely
        cancels and contributes almost nothing to ceiling uncertainty. It does
        NOT cancel in the PD floor, which is an *absolute* anchor. Propagating
        the posterior alone therefore produces a falsely tight ceiling.
      * **NOAEL censoring**: the true no-effect dose lies between the NOAEL and
        the LOAEL. Pass `loael_dose` for a log-uniform draw across that
        (honest) interval, or `noael_log10_sd` for a lognormal.
      * **Human physiology**: CSF volume and DRG mass, which do not cancel and
        in practice dominate the ceiling.

    Returns percentiles for ceiling and floor, per-safety-factor starting-dose
    percentiles and P(window survives), plus a one-source-at-a-time attribution
    of what actually drives the ceiling spread.
    """
    rng = np.random.default_rng(seed)
    nominal = fih_bracket(tox_phys, human_phys, bio_prior, noael_dose, site, day,
                          safety_factors, pd_floor_reduction, pd_floor_reduction,
                          transgene, t_end)

    def draw_biology():
        if posterior_samples is None or len(posterior_samples) == 0:
            return 1.0, 1.0
        s = np.asarray(posterior_samples)
        return tuple(s[rng.integers(0, len(s))][:2])

    def draw_noael():
        if loael_dose is not None and loael_dose > noael_dose:
            return float(np.exp(rng.uniform(np.log(noael_dose), np.log(loael_dose))))
        if noael_log10_sd > 0:
            return float(noael_dose * 10.0 ** rng.normal(0.0, noael_log10_sd))
        return float(noael_dose)

    def draw_physiol():
        if physiol_cv <= 0:
            return 1.0, 1.0
        return (float(rng.lognormal(0.0, physiol_cv)),
                float(rng.lognormal(0.0, physiol_cv)))

    def one_draw(bio_on=True, noael_on=True, phys_on=True, need_floor=True):
        f_up, f_cl = draw_biology() if bio_on else (1.0, 1.0)
        nd = draw_noael() if noael_on else noael_dose
        cm, dm = draw_physiol() if phys_on else (1.0, 1.0)
        b = di._bio_with(bio_prior, f_up, f_cl) if bio_on else dict(bio_prior)
        hp = _perturb_physiology(human_phys, cm, dm)
        # DRG load is exactly linear in dose, so the ceiling is closed-form:
        #   ceiling = NOAEL * L1_tox / L1_human
        l_tox = worst_drg_load(tox_phys, b, 1.0, site, day)
        l_hum = worst_drg_load(hp, b, 1.0, site, day)
        ceiling = nd * l_tox / l_hum
        floor = (ps.dose_for_pd_target(hp, b, site, pd_floor_reduction,
                                       transgene=transgene, t_end=t_end)
                 if need_floor else np.nan)
        return ceiling, floor

    ceilings, floors = [], []
    for _ in range(int(n_draws)):
        c, f = one_draw()
        ceilings.append(c); floors.append(f)
    ceilings, floors = np.array(ceilings), np.array(floors)

    pct = lambda a: {"p2.5": float(np.percentile(a, 2.5)),
                     "median": float(np.percentile(a, 50)),
                     "p97.5": float(np.percentile(a, 97.5))}

    starts = []
    for sf in safety_factors:
        s = ceilings / float(sf)
        starts.append({
            "safety_factor": float(sf),
            **{f"start_{k}": v for k, v in pct(s).items()},
            "p_window_survives": float(np.mean(s >= floors)),
            # conservative-tail test: low ceiling vs high floor
            "survives_conservative_tail": bool(
                np.percentile(s, 2.5) >= np.percentile(floors, 97.5)),
        })

    # one-source-at-a-time attribution for the ceiling (cheap: no floor solve)
    attrib = {}
    n_att = max(40, int(n_draws) // 2)
    for lab, kw in [("biology_only", dict(bio_on=True, noael_on=False, phys_on=False)),
                    ("noael_only", dict(bio_on=False, noael_on=True, phys_on=False)),
                    ("physiology_only", dict(bio_on=False, noael_on=False, phys_on=True))]:
        vals = np.array([one_draw(need_floor=False, **kw)[0] for _ in range(n_att)])
        attrib[lab] = float(np.percentile(vals, 97.5) / np.percentile(vals, 2.5))

    return {
        "nominal": nominal,
        "ceiling": pct(ceilings), "floor": pct(floors),
        "ceiling_spread_fold": float(np.percentile(ceilings, 97.5)
                                     / np.percentile(ceilings, 2.5)),
        "floor_spread_fold": float(np.percentile(floors, 97.5)
                                   / np.percentile(floors, 2.5)),
        "starting_doses": starts,
        "ceiling_attribution_fold": attrib,
        "n_draws": int(n_draws), "site": site, "day": float(day),
    }


def fih_bracket(tox_phys, human_phys, bio, noael_dose, site="ICM", day=28.0,
                safety_factors=DEFAULT_SAFETY_FACTORS,
                pd_floor_reduction=0.30, pd_target_reduction=0.70,
                transgene=None, t_end=120.0):
    """Full FIH bracket from a tox-species NOAEL.

    Returns the derived threshold, the exposure-matched human dose, candidate
    starting doses by safety factor, the PD floor/target, and whether each
    candidate start clears the floor (i.e. whether a window exists).
    """
    thr = threshold_from_noael(tox_phys, bio, noael_dose, site, day)
    hed = dose_for_drg_load(human_phys, bio, thr, site, day)
    bw = bodyweight_scaled_dose(tox_phys, human_phys, noael_dose)

    floor = ps.dose_for_pd_target(human_phys, bio, site, pd_floor_reduction,
                                  transgene=transgene, t_end=t_end)
    target = ps.dose_for_pd_target(human_phys, bio, site, pd_target_reduction,
                                   transgene=transgene, t_end=t_end)

    starts = []
    for sf in safety_factors:
        d = hed / float(sf)
        starts.append({
            "safety_factor": float(sf),
            "starting_dose_vg": float(d),
            "clears_pd_floor": bool(d >= floor),
            "fold_above_floor": float(d / floor) if floor > 0 else np.inf,
            "fold_of_pd_target": float(d / target) if target > 0 else np.inf,
        })

    return {
        "noael_dose_vg": float(noael_dose),
        "derived_threshold_vg_per_diploid": float(thr),
        "human_dose_matched_exposure_vg": float(hed),
        "bodyweight_scaled_dose_vg": bw,
        "bodyweight_vs_exposure_fold": float(bw / hed) if hed > 0 else np.inf,
        "pd_floor_vg": float(floor),
        "pd_target_vg": float(target),
        "pd_floor_reduction": float(pd_floor_reduction),
        "pd_target_reduction": float(pd_target_reduction),
        "starting_doses": starts,
        "window_exists": bool(any(s["clears_pd_floor"] for s in starts)),
        "site": site,
        "day": float(day),
    }

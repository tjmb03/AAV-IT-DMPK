"""
Cross-species dose translation: mouse -> NHP -> human.

The strategic question for an IT AAV program is *not* "what is the per-kg dose"
but "what dose in the next species reproduces the exposure that drove efficacy
(or toxicity) in the current species, and on what normalization basis is that
dose most stable?" Body-weight (mg/kg-style) scaling is usually the wrong basis
for CSF-delivered agents, because the determinant of CNS exposure is CSF volume
and turnover, not body mass.

This module:
  * matches a chosen exposure metric across species to project a dose, and
  * reports that dose on four normalization bases (total, /kg, /g-brain, /mL-CSF)
    so you can see which basis is most conserved -- and that the answer depends
    on which exposure metric is presumed to drive the effect.

Because the default model is linear in dose, the projected dose to match
metric value M* is simply  M* / (metric per unit dose in the target species).
"""

from types import SimpleNamespace
import numpy as np

from .model import simulate, exposure_metrics

# exposure metrics that can anchor a translation
METRIC_CHOICES = ("csf_auc_cranial", "transduced_per_g_cns", "protein_peak_AU")

METRIC_LABEL = {
    "csf_auc_cranial": "Cranial CSF exposure (AUC)",
    "transduced_per_g_cns": "Transduced vector per g CNS",
    "protein_peak_AU": "Peak transgene product",
}


def Arm(phys, site, dose=None, bio=None):
    """A dosing arm: a species physiology + route (+ optional dose / biology)."""
    return SimpleNamespace(phys=phys, site=site, dose=dose, bio=bio)


def _metric_per_unit_dose(phys, bio, site, metric_key, t_end=120.0):
    t, states = simulate(phys, bio, dose=1.0, site=site, t_end=t_end)
    return exposure_metrics(t, states, phys)[metric_key]


def project_dose(metric_key, source, target, t_end=120.0):
    """Project the target-species dose that matches the source-species `metric_key`.

    `source` must carry a dose; `target.dose` is ignored. Returns total vg.
    """
    if metric_key not in METRIC_CHOICES:
        raise ValueError(f"metric_key must be one of {METRIC_CHOICES}")
    pum_src = _metric_per_unit_dose(source.phys, source.bio, source.site, metric_key, t_end)
    m_star = source.dose * pum_src
    pum_tgt = _metric_per_unit_dose(target.phys, target.bio, target.site, metric_key, t_end)
    return m_star / pum_tgt


def dose_for_target(metric_key, arm, target_value, t_end=120.0):
    """Total vg in `arm`'s species/route to reach an absolute exposure target.

    Use this when efficacy is anchored to an absolute exposure level (e.g. a
    transduced-vector-per-gram threshold derived from a PD relationship), rather
    than to another species' exposure. Because the model is linear in dose,
    dose = target_value / (metric per unit dose).
    """
    pum = _metric_per_unit_dose(arm.phys, arm.bio, arm.site, metric_key, t_end)
    return target_value / pum


def exposure_at_dose(metric_key, arm, dose, t_end=120.0):
    """Absolute value of `metric_key` for a given dose in `arm`'s species/route."""
    return dose * _metric_per_unit_dose(arm.phys, arm.bio, arm.site, metric_key, t_end)


def normalizations(phys, total_dose):
    """Express a total vg dose on the common normalization bases."""
    return {
        "total_vg": total_dose,
        "vg_per_kg": total_dose / phys["body_weight_kg"],
        "vg_per_g_brain": total_dose / phys["brain_mass_g"],
        "vg_per_mL_csf": total_dose / phys["csf_total_mL"],
    }


def allometric_exponent(dose_a, bw_a, dose_b, bw_b):
    """Effective body-weight allometric exponent implied by two projected doses.

    b such that dose proportional to BW**b. b==1 is naive per-kg scaling;
    classic allometry uses ~0.75 for clearance. Values far from 1 show that
    body weight is a poor basis for this route.
    """
    return float(np.log(dose_b / dose_a) / np.log(bw_b / bw_a))

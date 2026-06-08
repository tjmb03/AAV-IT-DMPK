"""
Species-specific physiology for intrathecal (IT) AAV PK/biodistribution modeling.

The mechanistic premise of cross-species translation is that *physiological*
parameters (CSF volumes, CSF production/turnover, CNS tissue mass) are
species-specific and measurable, while the *biological* rate constants that
govern vector handling at the cell level (uptake, transduction, expression,
protein turnover) are more conserved. Translation is therefore achieved by
swapping the physiology while holding the biology fixed (see translation.py).

ALL VALUES BELOW ARE REPRESENTATIVE LITERATURE ORDER-OF-MAGNITUDE FIGURES,
intended as a starting scaffold. Replace them with your program's species,
strain, age and route-specific values before drawing any conclusion. See the
references section of the README.
"""

from copy import deepcopy

# CSF is discretized rostral -> caudal into three segments so that the model can
# represent the (clinically important) difference between cisternal/ICM dosing
# and lumbar IT dosing. The split fractions below are a simplification.
_CSF_SPLIT = (0.55, 0.25, 0.20)  # cranial , thoracic , lumbar


def _segments(total_mL):
    return [round(total_mL * f, 6) for f in _CSF_SPLIT]


SPECIES = {
    # ---------------------------------------------------------------- mouse
    "mouse": dict(
        label="Mouse",
        body_weight_kg=0.025,
        csf_total_mL=0.035,            # ~35 uL
        csf_seg_mL=_segments(0.035),
        Q_csf_mL_per_day=0.72,         # ~0.5 uL/min -> fast turnover (~20x/day)
        brain_mass_g=0.40,
        tissue_mass_g=[0.40, 0.04, 0.02],   # brain, thoracic cord, lumbar cord
        dna_ug_per_g=660.0,            # ~1e8 cells/g * 6.6 pg/cell
        # ganglia mass per segment: cranial entry = trigeminal/cranial-nerve
        # ganglia, thoracic & lumbar = spinal dorsal root ganglia (DRG).
        drg_mass_g=[0.008, 0.012, 0.016],
    ),
    # ----------------------------------------------------------- cynomolgus
    "cyno": dict(
        label="Cynomolgus NHP",
        body_weight_kg=4.0,
        csf_total_mL=13.0,
        csf_seg_mL=_segments(13.0),
        Q_csf_mL_per_day=43.2,         # ~30 uL/min -> turnover ~3.3x/day
        brain_mass_g=70.0,
        tissue_mass_g=[70.0, 4.0, 2.0],
        dna_ug_per_g=660.0,
        drg_mass_g=[0.20, 0.40, 0.60],
    ),
    # ---------------------------------------------------------------- human
    "human": dict(
        label="Human (adult)",
        body_weight_kg=70.0,
        csf_total_mL=150.0,
        csf_seg_mL=_segments(150.0),
        Q_csf_mL_per_day=504.0,        # ~0.35 mL/min, ~500 mL/day, turnover ~3.4x/day
        brain_mass_g=1350.0,
        tissue_mass_g=[1350.0, 20.0, 10.0],
        dna_ug_per_g=660.0,
        drg_mass_g=[2.0, 4.0, 6.0],
    ),
}


def get_species(name):
    """Return a deep copy of a species physiology dict so callers can mutate it."""
    if name not in SPECIES:
        raise KeyError(f"Unknown species '{name}'. Options: {list(SPECIES)}")
    return deepcopy(SPECIES[name])


def cns_mass_g(phys):
    return float(sum(phys["tissue_mass_g"]))


def csf_turnover_per_day(phys):
    return phys["Q_csf_mL_per_day"] / phys["csf_total_mL"]


# ---------------------------------------------------------------------------
# Serotype tropism modifiers.
#
# Capsid serotype changes *biology*, not physiology: how avidly vector is taken
# up by CNS parenchyma vs. dorsal root ganglia, and how efficiently it
# transduces. These multipliers scale the conserved biological rate constants
# (see model.apply_serotype). They are the lever that links capsid choice to
# BOTH efficacy and the DRG safety signal that has been the dose-limiting
# finding in NHP studies of IT-delivered AAV.
#
# AAV9 is the canonical broad-CNS capsid (e.g. the serotype in the approved
# SMA product) but is notably DRG-avid. A miRNA-detargeted variant (e.g.
# incorporating miR-183/miR-182 target sites that are active in DRG neurons)
# is a real strategy to suppress DRG expression while preserving CNS
# transduction; it is represented here as reduced DRG avidity.
#
# VALUES ARE ILLUSTRATIVE. Replace with capsid- and program-specific data.
SEROTYPES = {
    "generic": dict(
        label="Generic capsid (baseline)",
        f_uptake=1.0,        # CSF -> parenchyma uptake multiplier
        f_transduce=1.0,     # tissue -> internalized multiplier
        k_uptake_drg=0.015,  # CSF -> DRG uptake (1/day); modest baseline
    ),
    "AAV9": dict(
        label="AAV9 (broad CNS, DRG-avid)",
        f_uptake=1.3,
        f_transduce=1.2,
        k_uptake_drg=0.035,  # DRG-avid: the safety liability
    ),
    "AAV9_DRGdetarget": dict(
        label="AAV9 + DRG miRNA detargeting",
        f_uptake=1.3,
        f_transduce=1.2,
        k_uptake_drg=0.006,  # DRG expression suppressed ~6x; CNS preserved
    ),
}


def get_serotype(name):
    if name not in SEROTYPES:
        raise KeyError(f"Unknown serotype '{name}'. Options: {list(SEROTYPES)}")
    return deepcopy(SEROTYPES[name])

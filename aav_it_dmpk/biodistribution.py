"""
Biodistribution reporting in the units used in GLP study reports.

Internalized vector (Vcell, in vg) is converted to the readouts a reviewer
expects from a quantitative biodistribution (qPCR/ddPCR) dataset:
    vg per ug genomic DNA   and   vg per diploid genome.

Conversions use an assumed genomic DNA content per gram of tissue
(phys['dna_ug_per_g'], ~660 ug/g for ~1e8 cells/g at 6.6 pg/diploid genome).
These assumptions are explicit and adjustable; in a real program the DNA mass
is measured per sample.
"""

import numpy as np
import pandas as pd

from .model import simulate, TISSUES

PG_PER_DIPLOID_GENOME = 6.6          # pg
UG_PER_DIPLOID_GENOME = 6.6e-6       # ug


def biodistribution_table(phys, bio, dose, site, day):
    """Return a tidy biodistribution DataFrame at the given necropsy day."""
    t, states = simulate(phys, bio, dose=dose, site=site, t_end=day)
    rows = []
    dna_per_g = phys["dna_ug_per_g"]
    for r, tissue in enumerate(TISSUES):
        vcell = float(states["Vcell"][r][-1])
        mass_g = phys["tissue_mass_g"][r]
        ug_dna = mass_g * dna_per_g
        vg_per_ug = vcell / ug_dna
        rows.append({
            "tissue": tissue,
            "vector_genomes_vg": vcell,
            "vg_per_ug_DNA": vg_per_ug,
            "vg_per_diploid_genome": vg_per_ug * UG_PER_DIPLOID_GENOME,
        })
    df = pd.DataFrame(rows)
    return df

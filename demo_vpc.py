"""
Visual predictive check (VPC) walkthrough for the IT-AAV pipeline.

Run:  python demo_vpc.py
Writes fig13 (CSF PK VPC, standard + prediction-corrected) and fig14
(biodistribution VPC stratified by CNS region) to ./outputs/.

This is a *posterior-predictive* VPC (parameter uncertainty + residual error;
no fitted between-subject random effects — the mechanistic model is a
typical-subject model). See aav_it_dmpk/vpc.py for the scope note.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aav_it_dmpk import physiology as phys_mod
from aav_it_dmpk.model import default_bio
from aav_it_dmpk import data_integration as di
from aav_it_dmpk import vpc as VPC

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def panel(ax, time, bands, obs_pct, pi, ci, log=True, title=None, ylabel=None,
          legend=False):
    lo_p, mid_p, hi_p = pi
    ax.fill_between(time, bands[lo_p]["lo"], bands[lo_p]["hi"],
                    color="#e7a3a3", alpha=0.55,
                    label=f"sim {lo_p}th/{hi_p}th %ile {ci}% CI")
    ax.fill_between(time, bands[hi_p]["lo"], bands[hi_p]["hi"],
                    color="#e7a3a3", alpha=0.55)
    ax.fill_between(time, bands[mid_p]["lo"], bands[mid_p]["hi"],
                    color="#9bbce0", alpha=0.75, label=f"sim median {ci}% CI")
    ax.plot(time, obs_pct[mid_p], color="black", lw=1.8, label="obs median")
    ax.plot(time, obs_pct[lo_p], color="black", lw=1.0, ls="--",
            label=f"obs {lo_p}th/{hi_p}th %ile")
    ax.plot(time, obs_pct[hi_p], color="black", lw=1.0, ls="--")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("days")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")


def in_band(vc):
    p = vc["pi"][1]
    b = vc["bands"][p]
    inside = (vc["obs_pct"][p] >= b["lo"]) & (vc["obs_pct"][p] <= b["hi"])
    return int(inside.sum()), len(inside)


def main():
    cyno = phys_mod.get_species("cyno")
    bio = default_bio()
    dose, site = 3e13, "ICM"

    # ---- a replicate NHP study (the "observed" data) --------------------
    study = VPC.make_synthetic_study(
        cyno, bio, dose, site,
        csf_days=[0.25, 0.5, 1.0, 2.0, 3.0],      # CSF PK above LLOQ
        necropsy_days=[14.0, 42.0, 90.0],          # terminal biodistribution
        n_csf=8, n_tissue=6,
        residual_cv_csf=0.20, residual_cv_tissue=0.25,
        true_f_uptake=1.8, true_f_csf_clear=0.7, seed=11)

    # ---- fit the posterior to the study, then build the VPCs ------------
    obs_summary = VPC.study_to_observation(study)
    post = di.bayesian_update_from_observations(
        cyno, bio, dose, site, obs_summary, n_samples=3000, burn=800, thin=4, seed=0)
    samples = post["samples"]

    vc = VPC.vpc_csf(cyno, bio, dose, site, study, samples, n_sim=800)
    vc_pc = VPC.vpc_csf(cyno, bio, dose, site, study, samples, n_sim=800,
                        prediction_corrected=True)
    vt = VPC.vpc_tissue(cyno, bio, dose, site, study, samples, n_sim=800)

    print("Posterior-predictive VPC (NHP, ICM, 3e13 vg)")
    nin, ntot = in_band(vc)
    print(f"  CSF PK: observed median within simulated-median CI in {nin}/{ntot} time bins")
    for rp in vt["per_region"]:
        b = rp["bands"][50]
        inside = ((rp["obs_pct"][50] >= b["lo"]) & (rp["obs_pct"][50] <= b["hi"])).sum()
        print(f"  tissue [{rp['region']:8s}]: observed median within CI in "
              f"{int(inside)}/{len(rp['time'])} necropsy bins")

    # ---- fig13: CSF VPC, standard and prediction-corrected --------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    panel(ax[0], vc["time"], vc["bands"], vc["obs_pct"], vc["pi"], vc["ci"],
          title="CSF PK VPC", ylabel=vc["ylabel"], legend=True)
    panel(ax[1], vc_pc["time"], vc_pc["bands"], vc_pc["obs_pct"], vc_pc["pi"],
          vc_pc["ci"], log=False, title="CSF PK VPC (prediction-corrected)",
          ylabel=vc_pc["ylabel"])
    fig.suptitle("Posterior-predictive VPC — CSF vector kinetics", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig13_vpc_csf.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  -> saved fig13_vpc_csf.png")

    # ---- fig14: tissue VPC stratified by region -------------------------
    fig, ax = plt.subplots(1, N_REGIONS := len(vt["per_region"]),
                           figsize=(4.2 * N_REGIONS, 3.9), sharey=True)
    for a, rp in zip(np.atleast_1d(ax), vt["per_region"]):
        panel(a, rp["time"], rp["bands"], rp["obs_pct"], vt["pi"], vt["ci"],
              title=f"{rp['region']} cord/brain",
              ylabel=vt["ylabel"] if rp is vt["per_region"][0] else None)
        a.scatter(np.repeat(rp["time"], study["n_tissue"]),
                  study["tissue_obs"][:, vt["per_region"].index(rp), :].reshape(-1),
                  s=12, color="#555555", alpha=0.45, zorder=5)
    np.atleast_1d(ax)[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Posterior-predictive VPC — biodistribution by CNS region", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig14_vpc_tissue.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  -> saved fig14_vpc_tissue.png")
    print(f"\nVPC figures written to: {OUT}")


if __name__ == "__main__":
    main()

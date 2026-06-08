"""
Validation walkthrough for the IT-AAV translational DMPK pipeline.

Run:  python demo_validation.py
Prints a verification + validation report and writes fig9–fig12 to ./outputs/.

Layers (see README 'How to validate this model'):
  A. Verification        mass balance, exact-solution check, convergence, limiting cases
  B. Predictive          posterior predictive coverage + leave-one-cohort-out
  C. Calibration (Bayes) simulation-based calibration (rank uniformity)
  D. Sensitivity         Sobol global sensitivity on a decision output

With the package's illustrative parameters these validate the *method and
implementation*; quantitative validation needs the program's own data.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aav_it_dmpk import physiology as phys_mod
from aav_it_dmpk.model import default_bio, apply_serotype
from aav_it_dmpk import data_integration as di
from aav_it_dmpk import validation as V

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main():
    cyno = phys_mod.get_species("cyno")
    bio = default_bio()
    bio9 = apply_serotype(bio, "AAV9")
    dose, site = 3e13, "ICM"

    # ===================================================================== A
    banner("A. Verification — is the math solved correctly?")
    mb = V.mass_balance(cyno, bio9, dose, site)
    ev = V.verify_against_matrix_exponential(cyno, bio9, dose, site)
    cv = V.convergence(cyno, bio9, dose, site)
    lc = V.limiting_cases(cyno, site)
    print(f"  Mass balance (vg conserved)     : max rel imbalance {mb['max_rel_imbalance']:.1e}")
    print(f"  LSODA vs exact matrix exponential: max rel error     {ev['max_rel_error']:.1e}")
    print(f"  Solver-tolerance convergence     : rel diff           {cv['rel_diff']:.1e}")
    print(f"  Limiting cases                   : zero-uptake ok={lc['zero_uptake_ok']}, "
          f"linearity ok={lc['linearity_ok']}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(mb["t"], mb["in_system"] / mb["dose"], color="#2c6fbb", label="in system")
    ax[0].plot(mb["t"], mb["cumulative_removed"] / mb["dose"], color="#d98c00", label="removed (sinks)")
    ax[0].plot(mb["t"], mb["balance"] / mb["dose"], color="#3a7d44", lw=2, label="total (should = 1)")
    ax[0].axhline(1.0, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("days"); ax[0].set_ylabel("fraction of dose")
    ax[0].set_title("Mass balance: vector genomes conserved")
    ax[0].legend(frameon=False, fontsize=8)
    checks = ["mass\nbalance", "exact-soln\nagreement", "tolerance\nconvergence"]
    vals = [mb["max_rel_imbalance"], ev["max_rel_error"], cv["rel_diff"]]
    ax[1].bar(checks, np.maximum(vals, 1e-16), color="#2c6fbb")
    ax[1].set_yscale("log"); ax[1].axhline(1e-3, color="#9c2c2c", ls="--", lw=1, label="1e-3 tolerance")
    ax[1].set_ylabel("max relative error"); ax[1].set_title("Verification error magnitudes")
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Verification (code correctness)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_verification.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  -> saved fig9_verification.png")

    # ===================================================================== B
    banner("B. Predictive validation — does it predict held-out data?")
    obs = di.make_synthetic_observation(
        cyno, bio, dose, site, [14.0, 42.0], [0.5, 1.0, 2.0, 3.0],
        true_f_uptake=1.8, true_f_csf_clear=0.7, seed=7)
    post = di.bayesian_update_from_observations(
        cyno, bio, dose, site, obs, n_samples=3000, burn=800, thin=4, seed=0)

    ppc = V.posterior_predictive_check(cyno, bio, dose, site, obs, post["samples"], n_rep=400)
    print(f"  Posterior predictive 95% coverage: tissue {ppc['coverage_tissue']:.2f}, "
          f"CSF {ppc['coverage_csf']:.2f}, overall {ppc['coverage_overall']:.2f} (target ~0.95)")
    loco = V.leave_one_cohort_out(cyno, bio, dose, site, obs)
    for r in loco:
        print(f"  Hold out necropsy day {r['held_out_day']:.0f}: predicted within "
              f"{r['max_fold']:.2f}-fold of observed")

    tissue_obs = np.asarray(obs["tissue"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    xi = np.arange(len(tissue_obs))
    ax[0].errorbar(xi, ppc["tissue_med"],
                   yerr=[ppc["tissue_med"] - ppc["tissue_lo"], ppc["tissue_hi"] - ppc["tissue_med"]],
                   fmt="o", color="#2c6fbb", ecolor="#9bbce0", capsize=3, label="predictive 95%")
    ax[0].scatter(xi, tissue_obs, color="black", zorder=5, label="observed")
    ax[0].set_yscale("log"); ax[0].set_xlabel("tissue observation (day × region)")
    ax[0].set_ylabel("transduced vg"); ax[0].set_title("Posterior predictive check")
    ax[0].legend(frameon=False, fontsize=8)
    allp = np.concatenate([r["predicted"] for r in loco])
    allo = np.concatenate([r["observed"] for r in loco])
    lim = [min(allp.min(), allo.min()) * 0.7, max(allp.max(), allo.max()) * 1.4]
    ax[1].plot(lim, lim, color="gray", ls=":")
    ax[1].fill_between(lim, [l / 2 for l in lim], [l * 2 for l in lim], color="#3a7d44", alpha=0.12, label="2-fold")
    ax[1].scatter(allo, allp, color="#9c2c2c", zorder=5)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("observed (held-out)"); ax[1].set_ylabel("predicted")
    ax[1].set_title("Leave-one-cohort-out"); ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Predictive validation", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_predictive.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  -> saved fig10_predictive.png")

    # ===================================================================== C
    banner("C. Calibration — is the Bayesian posterior well-calibrated? (SBC)")
    print("  Running simulation-based calibration (this is the slow step)…")
    sbc = V.simulation_based_calibration(cyno, bio, dose, site,
                                         n_sims=64, n_samples=300, burn=150, thin=2, seed=1)
    print(f"  {sbc['n_sims']} replicates, {sbc['n_posterior']} posterior draws each. "
          f"Ranks should be ~uniform if calibrated.")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, key, lab in [(ax[0], "ranks_uptake", "uptake"), (ax[1], "ranks_clearance", "clearance")]:
        a.hist(sbc[key] / sbc["n_posterior"], bins=10, range=(0, 1),
               color="#2c6fbb", alpha=0.85)
        a.axhline(sbc["n_sims"] / 10, color="#9c2c2c", ls="--", lw=1, label="uniform")
        a.set_xlabel(f"normalized rank ({lab})"); a.set_ylabel("count")
        a.set_title(f"SBC rank histogram — {lab}"); a.legend(frameon=False, fontsize=8)
    fig.suptitle("Simulation-based calibration (rank uniformity)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig11_sbc.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  -> saved fig11_sbc.png")

    # ===================================================================== D
    banner("D. Global sensitivity — which parameters move the decision?")
    so = V.pipeline_sobol(cyno, base_bio=bio9, dose=dose, site=site,
                          output="transduced_per_g_cns", n_base=256, seed=0)
    order = np.argsort(so["ST"])[::-1]
    print(f"  Sobol indices for {so['output']} ({so['n_eval']} model evaluations):")
    for i in order:
        print(f"    {so['names'][i]:14s}  S1={so['S1'][i]:+.3f}  ST={so['ST'][i]:+.3f}")
    print(f"  -> {so['names'][order[0]]} dominates the transduction outcome.")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(so["names"])); w = 0.38
    ax.bar(x - w / 2, np.array(so["S1"])[order], w, label="first-order S1", color="#2c6fbb")
    ax.bar(x + w / 2, np.array(so["ST"])[order], w, label="total ST", color="#d98c00")
    ax.set_xticks(x); ax.set_xticklabels(np.array(so["names"])[order], rotation=20, fontsize=8)
    ax.set_ylabel("Sobol index"); ax.set_title("Global sensitivity of transduced vg / g CNS")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_sobol.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  -> saved fig12_sobol.png")

    banner("Done")
    print(f"Validation figures (fig9–fig12) written to: {OUT}")


if __name__ == "__main__":
    main()

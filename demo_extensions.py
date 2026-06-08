"""
Extension demonstration for the IT-AAV translational DMPK pipeline.

Run:  python demo_extensions.py
Produces console output and four figures (fig5-fig8) in ./outputs/.
Assumes the baseline demo concepts (demo.py); this script adds:

  A. Capsid serotype (AAV9) coupling CNS efficacy and DRG safety, with a
     miRNA-detargeted variant as the engineering lever.
  B. Saturable (receptor-limited) parenchymal uptake at high dose.
  C. A real transgene PD readout: a secreted, cross-correcting enzyme and the
     dose needed to hit a substrate-reduction target.
  D. Bayesian updating: a posterior over the fitted kinetic multipliers and the
     resulting credible interval on the projected human dose.

All capsid, PD and tox parameters are ILLUSTRATIVE (see module docstrings and
README). The point is the functional-modeling workflow, not the numbers.
"""

import os
import numpy as np
import pandas as pd

from aav_it_dmpk import physiology as phys_mod
from aav_it_dmpk.model import simulate, exposure_metrics, default_bio, apply_serotype
from aav_it_dmpk import translation as tr
from aav_it_dmpk import data_integration as di
from aav_it_dmpk import pd_safety as ps
from aav_it_dmpk import plotting as plot

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.4g}")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main():
    bio = default_bio()
    cyno = phys_mod.get_species("cyno")
    human = phys_mod.get_species("human")
    mouse = phys_mod.get_species("mouse")

    generic = apply_serotype(bio, "generic")
    aav9 = apply_serotype(bio, "AAV9")
    aav9_det = apply_serotype(bio, "AAV9_DRGdetarget")
    nhp_dose = 3e13

    # =====================================================================
    banner("A. Capsid serotype: AAV9 buys CNS efficacy but is DRG-avid")
    # =====================================================================
    print("CNS transduction (NHP, ICM, dose {:.0e} vg) by capsid:".format(nhp_dose))
    for lbl, b in [("generic", generic), ("AAV9", aav9), ("AAV9 + DRG-detarget", aav9_det)]:
        t, st = simulate(cyno, b, nhp_dose, "ICM")
        m = exposure_metrics(t, st, cyno)
        print(f"  {lbl:22s} transduced/g CNS = {m['transduced_per_g_cns']:.3e}   "
              f"DRG total = {m['drg_total_vg']:.2e} vg")
    print("  -> AAV9 raises CNS transduction ~1.3x over a generic capsid, but its\n"
          "     DRG avidity also raises ganglionic load. Same CNS biology in the\n"
          "     detargeted variant, with the DRG load engineered back down.")

    thr = ps.DRG_TOX_THRESHOLD_VG_PER_DIPLOID
    print(f"\nDRG safety readout (vg/diploid genome; illustrative threshold = {thr:.0f}):")
    for site in ("ICM", "lumbar"):
        tab = ps.drg_load_table(cyno, aav9, nhp_dose, site, day=28.0)
        print(f"\n  AAV9, {site}:")
        print(tab[["ganglion", "vg_per_diploid_genome", "safety_margin_x",
                   "exceeds_threshold"]].to_string(index=False))
    tab_det = ps.drg_load_table(cyno, aav9_det, nhp_dose, "ICM", day=28.0)
    print("\n  AAV9 + DRG-detarget, ICM:")
    print(tab_det[["ganglion", "vg_per_diploid_genome", "safety_margin_x",
                   "exceeds_threshold"]].to_string(index=False))
    print("\n  Reading: under ICM the cranial ganglia are the hotspot and exceed the\n"
          "  threshold; lumbar dosing relocates the hotspot to the lumbar DRG.\n"
          "  miRNA detargeting brings every ganglion under threshold at unchanged\n"
          "  CNS transduction -- a capsid-engineering lever on the dose-limiting tox.")
    f6 = plot.plot_drg_safety(cyno, aav9, nhp_dose, OUT, day=28.0,
                              bio_alt=aav9_det, alt_label="AAV9 + DRG-detarget",
                              serotype_label="AAV9")
    print(f"  -> saved {os.path.basename(f6)}")

    # =====================================================================
    banner("B. Saturable (receptor-limited) uptake at high dose")
    # =====================================================================
    doses = np.logspace(11, 15, 9)
    print("Transduced vg / g CNS per vg dosed (NHP, ICM) -- efficiency:")
    print(f"{'dose (vg)':>12} {'first-order':>14} {'saturable':>14} {'sat/linear':>12}")
    for d in doses:
        b_lin = dict(aav9); b_lin["saturable_uptake"] = False
        b_sat = dict(aav9); b_sat["saturable_uptake"] = True
        t, sl = simulate(cyno, b_lin, d, "ICM"); ml = exposure_metrics(t, sl, cyno)["transduced_per_g_cns"]
        t, ss = simulate(cyno, b_sat, d, "ICM"); ms = exposure_metrics(t, ss, cyno)["transduced_per_g_cns"]
        print(f"{d:>12.1e} {ml/d:>14.3e} {ms/d:>14.3e} {ms/ml:>12.3f}")
    print("  -> First-order uptake gives dose-proportional transduction (flat\n"
          "     efficiency). Receptor-limited uptake saturates: per-vg efficiency\n"
          "     falls at high dose, so escalation yields diminishing CNS exposure\n"
          "     (and the dose->exposure relationship is no longer linear).")
    f5 = plot.plot_saturable_dose_response(cyno, aav9, "ICM", OUT, doses=np.logspace(11, 15, 13))
    print(f"  -> saved {os.path.basename(f5)}")

    # =====================================================================
    banner("C. Transgene PD: a secreted, cross-correcting enzyme")
    # =====================================================================
    tg = ps.DEFAULT_TRANSGENE
    print(f"Transgene: {tg['label']}")
    print(f"Indication (illustrative): {tg['indication']}")
    print("\nBrain substrate reduction at day 120 (human, ICM, AAV9):")
    print(f"{'dose (vg)':>12} {'brain':>9} {'thor cord':>11} {'lumb cord':>11}")
    for d in (3e12, 1e13, 3e13, 1e14):
        pr = ps.pd_response(human, aav9, d, "ICM")
        rr = pr["terminal_reduction_per_region"]
        print(f"{d:>12.1e} {100*rr[0]:>8.1f}% {100*rr[1]:>10.1f}% {100*rr[2]:>10.1f}%")
    target = 0.70
    d_star = ps.dose_for_pd_target(human, aav9, "ICM", target)
    print(f"\nDose for >={100*target:.0f}% brain substrate reduction (human, ICM, AAV9): "
          f"{d_star:.2e} vg")
    print("  -> Expression -> secretion -> cross-correction -> substrate clearance.\n"
          "     The PD endpoint (substrate reduction), not vector genomes, is what\n"
          "     anchors the efficacious dose; the rostro-caudal gradient means cord\n"
          "     correction lags brain under a cisternal dose.")
    f7 = plot.plot_pd_response(human, aav9, "ICM", OUT, target=target)
    print(f"  -> saved {os.path.basename(f7)}")

    # =====================================================================
    banner("D. Bayesian updating: dose with a credible interval")
    # =====================================================================
    # Same efficacy anchor and synthetic NHP study as the baseline demo's
    # section 4, but now we quantify uncertainty rather than point-fit.
    M_star = tr.exposure_at_dose("transduced_per_g_cns",
                                 tr.Arm(mouse, site="ICM", bio=bio), 1e11)
    human_dose_prior = tr.dose_for_target("transduced_per_g_cns",
                                          tr.Arm(human, site="ICM", bio=bio), M_star)
    obs = di.make_synthetic_observation(
        cyno, bio, nhp_dose, "ICM",
        necropsy_days=[14.0, 42.0], csf_days=[0.5, 1.0, 2.0, 3.0],
        true_f_uptake=1.8, true_f_csf_clear=0.7,
        cv_tissue=0.20, cv_csf=0.15, seed=7)

    print("Sampling the posterior over (CSF->tissue uptake, CSF clearance) "
          "multipliers ...")
    post = di.bayesian_update_from_observations(
        cyno, bio, nhp_dose, "ICM", obs,
        n_samples=4000, burn=1000, thin=4, seed=0)
    fu_lo, fu_hi = post["f_uptake_ci"]
    fc_lo, fc_hi = post["f_csf_clear_ci"]
    print(f"  acceptance rate {post['acceptance']:.2f}, {len(post['samples'])} draws")
    print(f"  uptake multiplier   : median {post['f_uptake_median']:.2f}  "
          f"95% CrI [{fu_lo:.2f}, {fu_hi:.2f}]   (truth 1.80)")
    print(f"  clearance multiplier: median {post['f_csf_clear_median']:.2f}  "
          f"95% CrI [{fc_lo:.2f}, {fc_hi:.2f}]   (truth 0.70)")

    dp = di.propagate_dose_posterior(
        human, bio, "ICM", "transduced_per_g_cns", M_star,
        post["samples"], max_draws=400, seed=1)
    lo, hi = dp["ci"]
    print(f"\n  Prior point human dose (pre-data) : {human_dose_prior:.2e} vg")
    print(f"  Posterior human dose for M*       : {dp['median']:.2e} vg")
    print(f"  95% credible interval             : [{lo:.2e}, {hi:.2e}] vg")
    print(f"  CrI width as fraction of median   : {(hi-lo)/dp['median']:.0%}")
    print("  -> The point estimate (~the section-4 least-squares value) is now\n"
          "     wrapped in a defensible interval. With two NHP necropsy cohorts +\n"
          "     early CSF PK, the human dose is constrained to a ~+/-17% band\n"
          "     (about one-third of the median wide) -- the uncertainty statement\n"
          "     a starting dose actually needs.")
    f8 = plot.plot_bayesian(post, dp, OUT, truth=(1.8, 0.7), prior_dose=human_dose_prior)
    print(f"  -> saved {os.path.basename(f8)}")

    banner("Done")
    print(f"Figures (fig5-fig8) written to: {OUT}")


if __name__ == "__main__":
    main()

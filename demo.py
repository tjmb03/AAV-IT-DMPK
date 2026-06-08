"""
End-to-end demonstration of the IT-AAV translational DMPK pipeline.

Run:  python demo.py
Produces console tables and four figures in ./outputs/.

Story:
  1. Simulate an NHP cisternal (ICM) dose and inspect kinetics.
  2. Show why route matters (ICM vs lumbar) for brain exposure.
  3. Translate an efficacious mouse dose to NHP and human under three
     candidate efficacy drivers, and ask which dose-normalization basis is
     conserved.
  4. Integrate a new (synthetic) NHP biodistribution dataset, re-fit the model,
     and refresh the human projection.
  5. Print a study-style biodistribution table.
"""

import os
import numpy as np
import pandas as pd

from aav_it_dmpk import physiology as phys_mod
from aav_it_dmpk.model import simulate, exposure_metrics, default_bio
from aav_it_dmpk import translation as tr
from aav_it_dmpk import data_integration as di
from aav_it_dmpk import biodistribution as bd
from aav_it_dmpk import plotting as plot

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.3g}")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main():
    bio = default_bio()
    mouse = phys_mod.get_species("mouse")
    cyno = phys_mod.get_species("cyno")
    human = phys_mod.get_species("human")

    # ---- physiology snapshot ------------------------------------------------
    banner("1. Species physiology used for translation")
    phys_rows = []
    for p in (mouse, cyno, human):
        phys_rows.append({
            "species": p["label"],
            "BW_kg": p["body_weight_kg"],
            "CSF_mL": p["csf_total_mL"],
            "CSF_turnover_per_day": round(phys_mod.csf_turnover_per_day(p), 2),
            "brain_g": p["brain_mass_g"],
            "CNS_g": phys_mod.cns_mass_g(p),
        })
    print(pd.DataFrame(phys_rows).to_string(index=False))

    # ---- NHP kinetics -------------------------------------------------------
    banner("2. NHP cisternal (ICM) dose - exposure metrics")
    nhp_dose = 3e13
    t, states = simulate(cyno, bio, dose=nhp_dose, site="ICM")
    m = exposure_metrics(t, states, cyno)
    print(f"Dose: {nhp_dose:.2e} vg ICM")
    print(f"  Cranial CSF AUC      : {m['csf_auc_cranial']:.3e} vg*day/mL")
    print(f"  Transduced total     : {m['transduced_total_vg']:.3e} vg")
    print(f"  Transduced per g CNS : {m['transduced_per_g_cns']:.3e} vg/g")
    print(f"  Peak transgene (AU)  : {m['protein_peak_AU']:.3e}")
    f1 = plot.plot_kinetics(cyno, bio, nhp_dose, "ICM", OUT)
    f2 = plot.plot_route_effect(cyno, bio, nhp_dose, OUT)
    print(f"  -> saved {os.path.basename(f1)}, {os.path.basename(f2)}")

    # route effect, quantified
    _, s_icm = simulate(cyno, bio, dose=nhp_dose, site="ICM")
    _, s_lum = simulate(cyno, bio, dose=nhp_dose, site="lumbar")
    brain_icm = s_icm["Vcell"][0][-1]
    brain_lum = s_lum["Vcell"][0][-1]
    print(f"  Brain transduction ICM/lumbar ratio: {brain_icm / brain_lum:,.1f}x "
          f"(cisternal reaches brain far better than lumbar)")

    # ---- cross-species translation -----------------------------------------
    banner("3. Translate an efficacious MOUSE dose to NHP and human")
    mouse_eff_dose = 1e11          # example efficacious ICM dose in mouse
    src = tr.Arm(mouse, site="ICM", dose=mouse_eff_dose, bio=bio)
    print(f"Anchor: mouse efficacious dose = {mouse_eff_dose:.2e} vg ICM\n")

    all_tables = {}
    for metric in tr.METRIC_CHOICES:
        rows = []
        for sp, arm in [("mouse", src),
                        ("cyno", tr.Arm(cyno, site="ICM", bio=bio)),
                        ("human", tr.Arm(human, site="ICM", bio=bio))]:
            dose = mouse_eff_dose if sp == "mouse" else tr.project_dose(metric, src, arm)
            norm = tr.normalizations(arm.phys, dose)
            rows.append({"species": arm.phys["label"], **norm})
        df = pd.DataFrame(rows)
        all_tables[metric] = df

        # implied body-weight exponent over the full span vs the NHP->human bridge.
        # The mouse->human value is distorted by the mouse's very high CSF turnover;
        # the NHP->human exponent is the one to trust for first-in-human.
        tot = df["total_vg"].to_numpy()
        b_mh = tr.allometric_exponent(tot[0], mouse["body_weight_kg"],
                                      tot[2], human["body_weight_kg"])
        b_nh = tr.allometric_exponent(tot[1], cyno["body_weight_kg"],
                                      tot[2], human["body_weight_kg"])
        print(f"--- Driver: {tr.METRIC_LABEL[metric]} "
              f"(implied BW exponent  mouse->human {b_mh:.2f},  NHP->human {b_nh:.2f}) ---")
        print(df.to_string(index=False))

        # Decision-relevant question: on which normalization basis does the dose
        # barely change across the NHP->human bridge? That basis is the safer
        # scaling rule *for this driver* -- and the answer differs by driver,
        # which is the whole point: the basis is an output, not an assumption.
        nhp_row = df[df["species"] == cyno["label"]].iloc[0]
        hum_row = df[df["species"] == human["label"]].iloc[0]
        bases = ("vg_per_kg", "vg_per_g_brain", "vg_per_mL_csf")
        folds = {b: hum_row[b] / nhp_row[b] for b in bases}
        most_stable = min(folds, key=lambda b: abs(np.log(folds[b])))
        fold_str = ", ".join(f"{b} x{folds[b]:.2f}" for b in bases)
        print(f"  NHP->human fold-change by basis: {fold_str}")
        print(f"  -> most stable basis for this driver: {most_stable}\n")

    f3 = plot.plot_translation(all_tables["transduced_per_g_cns"], OUT)
    print(f"  -> saved {os.path.basename(f3)} (driver: transduced vector per g CNS)")

    # ---- real-time data integration -----------------------------------------
    banner("4. Integrate new NHP biodistribution data; refresh the human dose")

    # Efficacy anchor: an ABSOLUTE transduction target (transduced vg per g CNS)
    # required for effect. In practice this comes from the exposure-response /
    # PD analysis; here we take it as the level the efficacious mouse dose reaches
    # under the current model, then hold it fixed as kinetic parameters update.
    #
    # Why an absolute anchor here (vs the cross-species ratio match in section 3):
    # under conserved biology a single global multiplier cancels in a pure
    # species-to-species ratio, so new data could NOT move the projected dose.
    # Anchoring on a fixed efficacy target is what lets fresh data update the dose.
    mouse_arm = tr.Arm(mouse, site="ICM", bio=bio)
    M_star = tr.exposure_at_dose("transduced_per_g_cns", mouse_arm, mouse_eff_dose)
    human_arm = tr.Arm(human, site="ICM", bio=bio)
    human_dose_prior = tr.dose_for_target("transduced_per_g_cns", human_arm, M_star)
    print(f"Efficacy target  M* = {M_star:.3e} transduced vg / g CNS "
          f"(from mouse, held fixed)\n")

    # A new NHP study reads out two complementary streams: terminal transduction
    # per CNS region at necropsy cohorts (days 14, 42), and cisternal CSF vector
    # concentration at early sampling times (days 0.5-3). The true vector reaches
    # tissue more efficiently (higher CSF->tissue uptake) and clears from CSF more
    # slowly than our prior. We fit an uptake multiplier and a clearance multiplier:
    # the tissue stream informs uptake (internalized vector is uptake-limited here,
    # not transduction-limited), the CSF stream pins clearance.
    necropsy_days = [14.0, 42.0]
    csf_days = [0.5, 1.0, 2.0, 3.0]
    observed = di.make_synthetic_observation(
        cyno, bio, nhp_dose, "ICM", necropsy_days, csf_days,
        true_f_uptake=1.8, true_f_csf_clear=0.7,
        cv_tissue=0.20, cv_csf=0.15, seed=7)
    upd = di.update_from_observations(cyno, bio, nhp_dose, "ICM", observed)
    print(f"Fitted multipliers from NHP data:  CSF->tissue uptake x{upd['f_uptake']:.2f}, "
          f"CSF clearance x{upd['f_csf_clear']:.2f}   (truth 1.80 / 0.70)")

    # Refresh the human dose for the SAME efficacy target, updated biology.
    human_arm_u = tr.Arm(human, site="ICM", bio=upd["bio_updated"])
    human_dose_post = tr.dose_for_target("transduced_per_g_cns", human_arm_u, M_star)
    print(f"Human dose for M*  BEFORE update: {human_dose_prior:.3e} vg")
    print(f"Human dose for M*  AFTER  update: {human_dose_post:.3e} vg  "
          f"({human_dose_post / human_dose_prior:.2f}x)")
    print("  (vector reads out hotter + longer-lived than assumed -> less is "
          "needed to hit the efficacy target)")
    f4 = plot.plot_data_update(cyno, bio, upd["bio_updated"], nhp_dose, "ICM",
                               necropsy_days, observed["tissue"], OUT)
    print(f"  -> saved {os.path.basename(f4)}")

    # ---- biodistribution table ----------------------------------------------
    banner("5. Study-style biodistribution table (NHP, day 28)")
    table = bd.biodistribution_table(cyno, bio, nhp_dose, "ICM", 28.0)
    print(table.to_string(index=False))

    banner("Done")
    print(f"Figures written to: {OUT}")


if __name__ == "__main__":
    main()

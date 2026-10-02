"""
Interactive decision-support app for the IT-AAV translational DMPK pipeline.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy (free): push the repo to GitHub and point Streamlit Community Cloud
(https://share.streamlit.io) at this file.

Every number is illustrative — the parameters are literature-scale placeholders.
The app exposes the same functional model the package implements: vector
disposition in the CSF-CNS axis, cross-species dose translation, DRG safety,
saturable uptake, transgene PD, and Bayesian dose updating.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from aav_it_dmpk import physiology as P
from aav_it_dmpk.model import (default_bio, apply_serotype, simulate,
                               exposure_metrics, REGIONS)
from aav_it_dmpk import translation as tr
from aav_it_dmpk import data_integration as di
from aav_it_dmpk import pd_safety as ps
from aav_it_dmpk import fih as fh

st.set_page_config(page_title="IT-AAV translational DMPK",
                   page_icon="🧬", layout="wide")

COL = {"cranial": "#2c6fbb", "thoracic": "#d98c00", "lumbar": "#9c2c2c"}
SPECIES_LABEL = {"mouse": "Mouse", "cyno": "Cynomolgus NHP", "human": "Human"}
ROUTE_LABEL = {"ICM": "ICM / cisternal", "thoracic": "Thoracic IT", "lumbar": "Lumbar IT"}
SEROTYPE_LABEL = {"generic": "Generic capsid", "AAV9": "AAV9 (DRG-avid)",
                  "AAV9_DRGdetarget": "AAV9 + DRG detargeting"}


# --------------------------------------------------------------------------
# Cached compute (keyed on hashable primitives, so bio dicts are rebuilt inside)
# --------------------------------------------------------------------------
def build_bio(serotype, saturable_uptake=False, km_uptake=1e11):
    bio = apply_serotype(default_bio(), serotype)
    bio = dict(bio)
    bio["saturable_uptake"] = saturable_uptake
    bio["Km_uptake_conc"] = km_uptake
    return bio


@st.cache_data(show_spinner=False)
def run_sim(species, serotype, dose, site, t_end=120.0,
            saturable_uptake=False, km_uptake=1e11):
    phys = P.get_species(species)
    bio = build_bio(serotype, saturable_uptake, km_uptake)
    t, states = simulate(phys, bio, dose=dose, site=site, t_end=t_end)
    metrics = exposure_metrics(t, states, phys)
    return t, states, metrics


def _floor_log(y, rel=1e-8):
    y = np.asarray(y, dtype=float)
    peak = np.nanmax(y)
    out = y.copy()
    out[out < peak * rel] = np.nan
    return out


# --------------------------------------------------------------------------
# Sidebar — global controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 IT-AAV DMPK")
    st.caption("Mechanistic translational model — mouse → NHP → human")
    species = st.selectbox("Species", ["mouse", "cyno", "human"], index=1,
                           format_func=lambda s: SPECIES_LABEL[s])
    site = st.selectbox("Route", ["ICM", "thoracic", "lumbar"], index=0,
                        format_func=lambda s: ROUTE_LABEL[s])
    serotype = st.selectbox("Capsid serotype",
                            ["generic", "AAV9", "AAV9_DRGdetarget"], index=1,
                            format_func=lambda s: SEROTYPE_LABEL[s])
    log_dose = st.slider("log₁₀ total dose (vg)", 11.0, 15.0, 13.5, 0.1)
    dose = 10.0 ** log_dose
    st.metric("Total dose", f"{dose:.2e} vg")
    st.divider()
    st.caption("⚠️ Illustrative scaffold. All parameters are literature-scale "
               "placeholders; nothing here is a basis for a real decision.")

st.title("Intrathecal AAV — translational DMPK explorer")

tabs = st.tabs(["Disposition & route", "Cross-species translation",
                "DRG safety", "Dose–response (saturable)",
                "Transgene PD", "Bayesian update", "FIH dose bracket", "About"])

# ==========================================================================
# Tab 1 — Disposition & route
# ==========================================================================
with tabs[0]:
    st.subheader(f"{SPECIES_LABEL[species]} · {ROUTE_LABEL[site]} · "
                 f"{SEROTYPE_LABEL[serotype]}")
    t, states, m = run_sim(species, serotype, dose, site)
    phys = P.get_species(species)

    c = st.columns(4)
    c[0].metric("Cranial CSF AUC", f"{m['csf_auc_cranial']:.2e}", "vg·day/mL")
    c[1].metric("Transduced / g CNS", f"{m['transduced_per_g_cns']:.2e}", "vg/g")
    c[2].metric("Peak transgene", f"{m['protein_peak_AU']:.2e}", "AU")
    c[3].metric("DRG total", f"{m['drg_total_vg']:.2e}", "vg")

    left, right = st.columns(2)
    with left:
        Vol = phys["csf_seg_mL"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for r, reg in enumerate(REGIONS):
            ax[0].plot(t, _floor_log(states["Vcsf"][r] / Vol[r]), color=COL[reg], label=reg)
        ax[0].set_yscale("log"); ax[0].set_xlim(0, 10)
        ax[0].set_xlabel("days"); ax[0].set_ylabel("CSF conc (vg/mL)")
        ax[0].set_title("CSF kinetics"); ax[0].legend(frameon=False, fontsize=8)
        for r, reg in enumerate(REGIONS):
            ax[1].plot(t, states["Vcell"][r], color=COL[reg], label=reg)
        ax[1].set_xlabel("days"); ax[1].set_ylabel("transduced (vg)")
        ax[1].set_title("Transduction by region")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with right:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for s, cc in [("ICM", "#2c6fbb"), ("lumbar", "#9c2c2c")]:
            _, st_s, _ = run_sim(species, serotype, dose, s)
            ax[0].plot(_floor_log(st_s["Vcsf"][0] / phys["csf_seg_mL"][0]),
                       color=cc, label=ROUTE_LABEL.get(s, s))
            ax[1].plot(st_s["Vcell"][0], color=cc, label=ROUTE_LABEL.get(s, s))
        ax[0].set_yscale("log"); ax[0].set_xlim(0, 10)
        ax[0].set_xlabel("time index"); ax[0].set_ylabel("cranial CSF (vg/mL)")
        ax[0].set_title("Cranial CSF: ICM vs lumbar"); ax[0].legend(frameon=False, fontsize=8)
        ax[1].set_xlabel("time index"); ax[1].set_ylabel("brain transduced (vg)")
        ax[1].set_title("Brain transduction: ICM vs lumbar")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Cisternal dosing reaches the brain far better than lumbar; "
               "a lumbar dose must disperse rostrally against caudal drift.")

# ==========================================================================
# Tab 2 — Cross-species translation
# ==========================================================================
with tabs[1]:
    st.subheader("Project an efficacious dose across species")
    cc = st.columns([1, 1, 1])
    driver = cc[0].selectbox("Efficacy driver (anchor metric)",
                             list(tr.METRIC_CHOICES),
                             format_func=lambda k: tr.METRIC_LABEL[k])
    src_species = cc[1].selectbox("Source species", ["mouse", "cyno", "human"],
                                  index=0, format_func=lambda s: SPECIES_LABEL[s])
    src_logdose = cc[2].slider("Source efficacious dose (log₁₀ vg)", 9.0, 15.0, 11.0, 0.1)
    src_dose = 10.0 ** src_logdose
    tsite = "ICM"  # translation uses a fixed comparison route; linear model

    bio_lin = build_bio(serotype, saturable_uptake=False)
    src = tr.Arm(P.get_species(src_species), site=tsite, dose=src_dose, bio=bio_lin)
    rows = []
    for sp in ["mouse", "cyno", "human"]:
        arm = tr.Arm(P.get_species(sp), site=tsite, bio=bio_lin)
        d = src_dose if sp == src_species else tr.project_dose(driver, src, arm)
        rows.append({"species": SPECIES_LABEL[sp], **tr.normalizations(arm.phys, d)})
    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.format({"total_vg": "{:.2e}", "vg_per_kg": "{:.2e}",
                         "vg_per_g_brain": "{:.2e}", "vg_per_mL_csf": "{:.2e}"}),
        use_container_width=True)

    nh = df[df["species"] == "Human"].iloc[0]
    nn = df[df["species"] == "Cynomolgus NHP"].iloc[0]
    bases = ("total_vg", "vg_per_kg", "vg_per_g_brain", "vg_per_mL_csf")
    folds = {b: nh[b] / nn[b] for b in bases}
    stable = min(folds, key=lambda b: abs(np.log(folds[b])))
    st.info(f"Most-conserved basis across the **NHP → human** bridge for this "
            f"driver: **{stable.replace('_', ' ')}** "
            f"(fold-change {folds[stable]:.2f}). The conserved basis depends on "
            f"the driver — it is an output, not an assumption.")
    if driver in tr.EXTENSIVE_METRICS:
        st.warning(
            "**This driver is an extensive (whole-CNS total) metric, not a "
            "concentration.** Matching it across species conserves the *total* "
            "dose, so the per-gram level falls roughly in proportion to CNS "
            "mass (~19× from NHP to human) — and the implied body-weight "
            "exponent collapses toward 0. Read this panel as a diagnostic of "
            "the metric rather than a dosing rule; the per-g and CSF-AUC "
            "drivers give interpretable scaling.")

    x = np.arange(3)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(x, df["total_vg"], color="#2c6fbb")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(df["species"], fontsize=8)
    ax[0].set_ylabel("total dose (vg)"); ax[0].set_title("Projected total dose")
    w = 0.21
    for i, (col, lab, c2) in enumerate([("total_vg", "total vg", "#2c6fbb"),
                                        ("vg_per_kg", "vg/kg", "#9c2c2c"),
                                        ("vg_per_g_brain", "vg/g brain", "#d98c00"),
                                        ("vg_per_mL_csf", "vg/mL CSF", "#3a7d44")]):
        v = np.array(df[col], float); v = v / v[0]
        ax[1].bar(x + (i - 1.5) * w, v, w, label=lab, color=c2)
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(df["species"], fontsize=8)
    ax[1].axhline(1.0, color="gray", ls=":", lw=1)
    ax[1].set_ylabel("relative to source"); ax[1].set_title("Which basis is conserved?")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ==========================================================================
# Tab 3 — DRG safety
# ==========================================================================
with tabs[2]:
    st.subheader("Dorsal root ganglion load — the dose-limiting safety signal")
    thr = st.slider("Tolerability threshold (vg / diploid genome)",
                    200, 5000, int(ps.DRG_TOX_THRESHOLD_VG_PER_DIPLOID), 100)
    phys = P.get_species(species)
    bio_sero = build_bio(serotype)
    bio_det = build_bio("AAV9_DRGdetarget")

    cols = st.columns(2)
    with cols[0]:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        g = ps.GANGLIA; xg = np.arange(len(g)); w = 0.38
        for i, (s, c2) in enumerate([("ICM", "#2c6fbb"), ("lumbar", "#9c2c2c")]):
            tab = ps.drg_load_table(phys, bio_sero, dose, s, 28.0, thr)
            ax.bar(xg + (i - 0.5) * w, tab["vg_per_diploid_genome"], w,
                   label=ROUTE_LABEL.get(s, s), color=c2)
        ax.axhline(thr, color="black", ls="--", lw=1, label="threshold")
        ax.set_yscale("log"); ax.set_xticks(xg); ax.set_xticklabels(g, rotation=15, fontsize=8)
        ax.set_ylabel("vg / diploid genome"); ax.set_title(f"Route effect ({SEROTYPE_LABEL[serotype]})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with cols[1]:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        g = ps.GANGLIA; xg = np.arange(len(g)); w = 0.38
        for i, (b, lab, c2) in enumerate([(bio_sero, SEROTYPE_LABEL[serotype], "#2c6fbb"),
                                          (bio_det, "AAV9 detargeted", "#3a7d44")]):
            tab = ps.drg_load_table(phys, b, dose, "ICM", 28.0, thr)
            ax.bar(xg + (i - 0.5) * w, tab["vg_per_diploid_genome"], w, label=lab, color=c2)
        ax.axhline(thr, color="black", ls="--", lw=1, label="threshold")
        ax.set_yscale("log"); ax.set_xticks(xg); ax.set_xticklabels(g, rotation=15, fontsize=8)
        ax.set_ylabel("vg / diploid genome"); ax.set_title("Capsid-engineering lever (ICM)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    tab_show = ps.drg_load_table(phys, bio_sero, dose, site, 28.0, thr)
    st.dataframe(tab_show.style.format({"vector_genomes_vg": "{:.2e}",
                 "vg_per_diploid_genome": "{:.0f}", "safety_margin_x": "{:.2f}"}),
                 use_container_width=True)

# ==========================================================================
# Tab 4 — Dose–response (saturable uptake)
# ==========================================================================
with tabs[3]:
    st.subheader("Receptor-limited uptake bends the high-dose response")
    km_log = st.slider("Half-saturation CSF conc Km (log₁₀ vg/mL)", 9.0, 13.0, 11.0, 0.25)
    km = 10.0 ** km_log
    phys = P.get_species(species)
    doses = np.logspace(11, 15, 13)

    def eff_curve(sat):
        out = []
        for d in doses:
            _, _, mm = run_sim(species, serotype, float(d), site,
                               saturable_uptake=sat, km_uptake=km)
            out.append(mm["transduced_per_g_cns"])
        return np.array(out)

    lin = eff_curve(False); sat = eff_curve(True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].loglog(doses, lin, "o-", color="#2c6fbb", label="first-order")
    ax[0].loglog(doses, sat, "o-", color="#9c2c2c", label="saturable")
    ax[0].set_xlabel("dose (vg)"); ax[0].set_ylabel("transduced vg / g CNS")
    ax[0].set_title("Dose–response"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].loglog(doses, lin / doses, "o-", color="#2c6fbb", label="first-order")
    ax[1].loglog(doses, sat / doses, "o-", color="#9c2c2c", label="saturable")
    ax[1].set_xlabel("dose (vg)"); ax[1].set_ylabel("per-vg efficiency")
    ax[1].set_title("Efficiency falls once uptake saturates")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Lower Km ⇒ saturation begins at lower dose. Under saturable "
               "uptake the dose→exposure map is no longer linear.")

# ==========================================================================
# Tab 5 — Transgene PD
# ==========================================================================
with tabs[4]:
    st.subheader("Transgene PD — secreted, cross-correcting enzyme")
    tg = ps.DEFAULT_TRANSGENE
    st.caption(f"{tg['label']} — {tg['indication']}")
    target = st.slider("PD target: brain substrate reduction", 0.30, 0.95, 0.70, 0.05)
    phys = P.get_species(species)
    bio_sero = build_bio(serotype)

    pd_doses = [3e12, 1e13, 3e13, 1e14]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(pd_doses)))
    for d, c2 in zip(pd_doses, cmap):
        pr = ps.pd_response(phys, bio_sero, d, site)
        ax[0].plot(pr["t"], 100 * pr["reduction"][0], color=c2, label=f"{d:.0e}")
    ax[0].set_xlabel("days"); ax[0].set_ylabel("brain substrate reduction (%)")
    ax[0].set_ylim(0, 100); ax[0].set_title("PD over time (brain)")
    ax[0].legend(frameon=False, fontsize=8, title="dose (vg)")
    grid = np.logspace(11.7, 14.7, 14)
    term = [100 * ps.pd_response(phys, bio_sero, float(d), site)["brain_terminal_reduction"]
            for d in grid]
    ax[1].semilogx(grid, term, "o-", color="#3a7d44")
    ax[1].axhline(100 * target, color="black", ls="--", lw=1)
    try:
        d_star = ps.dose_for_pd_target(phys, bio_sero, site, target)
        ax[1].axvline(d_star, color="#9c2c2c", ls=":", lw=1.5)
        msg = f"Dose for ≥{target*100:.0f}% brain reduction ({ROUTE_LABEL[site]}): **{d_star:.2e} vg**"
    except ValueError:
        msg = "Target not reachable in the dose bracket for this configuration."
    ax[1].set_xlabel("dose (vg)"); ax[1].set_ylabel("terminal brain reduction (%)")
    ax[1].set_ylim(0, 100); ax[1].set_title("Dose for a PD target")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.success(msg)
    cur = ps.pd_response(phys, bio_sero, dose, site)
    st.metric("Brain substrate reduction at current sidebar dose",
              f"{100*cur['brain_terminal_reduction']:.0f}%")

# ==========================================================================
# Tab 6 — Bayesian update
# ==========================================================================
with tabs[5]:
    st.subheader("Bayesian update → human dose with a credible interval")
    st.write("A synthetic NHP study (two necropsy cohorts + early CSF PK) updates "
             "the two identifiable kinetic multipliers; the posterior is propagated "
             "to a credible interval on the human dose for a fixed efficacy target.")
    cc = st.columns(3)
    true_fu = cc[0].slider("True uptake multiplier", 0.5, 3.0, 1.8, 0.1)
    true_fc = cc[1].slider("True CSF-clearance multiplier", 0.3, 2.0, 0.7, 0.1)
    nhp_logdose = cc[2].slider("NHP study dose (log₁₀ vg)", 12.0, 14.0, 13.477, 0.1)
    nhp_dose = 10.0 ** nhp_logdose

    @st.cache_data(show_spinner=False)
    def run_bayes(true_fu, true_fc, nhp_dose, n_samples=3000):
        bio = default_bio()
        cyno = P.get_species("cyno"); human = P.get_species("human")
        mouse = P.get_species("mouse")
        M_star = tr.exposure_at_dose("transduced_per_g_cns",
                                     tr.Arm(mouse, site="ICM", bio=bio), 1e11)
        prior = tr.dose_for_target("transduced_per_g_cns",
                                   tr.Arm(human, site="ICM", bio=bio), M_star)
        obs = di.make_synthetic_observation(
            cyno, bio, nhp_dose, "ICM", [14.0, 42.0], [0.5, 1.0, 2.0, 3.0],
            true_f_uptake=true_fu, true_f_csf_clear=true_fc, seed=7)
        post = di.bayesian_update_from_observations(
            cyno, bio, nhp_dose, "ICM", obs,
            n_samples=n_samples, burn=800, thin=4, seed=0)
        dp = di.propagate_dose_posterior(
            human, bio, "ICM", "transduced_per_g_cns", M_star,
            post["samples"], max_draws=200, seed=1)
        return post, dp, prior

    if st.button("▶ Run Bayesian update", type="primary"):
        with st.spinner("Sampling the posterior…"):
            st.session_state["bayes"] = run_bayes(true_fu, true_fc, nhp_dose)

    if "bayes" in st.session_state:
        post, dp, prior = st.session_state["bayes"]
        c = st.columns(3)
        c[0].metric("Prior point dose", f"{prior:.2e} vg")
        c[1].metric("Posterior median dose", f"{dp['median']:.2e} vg")
        c[2].metric("95% credible interval",
                    f"{dp['ci'][0]:.2e} – {dp['ci'][1]:.2e}")
        u_lo, u_hi = post["f_uptake_ci"]; c_lo, c_hi = post["f_csf_clear_ci"]
        st.caption(f"Uptake multiplier {post['f_uptake_median']:.2f} "
                   f"[{u_lo:.2f}, {u_hi:.2f}] (truth {true_fu:.2f}) · "
                   f"Clearance {post['f_csf_clear_median']:.2f} "
                   f"[{c_lo:.2f}, {c_hi:.2f}] (truth {true_fc:.2f}) · "
                   f"acceptance {post['acceptance']:.2f}")

        s = np.asarray(post["samples"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].scatter(s[:, 0], s[:, 1], s=8, alpha=0.25, color="#2c6fbb", edgecolors="none")
        ax[0].scatter([post["f_uptake_median"]], [post["f_csf_clear_median"]],
                      s=80, color="#3a7d44", zorder=5, label="median")
        ax[0].scatter([true_fu], [true_fc], marker="*", s=200, color="#9c2c2c",
                      zorder=6, label="truth")
        ax[0].set_xlabel("uptake multiplier"); ax[0].set_ylabel("clearance multiplier")
        ax[0].set_title("Joint posterior"); ax[0].legend(frameon=False, fontsize=8)
        d = np.asarray(dp["doses"])
        ax[1].hist(d, bins=30, color="#2c6fbb", alpha=0.8)
        ax[1].axvspan(dp["ci"][0], dp["ci"][1], color="#3a7d44", alpha=0.18, label="95% CrI")
        ax[1].axvline(dp["median"], color="#3a7d44", lw=2, label="median")
        ax[1].axvline(prior, color="#9c2c2c", ls=":", lw=1.5, label="prior")
        ax[1].set_xlabel("projected human dose (vg)"); ax[1].set_ylabel("draws")
        ax[1].set_title("Human dose: posterior + 95% CrI")
        ax[1].legend(frameon=False, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.info("Set the synthetic-study parameters, then click **Run Bayesian update**.")

# ==========================================================================
# Tab 7 — FIH dose bracket
# ==========================================================================
with tabs[6]:
    st.subheader("First-in-human dose bracket (NOAEL-anchored)")
    st.markdown(
        "The translation tab projects an **efficacious** dose. A starting dose is "
        "a different question, set by safety. This tab runs the inversion: the tox "
        "study's NOAEL *dose* → the DRG exposure it produced (**that** is your "
        "threshold, derived rather than chosen) → the human dose carrying the same "
        "exposure → a safety factor → check it still clears the efficacy floor.")

    fc = st.columns([1, 1, 1])
    tox_species = fc[0].selectbox("Tox species", ["cyno", "mouse"], index=0,
                                  format_func=lambda s: SPECIES_LABEL[s])
    noael_log = fc[1].slider("NOAEL from tox study (log₁₀ vg)", 10.0, 15.0, 13.0, 0.1)
    noael = 10.0 ** noael_log
    necropsy_day = fc[2].slider("Necropsy day for DRG readout", 7, 90, 28, 7)

    fc2 = st.columns([1, 1, 1])
    sf_sel = fc2[0].multiselect("Safety factors", [2, 3, 6, 10, 20], default=[3, 6, 10])
    floor_pct = fc2[1].slider("Efficacy floor (% brain substrate reduction)", 10, 60, 30, 5)
    target_pct = fc2[2].slider("Efficacy target (% reduction)", 50, 95, 70, 5)

    bio_fih = build_bio(serotype, saturable_uptake=False)
    if not sf_sel:
        st.info("Select at least one safety factor.")
    else:
        with st.spinner("Inverting NOAEL → threshold → human dose…"):
            br = fh.fih_bracket(
                P.get_species(tox_species), P.get_species("human"), bio_fih,
                noael_dose=noael, site="ICM", day=float(necropsy_day),
                safety_factors=tuple(float(s) for s in sorted(sf_sel)),
                pd_floor_reduction=floor_pct / 100.0,
                pd_target_reduction=target_pct / 100.0)

        m = st.columns(4)
        m[0].metric("Derived DRG threshold",
                    f"{br['derived_threshold_vg_per_diploid']:.0f}", "vg/diploid")
        m[1].metric("Human dose, matched exposure",
                    f"{br['human_dose_matched_exposure_vg']:.2e}", "vg")
        m[2].metric(f"Efficacy floor (≥{floor_pct}%)", f"{br['pd_floor_vg']:.2e}", "vg")
        m[3].metric(f"Efficacy target (≥{target_pct}%)", f"{br['pd_target_vg']:.2e}", "vg")

        st.caption(
            f"Naive body-weight scaling of the NOAEL would give "
            f"{br['bodyweight_scaled_dose_vg']:.2e} vg — "
            f"**{br['bodyweight_vs_exposure_fold']:.2f}×** the exposure-matched dose. "
            "That gap is why the mechanistic model exists.")

        sdf = pd.DataFrame(br["starting_doses"]).rename(columns={
            "safety_factor": "safety factor", "starting_dose_vg": "starting dose (vg)",
            "clears_pd_floor": "clears efficacy floor",
            "fold_above_floor": "× above floor", "fold_of_pd_target": "× of target"})
        st.dataframe(sdf.style.format({"starting dose (vg)": "{:.2e}",
                                       "× above floor": "{:.2f}",
                                       "× of target": "{:.2f}"}),
                     use_container_width=True)

        if br["window_exists"]:
            st.success("**A window exists.** At least one safety-factored starting "
                       "dose sits above the minimally active dose.")
        else:
            st.error("**No window at these settings.** Every safety-factored start "
                     "falls below the efficacy floor — the lever is capsid "
                     "detargeting or route, not dose selection.")

        fig, ax = plt.subplots(figsize=(9.5, 2.9))
        hed = br["human_dose_matched_exposure_vg"]
        ax.axvspan(br["pd_floor_vg"], hed, color="#3a7d44", alpha=0.10,
                   label="therapeutic window")
        ax.axvline(hed, color="#9c2c2c", lw=2, label="DRG-limited (matched NOAEL exposure)")
        ax.axvline(br["pd_floor_vg"], color="#3a7d44", lw=2, ls="--",
                   label=f"efficacy floor (≥{floor_pct}%)")
        ax.axvline(br["pd_target_vg"], color="#d98c00", lw=2, ls=":",
                   label=f"efficacy target (≥{target_pct}%)")
        for s in br["starting_doses"]:
            ax.plot(s["starting_dose_vg"], 0.5, "o", ms=11, color="#2c6fbb", zorder=5)
            ax.annotate(f"SF {s['safety_factor']:.0f}×",
                        (s["starting_dose_vg"], 0.5), textcoords="offset points",
                        xytext=(0, 13), ha="center", fontsize=8)
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xlabel("total dose (vg)")
        ax.set_title("FIH dose bracket")
        ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.caption(
            "What the model supplies: the NOAEL→exposure conversion and the "
            "cross-species exposure match. What it does **not** supply: the NOAEL "
            "itself, the histopathology behind it, or the choice of safety factor. "
            "Parameters here are illustrative placeholders.")

        # ---------------- uncertainty propagation ----------------
        st.markdown("---")
        st.markdown("#### Does the window survive uncertainty?")
        st.markdown(
            "Three sources are propagated separately, because they do not act alike. "
            "The ceiling is a **ratio** anchor (tox exposure matched in human), so a "
            "global biology multiplier largely cancels; the floor is an **absolute** "
            "anchor and carries the biology uncertainty in full. Propagating the "
            "posterior alone would give a falsely tight ceiling.")

        uc = st.columns([1, 1, 1, 1])
        loael_log = uc[0].slider("LOAEL (log₁₀ vg)", noael_log + 0.1, noael_log + 1.5,
                                 min(noael_log + 0.5, 15.0), 0.1,
                                 help="Next dose group up. The true threshold lies "
                                      "between NOAEL and LOAEL — the censoring window.")
        phys_cv = uc[1].slider("Human physiology CV", 0.0, 0.40, 0.15, 0.05,
                               help="CSF volume and DRG mass uncertainty.")
        n_draws = uc[2].slider("Draws", 40, 200, 80, 20)
        use_post = uc[3].checkbox("Use Bayesian posterior", value=True,
                                  help="Requires a run on the Bayesian update tab.")

        post_samples = None
        if use_post:
            if "bayes" in st.session_state:
                post_samples = st.session_state["bayes"][0]["samples"]
            else:
                st.caption("⚠️ No posterior yet — run the **Bayesian update** tab "
                           "first, or untick to use the prior.")

        if st.button("▶ Propagate uncertainty", type="primary"):
            with st.spinner(f"Propagating {n_draws} draws…"):
                st.session_state["fih_unc"] = fh.propagate_fih_bracket(
                    P.get_species(tox_species), P.get_species("human"), bio_fih,
                    noael_dose=noael, posterior_samples=post_samples,
                    site="ICM", day=float(necropsy_day),
                    safety_factors=tuple(float(s) for s in sorted(sf_sel)),
                    pd_floor_reduction=floor_pct / 100.0,
                    loael_dose=10.0 ** loael_log, physiol_cv=phys_cv,
                    n_draws=int(n_draws), seed=1)

        if "fih_unc" in st.session_state:
            r = st.session_state["fih_unc"]
            cc, ff = r["ceiling"], r["floor"]
            u = st.columns(2)
            u[0].metric("DRG ceiling (median)", f"{cc['median']:.2e}",
                        f"95% CI {cc['p2.5']:.1e} – {cc['p97.5']:.1e}  "
                        f"({r['ceiling_spread_fold']:.1f}× spread)")
            u[1].metric("Efficacy floor (median)", f"{ff['median']:.2e}",
                        f"95% CI {ff['p2.5']:.1e} – {ff['p97.5']:.1e}  "
                        f"({r['floor_spread_fold']:.1f}× spread)")

            att = r["ceiling_attribution_fold"]
            st.markdown("**What actually drives the ceiling spread** "
                        "(97.5/2.5 fold, one source at a time):")
            adf = pd.DataFrame([
                {"source": "biology (posterior)", "spread (fold)": att["biology_only"]},
                {"source": "NOAEL censoring (dose spacing)", "spread (fold)": att["noael_only"]},
                {"source": "human physiology", "spread (fold)": att["physiology_only"]}])
            st.dataframe(adf.style.format({"spread (fold)": "{:.2f}×"}),
                         use_container_width=True, hide_index=True)
            if att["biology_only"] < 1.1:
                st.info("The biology posterior contributes essentially **nothing** to "
                        "ceiling uncertainty — it cancels in the ratio anchor. Tightening "
                        "the posterior will not tighten the ceiling; narrower dose "
                        "spacing and better human physiology estimates will.")

            wdf = pd.DataFrame([{
                "safety factor": s["safety_factor"],
                "starting dose (median)": s["start_median"],
                "95% low": s["start_p2.5"], "95% high": s["start_p97.5"],
                "P(window survives)": s["p_window_survives"],
                "survives conservative tail": s["survives_conservative_tail"],
            } for s in r["starting_doses"]])
            st.dataframe(wdf.style.format({
                "starting dose (median)": "{:.2e}", "95% low": "{:.2e}",
                "95% high": "{:.2e}", "P(window survives)": "{:.2f}"}),
                use_container_width=True, hide_index=True)

            worst = min(s["p_window_survives"] for s in r["starting_doses"])
            if all(s["survives_conservative_tail"] for s in r["starting_doses"]):
                st.success("**Window survives the conservative tail** for every safety "
                           "factor — low-end ceiling still clears the high-end floor.")
            elif worst > 0.8:
                st.warning("Window holds at the medians but **not** under the "
                           "conservative tail for every safety factor.")
            else:
                st.error("**Window does not reliably survive uncertainty.**")

            fig, ax = plt.subplots(figsize=(9.5, 3.1))
            ax.axvspan(ff["p2.5"], ff["p97.5"], color="#3a7d44", alpha=0.18,
                       label="efficacy floor 95% CI")
            ax.axvspan(cc["p2.5"], cc["p97.5"], color="#9c2c2c", alpha=0.18,
                       label="DRG ceiling 95% CI")
            ax.axvline(ff["median"], color="#3a7d44", lw=2, ls="--")
            ax.axvline(cc["median"], color="#9c2c2c", lw=2)
            for s in r["starting_doses"]:
                ax.plot(s["start_median"], 0.5, "o", ms=10, color="#2c6fbb", zorder=6)
                ax.hlines(0.5, s["start_p2.5"], s["start_p97.5"],
                          color="#2c6fbb", lw=2, alpha=0.6, zorder=5)
                ax.annotate(f"SF {s['safety_factor']:.0f}×",
                            (s["start_median"], 0.5), textcoords="offset points",
                            xytext=(0, 13), ha="center", fontsize=8)
            ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_yticks([])
            ax.set_xlabel("total dose (vg)")
            ax.set_title("FIH bracket with uncertainty")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ==========================================================================
# Tab 8 — About
# ==========================================================================
with tabs[7]:
    st.markdown(
        """
### About

This app drives **`aav_it_dmpk`**, a mechanistic (functional, not tool-based)
model of intrathecally-delivered AAV gene therapy, built to make a translational
argument **legible**: vector disposition in a segmented CSF–CNS axis, cross-species
dose scaling by *physiology-swap / biology-fixed*, DRG safety, saturable uptake,
a cross-correcting-enzyme PD readout, and Bayesian dose updating.

**This is an illustrative scaffold.** Every parameter is a literature-scale
placeholder; none of the doses, fold-changes, or projections is a basis for a
real decision until replaced with program-specific data. The structure is the
contribution; the numbers are placeholders.

The same logic is in the package's tests (each asserts a *scientific* property)
and is reproduced by CI. See the repository README for the full write-up,
regulatory framing, and references.
        """
    )
    st.caption("Built for demonstration of functional translational DMPK modeling.")

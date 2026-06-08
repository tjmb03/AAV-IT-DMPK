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
                "Transgene PD", "Bayesian update", "About"])

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
    bases = ("vg_per_kg", "vg_per_g_brain", "vg_per_mL_csf")
    folds = {b: nh[b] / nn[b] for b in bases}
    stable = min(folds, key=lambda b: abs(np.log(folds[b])))
    st.info(f"Most-conserved basis across the **NHP → human** bridge for this "
            f"driver: **{stable.replace('_', ' ')}** "
            f"(fold-change {folds[stable]:.2f}). The conserved basis depends on "
            f"the driver — it is an output, not an assumption.")

    x = np.arange(3)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(x, df["total_vg"], color="#2c6fbb")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(df["species"], fontsize=8)
    ax[0].set_ylabel("total dose (vg)"); ax[0].set_title("Projected total dose")
    w = 0.27
    for i, (col, lab, c2) in enumerate([("vg_per_kg", "vg/kg", "#9c2c2c"),
                                        ("vg_per_g_brain", "vg/g brain", "#d98c00"),
                                        ("vg_per_mL_csf", "vg/mL CSF", "#3a7d44")]):
        v = np.array(df[col], float); v = v / v[0]
        ax[1].bar(x + (i - 1) * w, v, w, label=lab, color=c2)
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
# Tab 7 — About
# ==========================================================================
with tabs[6]:
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

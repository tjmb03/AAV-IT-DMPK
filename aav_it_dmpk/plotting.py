"""Plotting utilities. Each function saves a PNG and returns its path."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import simulate, REGIONS

_COL = {"cranial": "#2c6fbb", "thoracic": "#d98c00", "lumbar": "#9c2c2c"}


def _floor_for_log(conc, rel=1e-8):
    """Mask concentrations below peak*rel (sub-resolution numerical tail) with NaN
    so log-scale plots show the real decay without integrator noise hash."""
    conc = np.asarray(conc, dtype=float)
    peak = np.nanmax(conc)
    out = conc.copy()
    out[out < peak * rel] = np.nan
    return out


def plot_kinetics(phys, bio, dose, site, outdir, t_end=120.0):
    t, states = simulate(phys, bio, dose=dose, site=site, t_end=t_end)
    Vol = phys["csf_seg_mL"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    for r, reg in enumerate(REGIONS):
        ax[0].plot(t, _floor_for_log(states["Vcsf"][r] / Vol[r]), color=_COL[reg], label=reg)
    ax[0].set_yscale("log"); ax[0].set_xlim(0, 10)
    ax[0].set_xlabel("days"); ax[0].set_ylabel("CSF vector conc (vg/mL)")
    ax[0].set_title("CSF kinetics by segment"); ax[0].legend(frameon=False)

    for r, reg in enumerate(REGIONS):
        ax[1].plot(t, states["Vcell"][r], color=_COL[reg], label=reg)
    ax[1].set_xlabel("days"); ax[1].set_ylabel("transduced vector (vg)")
    ax[1].set_title("Internalized / transduced vector"); ax[1].legend(frameon=False)

    total_p = sum(states["P"][r] for r in range(len(REGIONS)))
    ax[2].plot(t, total_p, color="#3a7d44", lw=2)
    ax[2].set_xlabel("days"); ax[2].set_ylabel("transgene product (AU)")
    ax[2].set_title("Total transgene expression")

    fig.suptitle(f"{phys['label']} - {site} - dose {dose:.2e} vg", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_kinetics.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_route_effect(phys, bio, dose, outdir, t_end=120.0):
    """ICM vs lumbar: cranial CSF and transduction in the cranial region."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for site, c in [("ICM", "#2c6fbb"), ("lumbar", "#9c2c2c")]:
        t, states = simulate(phys, bio, dose=dose, site=site, t_end=t_end)
        ax[0].plot(t, _floor_for_log(states["Vcsf"][0] / phys["csf_seg_mL"][0]), color=c, label=site)
        ax[1].plot(t, states["Vcell"][0], color=c, label=site)
    ax[0].set_yscale("log"); ax[0].set_xlim(0, 10)
    ax[0].set_xlabel("days"); ax[0].set_ylabel("cranial CSF conc (vg/mL)")
    ax[0].set_title("Cranial CSF exposure: ICM vs lumbar"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("days"); ax[1].set_ylabel("transduced vector, cranial (vg)")
    ax[1].set_title("Brain transduction: ICM vs lumbar"); ax[1].legend(frameon=False)
    fig.suptitle(f"{phys['label']} - route comparison", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_route_effect.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_translation(table, outdir):
    """Bar charts of projected total dose and of the normalization bases."""
    species = list(table["species"])
    x = np.arange(len(species))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ax[0].bar(x, table["total_vg"], color="#2c6fbb")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(species)
    ax[0].set_ylabel("projected total dose (vg)")
    ax[0].set_title("Total dose to match the same exposure")

    width = 0.27
    for i, (col, lab, c) in enumerate([
        ("vg_per_kg", "vg/kg", "#9c2c2c"),
        ("vg_per_g_brain", "vg/g brain", "#d98c00"),
        ("vg_per_mL_csf", "vg/mL CSF", "#3a7d44"),
    ]):
        vals = np.array(table[col], dtype=float)
        vals = vals / vals[0]                       # normalize to first species
        ax[1].bar(x + (i - 1) * width, vals, width, label=lab, color=c)
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(species)
    ax[1].set_ylabel("dose on each basis (relative to mouse)")
    ax[1].set_title("Which basis is conserved?\n(flat across species = good basis)")
    ax[1].legend(frameon=False)
    ax[1].axhline(1.0, color="gray", ls=":", lw=1)

    fig.tight_layout()
    path = os.path.join(outdir, "fig3_translation.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_data_update(phys, bio_prior, bio_updated, dose, site, days,
                     observed, outdir):
    """Observed vs prior-model vs updated-model transduction, one panel per
    necropsy day. `observed` is the day-major stacked vector (len = days x regions)
    produced by data_integration._stack_days / make_synthetic_observation.
    """
    from .data_integration import predict_region_transduced
    days = np.atleast_1d(days).astype(float)
    observed = np.asarray(observed, dtype=float).reshape(len(days), len(REGIONS))
    x = np.arange(len(REGIONS))
    fig, axes = plt.subplots(1, len(days), figsize=(4.7 * len(days), 4.4),
                             squeeze=False)
    for j, day in enumerate(days):
        ax = axes[0][j]
        pred_prior = predict_region_transduced(phys, bio_prior, dose, site, day)
        pred_post = predict_region_transduced(phys, bio_updated, dose, site, day)
        ax.scatter(x, observed[j], s=80, color="black", zorder=5, label="observed")
        ax.plot(x, pred_prior, "o--", color="#9c2c2c", label="prior model")
        ax.plot(x, pred_post, "o-", color="#3a7d44", label="updated model")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(REGIONS)
        ax.set_title(f"necropsy day {day:g}")
        if j == 0:
            ax.set_ylabel("transduced vector (vg)")
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{phys['label']}: model update from new biodistribution data",
                 y=1.03)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_data_update.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Extension figures
# ---------------------------------------------------------------------------

def plot_saturable_dose_response(phys, bio, site, outdir, doses=None, t_end=120.0):
    """Linear vs receptor-limited (saturable) uptake: dose-response and per-vg
    efficiency. Saturation makes high doses give diminishing transduction."""
    from .model import simulate, exposure_metrics
    if doses is None:
        doses = np.logspace(11, 15, 13)
    doses = np.asarray(doses, dtype=float)

    def curve(sat_up):
        b = dict(bio); b["saturable_uptake"] = sat_up
        vals = []
        for d in doses:
            t, st = simulate(phys, b, dose=d, site=site, t_end=t_end)
            vals.append(exposure_metrics(t, st, phys)["transduced_per_g_cns"])
        return np.array(vals)

    lin = curve(False)
    sat = curve(True)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].loglog(doses, lin, "o-", color="#2c6fbb", label="first-order uptake")
    ax[0].loglog(doses, sat, "o-", color="#9c2c2c", label="saturable uptake")
    ax[0].set_xlabel("total dose (vg)")
    ax[0].set_ylabel("transduced vg / g CNS")
    ax[0].set_title("Dose-response"); ax[0].legend(frameon=False)

    ax[1].loglog(doses, lin / doses, "o-", color="#2c6fbb", label="first-order")
    ax[1].loglog(doses, sat / doses, "o-", color="#9c2c2c", label="saturable")
    ax[1].set_xlabel("total dose (vg)")
    ax[1].set_ylabel("transduced (vg/g) per vg dosed")
    ax[1].set_title("Per-vg efficiency\n(falls once uptake saturates)")
    ax[1].legend(frameon=False)

    fig.suptitle(f"{phys['label']} - {site}: saturable uptake at high dose", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_saturable.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_drg_safety(phys, bio_serotype, dose, outdir, day=28.0,
                    threshold=None, bio_alt=None, alt_label="alt serotype",
                    serotype_label="AAV9"):
    """DRG vector load by ganglion. Panel A: route effect (ICM vs lumbar).
    Panel B: serotype effect (e.g. AAV9 vs DRG-detargeted) at ICM."""
    from . import pd_safety as ps
    if threshold is None:
        threshold = ps.DRG_TOX_THRESHOLD_VG_PER_DIPLOID
    gang = ps.GANGLIA
    x = np.arange(len(gang))
    w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for i, (site, c) in enumerate([("ICM", "#2c6fbb"), ("lumbar", "#9c2c2c")]):
        tab = ps.drg_load_table(phys, bio_serotype, dose, site, day, threshold)
        ax[0].bar(x + (i - 0.5) * w, tab["vg_per_diploid_genome"], w, label=site, color=c)
    ax[0].axhline(threshold, color="black", ls="--", lw=1, label="tox threshold")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(gang, rotation=15)
    ax[0].set_ylabel("vg / diploid genome (DRG)")
    ax[0].set_title(f"{serotype_label}: route effect on DRG load")
    ax[0].legend(frameon=False, fontsize=9)

    tab_main = ps.drg_load_table(phys, bio_serotype, dose, "ICM", day, threshold)
    ax[1].bar(x - w / 2, tab_main["vg_per_diploid_genome"], w,
              label=serotype_label, color="#2c6fbb")
    if bio_alt is not None:
        tab_alt = ps.drg_load_table(phys, bio_alt, dose, "ICM", day, threshold)
        ax[1].bar(x + w / 2, tab_alt["vg_per_diploid_genome"], w,
                  label=alt_label, color="#3a7d44")
    ax[1].axhline(threshold, color="black", ls="--", lw=1, label="tox threshold")
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(gang, rotation=15)
    ax[1].set_ylabel("vg / diploid genome (DRG)")
    ax[1].set_title("Capsid-engineering lever (ICM)")
    ax[1].legend(frameon=False, fontsize=9)

    fig.suptitle(f"{phys['label']}: DRG safety readout - dose {dose:.1e} vg", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig6_drg_safety.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_pd_response(phys, bio, site, outdir, transgene=None, doses=None,
                     target=0.7, t_end=120.0):
    """Substrate reduction over time at several doses, and the terminal
    dose-response with a PD target marked."""
    from . import pd_safety as ps
    if doses is None:
        doses = [3e12, 1e13, 3e13, 1e14]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(doses)))
    for d, c in zip(doses, cmap):
        pr = ps.pd_response(phys, bio, d, site, transgene, t_end)
        ax[0].plot(pr["t"], 100 * pr["reduction"][0], color=c, label=f"{d:.0e} vg")
    ax[0].set_xlabel("days"); ax[0].set_ylabel("brain substrate reduction (%)")
    ax[0].set_ylim(0, 100); ax[0].set_title("PD over time (brain)")
    ax[0].legend(frameon=False, fontsize=8, title="dose")

    dose_grid = np.logspace(11.7, 14.7, 16)
    term = [100 * ps.pd_response(phys, bio, d, site, transgene, t_end)["brain_terminal_reduction"]
            for d in dose_grid]
    ax[1].semilogx(dose_grid, term, "o-", color="#3a7d44")
    ax[1].axhline(100 * target, color="black", ls="--", lw=1,
                  label=f"target {100*target:.0f}%")
    try:
        d_star = ps.dose_for_pd_target(phys, bio, site, target, transgene, t_end)
        ax[1].axvline(d_star, color="#9c2c2c", ls=":", lw=1.5,
                      label=f"dose = {d_star:.1e} vg")
    except ValueError:
        pass
    ax[1].set_xlabel("total dose (vg)")
    ax[1].set_ylabel("terminal brain substrate reduction (%)")
    ax[1].set_ylim(0, 100); ax[1].set_title("Dose for a PD target")
    ax[1].legend(frameon=False, fontsize=9)

    fig.suptitle(f"{phys['label']} - {site}: transgene PD (cross-correcting enzyme)",
                 y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig7_pd_response.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def plot_bayesian(post, dose_post, outdir, truth=None, prior_dose=None):
    """Posterior over the two fitted multipliers, and the resulting human-dose
    posterior with a 95% credible interval."""
    samples = np.asarray(post["samples"], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax[0].scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.25, color="#2c6fbb",
                  edgecolors="none")
    ax[0].scatter([post["f_uptake_median"]], [post["f_csf_clear_median"]],
                  s=90, color="#3a7d44", zorder=5, label="posterior median")
    if truth is not None:
        ax[0].scatter([truth[0]], [truth[1]], marker="*", s=220, color="#9c2c2c",
                      zorder=6, label="truth")
    ax[0].set_xlabel("CSF->tissue uptake multiplier")
    ax[0].set_ylabel("CSF clearance multiplier")
    ax[0].set_title("Joint posterior"); ax[0].legend(frameon=False, fontsize=9)

    doses = np.asarray(dose_post["doses"], dtype=float)
    ax[1].hist(doses, bins=30, color="#2c6fbb", alpha=0.8)
    lo, hi = dose_post["ci"]
    ax[1].axvspan(lo, hi, color="#3a7d44", alpha=0.18, label="95% CrI")
    ax[1].axvline(dose_post["median"], color="#3a7d44", lw=2, label="median")
    if prior_dose is not None:
        ax[1].axvline(prior_dose, color="#9c2c2c", ls=":", lw=1.5,
                      label="prior point dose")
    ax[1].set_xlabel("projected human dose (vg)")
    ax[1].set_ylabel("posterior draws")
    ax[1].set_title("Human dose: posterior + 95% CrI")
    ax[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Bayesian update: parameter posterior -> dose uncertainty", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "fig8_bayesian.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path

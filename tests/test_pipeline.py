"""
Tests that double as executable documentation of the model's claims.

Each test asserts a *scientific* property the package is supposed to have, not
just that code runs: route effects, dose linearity, the identifiability-driven
data update, the DRG safety straddle, saturable sublinearity, PD monotonicity,
and Bayesian recovery of known truth. Tolerances are intentionally loose enough
to be robust across platforms but tight enough to catch regressions.
"""

import numpy as np
import pytest

from aav_it_dmpk import physiology as P
from aav_it_dmpk.model import (default_bio, apply_serotype, simulate,
                               exposure_metrics)
from aav_it_dmpk import translation as tr
from aav_it_dmpk import data_integration as di
from aav_it_dmpk import pd_safety as ps
from aav_it_dmpk import validation as V
from aav_it_dmpk import vpc as VPC


@pytest.fixture
def bio():
    return default_bio()


@pytest.fixture
def cyno():
    return P.get_species("cyno")


@pytest.fixture
def human():
    return P.get_species("human")


def test_route_brain_transduction_icm_beats_lumbar(bio, cyno):
    """Cisternal dosing reaches the brain far better than lumbar."""
    _, s_icm = simulate(cyno, bio, dose=3e13, site="ICM")
    _, s_lum = simulate(cyno, bio, dose=3e13, site="lumbar")
    ratio = s_icm["Vcell"][0][-1] / s_lum["Vcell"][0][-1]
    assert 4.0 < ratio < 6.0, ratio


def test_dose_linearity_baseline(bio, cyno):
    """Baseline model is linear in dose: unit-dose trajectory * dose == full sim."""
    t1, s1 = simulate(cyno, bio, dose=1.0, site="ICM")
    tD, sD = simulate(cyno, bio, dose=3e13, site="ICM")
    np.testing.assert_allclose(sD["Vcell"], s1["Vcell"] * 3e13, rtol=1e-6)


def test_biodistribution_units_and_known_artifact(bio, cyno):
    """qPCR-style units; brain dominates absolute load but per-mass is highest
    in cord (the documented coarse-model artifact)."""
    from aav_it_dmpk import biodistribution as bd
    df = bd.biodistribution_table(cyno, bio, 3e13, "ICM", 28.0).set_index("tissue")
    assert 110 < df.loc["brain", "vg_per_diploid_genome"] < 160
    assert df.loc["brain", "vector_genomes_vg"] > df.loc["lumbar_cord", "vector_genomes_vg"]
    assert df.loc["lumbar_cord", "vg_per_diploid_genome"] > df.loc["brain", "vg_per_diploid_genome"]


def test_data_update_lowers_human_dose(bio, cyno, human):
    """New NHP data (hotter + longer-lived than prior) lowers the human dose
    needed to hit a fixed efficacy target."""
    M_star = tr.exposure_at_dose("transduced_per_g_cns",
                                 tr.Arm(P.get_species("mouse"), site="ICM", bio=bio), 1e11)
    prior = tr.dose_for_target("transduced_per_g_cns",
                               tr.Arm(human, site="ICM", bio=bio), M_star)
    obs = di.make_synthetic_observation(
        cyno, bio, 3e13, "ICM", [14.0, 42.0], [0.5, 1.0, 2.0, 3.0],
        true_f_uptake=1.8, true_f_csf_clear=0.7, seed=7)
    upd = di.update_from_observations(cyno, bio, 3e13, "ICM", obs)
    post = tr.dose_for_target("transduced_per_g_cns",
                              tr.Arm(human, site="ICM", bio=upd["bio_updated"]), M_star)
    assert 0.5 < post / prior < 0.7
    # the fit should land near the truth
    assert 1.4 < upd["f_uptake"] < 2.0
    assert 0.6 < upd["f_csf_clear"] < 1.0


def test_apply_serotype_multipliers(bio):
    a = apply_serotype(bio, "AAV9")
    assert a["k_uptake"] == pytest.approx(bio["k_uptake"] * 1.3)
    assert a["k_transduce"] == pytest.approx(bio["k_transduce"] * 1.2)
    assert a["k_uptake_drg"] == pytest.approx(0.035)


def test_drg_safety_straddles_threshold(bio, cyno):
    """Route relocates the DRG hotspot; detargeting clears the threshold."""
    thr = ps.DRG_TOX_THRESHOLD_VG_PER_DIPLOID
    aav9 = apply_serotype(bio, "AAV9")
    det = apply_serotype(bio, "AAV9_DRGdetarget")

    icm = ps.drg_load_table(cyno, aav9, 3e13, "ICM", 28.0).set_index("ganglion")
    lum = ps.drg_load_table(cyno, aav9, 3e13, "lumbar", 28.0).set_index("ganglion")
    det_icm = ps.drg_load_table(cyno, det, 3e13, "ICM", 28.0)

    assert icm.loc["cranial_ganglia", "vg_per_diploid_genome"] > thr      # ICM hotspot
    assert icm.loc["lumbar_DRG", "vg_per_diploid_genome"] < thr
    assert lum.loc["lumbar_DRG", "vg_per_diploid_genome"] > thr           # lumbar hotspot
    assert not det_icm["exceeds_threshold"].any()                         # detarget clears


def test_saturable_uptake_is_sublinear(bio, cyno):
    """Per-vg efficiency falls with dose under saturable uptake; flat otherwise."""
    def eff(sat, d):
        b = dict(bio); b["saturable_uptake"] = sat
        t, s = simulate(cyno, b, dose=d, site="ICM")
        return exposure_metrics(t, s, cyno)["transduced_per_g_cns"] / d
    # linear: efficiency constant across 4 logs
    assert eff(False, 1e11) == pytest.approx(eff(False, 1e15), rel=1e-3)
    # saturable: efficiency strictly lower at high dose
    assert eff(True, 1e15) < 0.2 * eff(True, 1e11)


def test_pd_monotonic_and_target(bio, human):
    aav9 = apply_serotype(bio, "AAV9")
    r_lo = ps.pd_response(human, aav9, 1e13, "ICM")["brain_terminal_reduction"]
    r_hi = ps.pd_response(human, aav9, 3e13, "ICM")["brain_terminal_reduction"]
    assert 0.0 < r_lo < r_hi < 1.0
    d_star = ps.dose_for_pd_target(human, aav9, "ICM", 0.70)
    achieved = ps.pd_response(human, aav9, d_star, "ICM")["brain_terminal_reduction"]
    assert achieved == pytest.approx(0.70, abs=0.02)


def test_bayesian_recovers_truth_within_ci(bio, cyno):
    """Seeded posterior should bracket the known truth and accept reasonably."""
    obs = di.make_synthetic_observation(
        cyno, bio, 3e13, "ICM", [14.0, 42.0], [0.5, 1.0, 2.0, 3.0],
        true_f_uptake=1.8, true_f_csf_clear=0.7, seed=7)
    post = di.bayesian_update_from_observations(
        cyno, bio, 3e13, "ICM", obs, n_samples=1500, burn=400, thin=3, seed=0)
    assert 0.2 < post["acceptance"] < 0.7
    lo_u, hi_u = post["f_uptake_ci"]
    lo_c, hi_c = post["f_csf_clear_ci"]
    assert lo_u < 1.8 < hi_u
    assert lo_c < 0.7 < hi_c


def test_dose_posterior_credible_interval(bio, cyno, human):
    M_star = tr.exposure_at_dose("transduced_per_g_cns",
                                 tr.Arm(P.get_species("mouse"), site="ICM", bio=bio), 1e11)
    obs = di.make_synthetic_observation(
        cyno, bio, 3e13, "ICM", [14.0, 42.0], [0.5, 1.0, 2.0, 3.0],
        true_f_uptake=1.8, true_f_csf_clear=0.7, seed=7)
    post = di.bayesian_update_from_observations(
        cyno, bio, 3e13, "ICM", obs, n_samples=1500, burn=400, thin=3, seed=0)
    dp = di.propagate_dose_posterior(
        human, bio, "ICM", "transduced_per_g_cns", M_star,
        post["samples"], max_draws=200, seed=1)
    lo, hi = dp["ci"]
    assert lo < dp["median"] < hi
    assert 2e13 < dp["median"] < 5e13


# --------------------------------------------------------------------------
# Validation-layer tests (verification, predictive, calibration, sensitivity)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def obs_and_post():
    bio = default_bio()
    cyno = P.get_species("cyno")
    obs = di.make_synthetic_observation(
        cyno, bio, 3e13, "ICM", [14.0, 42.0], [0.5, 1.0, 2.0, 3.0],
        true_f_uptake=1.8, true_f_csf_clear=0.7, seed=7)
    post = di.bayesian_update_from_observations(
        cyno, bio, 3e13, "ICM", obs, n_samples=800, burn=300, thin=2, seed=0)
    return obs, post


def test_mass_balance_conserved(bio, cyno):
    mb = V.mass_balance(cyno, apply_serotype(bio, "AAV9"), 3e13, "ICM")
    assert mb["max_rel_imbalance"] < 1e-6


def test_matrix_exponential_agreement(bio, cyno):
    ev = V.verify_against_matrix_exponential(cyno, apply_serotype(bio, "AAV9"), 3e13, "ICM")
    assert ev["max_rel_error"] < 1e-5


def test_solver_convergence(bio, cyno):
    assert V.convergence(cyno, apply_serotype(bio, "AAV9"), 3e13, "ICM")["rel_diff"] < 1e-4


def test_limiting_cases(cyno):
    lc = V.limiting_cases(cyno, "ICM")
    assert lc["zero_uptake_ok"] and lc["linearity_ok"]


def test_posterior_predictive_coverage(bio, cyno, obs_and_post):
    obs, post = obs_and_post
    ppc = V.posterior_predictive_check(cyno, bio, 3e13, "ICM", obs, post["samples"], n_rep=200)
    assert ppc["coverage_overall"] >= 0.8


def test_leave_one_cohort_out_within_2fold(bio, cyno, obs_and_post):
    obs, _ = obs_and_post
    for r in V.leave_one_cohort_out(cyno, bio, 3e13, "ICM", obs):
        assert r["max_fold"] < 2.0


def test_sobol_identifies_top_driver(bio, cyno):
    so = V.pipeline_sobol(cyno, base_bio=apply_serotype(bio, "AAV9"),
                          dose=3e13, site="ICM",
                          output="transduced_per_g_cns", n_base=64, seed=0)
    assert len(so["S1"]) == 6 and len(so["ST"]) == 6
    assert 0.5 < float(sum(so["ST"])) < 1.6          # mostly additive
    assert so["names"][int(np.argmax(so["ST"]))] == "k_vcell_loss"


def test_sbc_runs_and_ranks_in_range(bio, cyno):
    sbc = V.simulation_based_calibration(
        cyno, bio, 3e13, "ICM", n_sims=8, n_samples=150, burn=60, thin=2, seed=1)
    assert len(sbc["ranks_uptake"]) == 8
    assert sbc["ranks_uptake"].min() >= 0
    assert sbc["ranks_uptake"].max() <= sbc["n_posterior"]


# --------------------------------------------------------------------------
# VPC (visual predictive check) tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vpc_study_post():
    bio = default_bio()
    cyno = P.get_species("cyno")
    study = VPC.make_synthetic_study(
        cyno, bio, 3e13, "ICM",
        csf_days=[0.25, 0.5, 1.0, 2.0, 3.0], necropsy_days=[14.0, 42.0, 90.0],
        n_csf=6, n_tissue=5, true_f_uptake=1.8, true_f_csf_clear=0.7, seed=11)
    post = di.bayesian_update_from_observations(
        cyno, bio, 3e13, "ICM", VPC.study_to_observation(study),
        n_samples=600, burn=200, thin=2, seed=0)
    return study, post["samples"]


def test_study_to_observation_shape(vpc_study_post):
    study, _ = vpc_study_post
    obs = VPC.study_to_observation(study)
    assert len(obs["tissue"]) == len(study["necropsy_days"]) * 3
    assert len(obs["csf"]) == len(study["csf_days"])


def test_vpc_csf_bands_ordered_and_cover(vpc_study_post):
    study, samples = vpc_study_post
    cyno, bio = P.get_species("cyno"), default_bio()
    vc = VPC.vpc_csf(cyno, bio, 3e13, "ICM", study, samples, n_sim=200)
    nt = len(study["csf_days"])
    for p in vc["pi"]:
        b = vc["bands"][p]
        assert len(b["median"]) == nt and len(b["lo"]) == nt and len(b["hi"]) == nt
        assert np.all(b["lo"] <= b["median"] + 1e-9)
        assert np.all(b["median"] <= b["hi"] + 1e-9)
        assert len(vc["obs_pct"][p]) == nt
    # observed median sits inside the outer simulated band at every time
    assert np.all(vc["obs_pct"][50] >= vc["bands"][5]["lo"])
    assert np.all(vc["obs_pct"][50] <= vc["bands"][95]["hi"])


def test_pcvpc_runs_finite(vpc_study_post):
    study, samples = vpc_study_post
    cyno, bio = P.get_species("cyno"), default_bio()
    vc = VPC.vpc_csf(cyno, bio, 3e13, "ICM", study, samples, n_sim=200,
                     prediction_corrected=True)
    assert vc["prediction_corrected"]
    for p in vc["pi"]:
        for key in ("median", "lo", "hi"):
            assert np.all(np.isfinite(vc["bands"][p][key]))


def test_vpc_tissue_per_region(vpc_study_post):
    study, samples = vpc_study_post
    cyno, bio = P.get_species("cyno"), default_bio()
    vt = VPC.vpc_tissue(cyno, bio, 3e13, "ICM", study, samples, n_sim=200)
    assert len(vt["per_region"]) == 3
    nt = len(study["necropsy_days"])
    for rp in vt["per_region"]:
        assert len(rp["bands"][50]["median"]) == nt
        assert len(rp["obs_pct"][50]) == nt

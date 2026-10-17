import numpy as np
from statsmodels.stats.multitest import multipletests

from qfmb.stats import benjamini_hochberg, corrected_ttest, holm


def test_holm_bh_match_statsmodels():
    p = np.random.default_rng(0).uniform(0, 0.2, 30)
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])
    assert np.allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_corrected_ttest_is_more_conservative():
    d = np.random.default_rng(1).normal(0.01, 0.03, 100)
    r = corrected_ttest(d, 270, 30)
    from scipy import stats
    assert r["p"] > stats.ttest_1samp(d, 0).pvalue
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]

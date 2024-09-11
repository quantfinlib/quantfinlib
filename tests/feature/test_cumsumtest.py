import numpy as np

from quantfinlib.feature.structural_breaks.cusumtest import chu_stinchcombe_white_cusum_test

np.random.seed(42)
y_input = np.random.lognormal(mean=0.1, sigma=0.1, size=1000)


def test_bruteforce_calculation_cumsumtest():
    """Test the CUSUM test statistic and critical values against a brute-force calculation with nested for loops."""
    S_calculated, crit_vals_calculated = chu_stinchcombe_white_cusum_test(y=y_input)

    T = len(y_input)
    b_alpha = 4.6
    log_y = np.log(y_input)
    S_expected = np.zeros(T)
    crit_vals_expected = np.zeros(T)
    for t in range(1, T):
        max_s_n_t: float = -np.inf
        crit_val: float = 0.0
        squared_diff = (log_y[1 : t + 1] - log_y[0:t]) ** 2.0
        sigma_hat_t = np.sqrt(np.mean(squared_diff))
        for n in range(0, t):
            s_n_t = (log_y[t] - log_y[n]) / (sigma_hat_t * np.sqrt(t - n))
            if s_n_t > max_s_n_t:
                max_s_n_t = s_n_t
                crit_val = b_alpha + np.log(t - n)
        S_expected[t] = max_s_n_t
        crit_vals_expected[t] = crit_val
    np.testing.assert_allclose(
        S_calculated, S_expected, rtol=1e-5, err_msg="The calculated and expected CUSUM test statistic do not match."
    )
    np.testing.assert_allclose(
        crit_vals_calculated,
        crit_vals_expected,
        rtol=1e-5,
        err_msg="The calculated and expected critical values do not match.",
    )
    return None


def test_trivial_properties_cumsumtest():
    """Test the CUSUM test statistic and critical values for trivial properties."""
    S, crit_vals = chu_stinchcombe_white_cusum_test(y=y_input)
    assert S.shape[0] == len(y_input), "The length of the CUSUM test statistic array is not equal to the input array."
    assert crit_vals.shape[0] == len(
        y_input
    ), "The length of the critical values array is not equal to the input array."
    b_alpha = 4.6
    assert np.all(crit_vals[1:] >= b_alpha), "The critical values are not greater than the constant b_alpha."

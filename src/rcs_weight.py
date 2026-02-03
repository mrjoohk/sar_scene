import numpy as np
from scipy.interpolate import interp1d

def make_W_of_theta(theta_deg_samples, rcs_dbsm_samples, kind="linear"):
    """
    W(theta) = sigma(theta)/mean(sigma), 주기 보간.
    """
    th = np.asarray(theta_deg_samples) % 360.0
    sig_lin = 10.0 ** (np.asarray(rcs_dbsm_samples) / 10.0)
    sig_lin = np.maximum(sig_lin, 1e-30)

    th2  = np.concatenate([th, th[:1] + 360.0])
    sig2 = np.concatenate([sig_lin, sig_lin[:1]])

    W2 = sig2 / (np.mean(sig_lin) + 1e-30)
    f = interp1d(th2, W2, kind=kind, fill_value="extrapolate", bounds_error=False)

    def W(theta_deg):
        return float(f(float(theta_deg) % 360.0))

    return W

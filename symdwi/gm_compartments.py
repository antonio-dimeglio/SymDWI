import numpy as np
from scipy.special import erf


GAMMA = 2.67513e8  # proton gyromagnetic ratii=o
_NEUMAN_ROOTS = np.array([
    1.8412, 5.3314, 8.5363, 11.7060, 14.8636, 18.0155, 21.1644, 24.3113,
    27.4571, 30.6019, 33.7462, 36.8837, 40.0152, 43.1412, 46.2622, 49.3786,
    52.4903, 55.5978, 58.7012, 61.8010,
])


def gpd_sphere_attenuation(
    bvals: np.ndarray,
    big_delta_ms: float,
    small_delta_ms: float,
    d_is_mm2_s: float,
    r_s_um: float,
    n_roots: int = 20,
) -> np.ndarray:
    """
    Signal attenuation S/S0 for restricted diffusion inside an impermeable
    sphere, under the Gaussian Phase Distribution (GPD) approximation for a
    rectangular-pulse (PGSE) sequence.

    Parameters
    ----------
    bvals : np.ndarray, shape (N,)
        b-values in s/mm^2.
    big_delta_ms : float
        Diffusion time Delta, in ms (time between gradient pulse leading
        edges).
    small_delta_ms : float
        Gradient pulse duration delta, in ms.
    d_is_mm2_s : float
        Intra-soma (intrinsic, unrestricted) diffusivity, mm^2/s.
    r_s_um : float
        Soma (sphere) radius, in micrometers.
    n_roots : int
        Number of terms in the Neuman series. Default 20 (converged).

    Returns
    -------
    np.ndarray, shape (N,)
        Dimensionless attenuation in [0, 1]. Exactly 1.0 at b=0.
    """
    bvals = np.asarray(bvals, dtype=float)

    b_si = bvals * 1e6
    delta_s = small_delta_ms * 1e-3
    Delta_s = big_delta_ms * 1e-3
    r_s_m = r_s_um * 1e-6
    d_is_si = d_is_mm2_s * 1e-6

    alpha = _NEUMAN_ROOTS[:n_roots] / r_s_m

    a2d = alpha[None, :] ** 2 * d_is_si
    term = (
        2 * a2d * delta_s - 2
        + 2 * np.exp(-a2d * delta_s)
        + 2 * np.exp(-a2d * Delta_s)
        - np.exp(-a2d * (Delta_s - delta_s))
        - np.exp(-a2d * (Delta_s + delta_s))
    )
    denom = d_is_si ** 2 * alpha[None, :] ** 6 * (alpha[None, :] ** 2 * r_s_m ** 2 - 2)
    series_sum = np.sum(term / denom, axis=1)

    nonzero = b_si > 0.0
    G2 = np.zeros_like(b_si)
    G2[nonzero] = b_si[nonzero] / (GAMMA ** 2 * delta_s ** 2 * (Delta_s - delta_s / 3.0))

    log_atten = -2.0 * GAMMA ** 2 * G2 * series_sum
    atten = np.exp(log_atten)
    atten = np.where(nonzero, atten, 1.0)
    return atten


def isotropic_stick_attenuation(bvals: np.ndarray, d_in_mm2_s: float) -> np.ndarray:
    """
    Direction-averaged (powder-averaged) signal for a stick compartment
    (zero radial diffusivity) whose axis is uniformly distributed over the
    sphere.

    Parameters
    ----------
    bvals : np.ndarray, shape (N,)
        b-values in s/mm^2.
    d_in_mm2_s : float
        Neurite (stick) diffusivity, mm^2/s.

    Returns
    -------
    np.ndarray, shape (N,)
        Dimensionless attenuation in [0, 1]. Exactly 1.0 at b=0.
    """
    bvals = np.asarray(bvals, dtype=float)
    x = bvals * d_in_mm2_s
    sqrt_x = np.sqrt(np.where(x > 0.0, x, 1.0))
    atten = np.where(
        x > 1e-12,
        np.sqrt(np.pi) / (2.0 * sqrt_x) * erf(sqrt_x),
        1.0,
    )
    return atten
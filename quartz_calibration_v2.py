# Determine peak position of the ~464 cm⁻¹ quartz band for Raman shift
# calibration. Applies a linear baseline correction and fits using both a
# Lorentzian (or pseudo-Voigt) and a parabolic apex method. The parabolic
# fit is recommended for sparse data. Also computes the corrected laser
# wavelength using Fred's formula.

# Author: Berke Santos
# Script developed with the help of Claude.AI
# Created: 10/02/2026

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
import warnings


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_xy_txt(path):

    data = np.genfromtxt(path, comments="#", delimiter=None)

    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError("File must contain at least two columns.")

    # check for malformed lines that produced NaNs
    nan_mask = np.isnan(data[:, 0]) | np.isnan(data[:, 1])
    if nan_mask.any():
        n_bad = nan_mask.sum()
        warnings.warn(
            f"Dropped {n_bad} row(s) containing NaN values "
            f"(malformed lines in input file)."
        )
        data = data[~nan_mask]

    if len(data) == 0:
        raise ValueError("No valid data rows after removing NaNs.")

    x, y = data[:, 0], data[:, 1]

    # guarantee monotonic x order
    order = np.argsort(x)
    return x[order], y[order]


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------
def linear_baseline(x, y, bl_regions):

    mask = np.zeros_like(x, dtype=bool)
    for lo, hi in bl_regions:
        mask |= (x >= lo) & (x <= hi)

    p = np.polyfit(x[mask], y[mask], 1)
    return np.polyval(p, x)


# ---------------------------------------------------------------------------
# NOISE ESTIMATE
# ---------------------------------------------------------------------------
def estimate_noise(y, bl_mask):
    """Estimate per-point sigma from the standard deviation in the
    baseline regions. Returns an array of constant sigma values
    (homoscedastic assumption) for use in curve_fit."""

    residuals = y[bl_mask] - np.mean(y[bl_mask])
    sigma = np.std(residuals, ddof=1)

    if sigma == 0:
        warnings.warn("Zero noise estimated from baseline regions; "
                       "falling back to unit weights.")
        return np.ones_like(y)

    return np.full_like(y, sigma)


# ---------------------------------------------------------------------------
# LINE PROFILES
# ---------------------------------------------------------------------------
def lorentzian(x, x0, gamma, A, offset):

    return A * gamma / (4 * (x - x0) ** 2 + gamma ** 2) + offset


def pseudo_voigt(x, x0, fwhm, A, offset, eta):
    """Pseudo-Voigt: linear combination of Lorentzian and Gaussian
    sharing the same FWHM and center.
        eta = 1  → pure Lorentzian
        eta = 0  → pure Gaussian
    """

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))   # Gaussian std dev

    L = (fwhm / 2) ** 2 / ((x - x0) ** 2 + (fwhm / 2) ** 2)
    G = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    # normalise so that A controls peak height
    return A * (eta * L + (1 - eta) * G) + offset


# ---------------------------------------------------------------------------
# FITTING
# ---------------------------------------------------------------------------
def fit_peak(x, y, sigma, profile="lorentzian", p0=None):
    """Fit the chosen profile to baseline-corrected data.

    Parameters
    ----------
    profile : str
        'lorentzian' or 'pseudo-voigt'
    sigma : array
        Per-point uncertainty estimate (from estimate_noise).

    Returns
    -------
    popt, pcov, yfit, param_names
    """

    idx_max = np.argmax(y)
    x0_guess = x[idx_max]
    fwhm_guess = 10.0
    offset_guess = np.median(np.concatenate([y[:5], y[-5:]]))
    A_guess = y[idx_max] - offset_guess

    if profile == "lorentzian":
        func = lorentzian
        if p0 is None:
            gamma_guess = fwhm_guess
            A_lor = A_guess * gamma_guess   # A param scales differently
            p0 = [x0_guess, gamma_guess, A_lor, offset_guess]
        param_names = ["x0", "gamma (FWHM)", "A", "offset"]
        bounds = (-np.inf, np.inf)

    elif profile == "pseudo-voigt":
        func = pseudo_voigt
        if p0 is None:
            p0 = [x0_guess, fwhm_guess, A_guess, offset_guess, 0.5]
        param_names = ["x0", "FWHM", "A", "offset", "eta (Lor fraction)"]
        # eta must stay in [0, 1]
        bounds = (
            [-np.inf, 0, 0, -np.inf, 0.0],
            [ np.inf, np.inf, np.inf, np.inf, 1.0],
        )
    else:
        raise ValueError(f"Unknown profile '{profile}'. "
                         f"Use 'lorentzian' or 'pseudo-voigt'.")

    popt, pcov = curve_fit(
        func, x, y, p0=p0, sigma=sigma,
        absolute_sigma=True, maxfev=10000, bounds=bounds,
    )
    yfit = func(x, *popt)
    return popt, pcov, yfit, param_names


# ---------------------------------------------------------------------------
# OFFSET SANITY CHECK
# ---------------------------------------------------------------------------
def check_offset(popt, param_names):
    """Warn if the fitted offset is large relative to peak height,
    which may indicate a poor baseline correction."""

    offset = popt[param_names.index("offset")]
    A = popt[param_names.index("A")]

    # for the Lorentzian, peak height ≈ A / gamma; for pseudo-Voigt, A is
    # already the peak height. Use a simple heuristic: compare |offset|
    # to the maximum of the fitted curve above offset.
    peak_height = abs(A)
    if peak_height == 0:
        return

    ratio = abs(offset) / peak_height
    if ratio > 0.10:
        warnings.warn(
            f"Fitted offset ({offset:.2f}) is {ratio:.0%} of the peak "
            f"height. This may indicate an inaccurate baseline correction."
        )


# ---------------------------------------------------------------------------
# PARABOLIC APEX FIT
# ---------------------------------------------------------------------------
def fit_parabolic_apex(x, y, n_points=5):
    """Fit a parabola to the n_points highest points around the peak
    maximum and return the vertex position.

    This is robust with sparse data because it only needs 3 parameters
    (a, b, c) and focuses on the well-sampled apex region.

    Returns
    -------
    x0 : float
        Peak center (vertex of parabola)
    x0_unc : float
        Uncertainty on peak center from covariance
    yfit_apex : tuple of (x_dense, y_dense)
        Dense parabola curve for plotting
    x_used, y_used : arrays
        The points actually used in the fit
    """

    idx_max = np.argmax(y)

    # select n_points centered on the maximum
    half = n_points // 2
    i_lo = max(0, idx_max - half)
    i_hi = min(len(x), i_lo + n_points)
    i_lo = max(0, i_hi - n_points)  # adjust if near array end

    x_used = x[i_lo:i_hi]
    y_used = y[i_lo:i_hi]

    if len(x_used) < 3:
        raise ValueError("Need at least 3 points for parabolic fit.")

    # fit y = a*x^2 + b*x + c
    coeffs, cov = np.polyfit(x_used, y_used, 2, cov=True)
    a, b, c = coeffs

    if a >= 0:
        warnings.warn("Parabolic fit has positive curvature — "
                       "no maximum found. Check your data/window.")
        return np.nan, np.nan, (np.array([]), np.array([])), x_used, y_used

    # vertex: x0 = -b / (2a)
    x0 = -b / (2 * a)

    # uncertainty via error propagation: σ_x0 from cov(a,b)
    # ∂x0/∂a = b/(2a²),  ∂x0/∂b = -1/(2a)
    da = b / (2 * a ** 2)
    db = -1.0 / (2 * a)
    x0_var = da**2 * cov[0, 0] + db**2 * cov[1, 1] + 2 * da * db * cov[0, 1]
    x0_unc = np.sqrt(max(x0_var, 0))

    # dense curve for plotting
    x_dense = np.linspace(x_used[0], x_used[-1], 200)
    y_dense = np.polyval(coeffs, x_dense)

    return x0, x0_unc, (x_dense, y_dense), x_used, y_used


# ---------------------------------------------------------------------------
# CENTROID (for comparison)
# ---------------------------------------------------------------------------
def centroid(x, y):

    w = np.clip(y, 0, None)
    if w.sum() == 0:
        return np.nan

    return (x * w).sum() / w.sum()


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
infile = r"C:\Users\berke.santos\Documents\CEMHTI\RAMAN\Quartz-calibrations\2026-02-24.txt"

# spectral window around the ~464 cm-1 quartz band
xmin, xmax = 410, 530

# baseline regions: flat zones on either side of the peak
bl_regions = [(415, 430), (505, 520)]

# fit profile: 'lorentzian' or 'pseudo-voigt'
profile = "pseudo-voigt"

# laser wavelength correction (Fred's Excel replacement)
nominal_laser_nm = 532.360          # current laser wavelength in software (nm)
quartz_theoretical = 464.500        # theoretical quartz position (cm⁻¹)

# parabolic apex fit: number of points around the maximum to use
n_apex_points = 5

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
x, y = load_xy_txt(infile)

# extract window
m = (x >= xmin) & (x <= xmax)
xw, yw = x[m], y[m]

# baseline
baseline = linear_baseline(xw, yw, bl_regions)
ycorr = yw - baseline

# noise estimate from baseline regions for proper error propagation
bl_mask = np.zeros_like(xw, dtype=bool)
for lo, hi in bl_regions:
    bl_mask |= (xw >= lo) & (xw <= hi)
sigma = estimate_noise(ycorr, bl_mask)

# fit
popt, pcov, yfit, param_names = fit_peak(xw, ycorr, sigma, profile=profile)
perr = np.sqrt(np.diag(pcov))

# offset sanity check
check_offset(popt, param_names)

# extract key results
peak_center = popt[0]
peak_unc = perr[0]
fwhm = abs(popt[1])
fwhm_unc = perr[1]

# reduced chi-squared
residuals = ycorr - yfit
ndata = len(xw)
nparam = len(popt)
chi2_red = np.sum((residuals / sigma) ** 2) / (ndata - nparam)

# centroid for comparison
xc = centroid(xw, ycorr)

# parabolic apex fit
para_center, para_unc, (xp_dense, yp_dense), xp_used, yp_used = \
    fit_parabolic_apex(xw, ycorr, n_points=n_apex_points)

# laser wavelength correction (Fred's formula) — using parabolic center
corrected_laser_nm = 1.0 / (
    (1.0 / nominal_laser_nm) - (para_center - quartz_theoretical) / 1e7
)
laser_shift_nm = corrected_laser_nm - nominal_laser_nm

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
print("─" * 50)
print(f"  Quartz calibration – {profile} fit")
print("─" * 50)
print(f"  Lorentzian   : {peak_center:.3f} ± {peak_unc:.3f} cm⁻¹")
print(f"  FWHM         : {fwhm:.3f} ± {fwhm_unc:.3f} cm⁻¹")
if profile == "pseudo-voigt":
    eta = popt[4]
    print(f"  η (Lor frac) : {eta:.3f} ± {perr[4]:.3f}")
print(f"  Red. χ²      : {chi2_red:.3f}")
print(f"  Parabolic    : {para_center:.3f} ± {para_unc:.3f} cm⁻¹  ◄ RECOMMENDED")
print(f"  Centroid      : {xc:.3f} cm⁻¹")
print(f"  Δ (Lor−Para)  : {peak_center - para_center:.3f} cm⁻¹")
print("─" * 50)
print(f"  Laser wavelength correction (from parabolic fit)")
print("─" * 50)
print(f"  Nominal λ    : {nominal_laser_nm:.3f} nm")
print(f"  Corrected λ  : {corrected_laser_nm:.3f} nm")
print(f"  Δλ           : {laser_shift_nm:+.3f} nm")
print("─" * 50)

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- left: raw + baseline ---
ax = axes[0]
ax.plot(xw, yw, "o", ms=3, alpha=0.5, label="Raw spectrum")
ax.plot(xw, baseline, "--", lw=2, label="Linear baseline")
for lo, hi in bl_regions:
    ax.axvspan(lo, hi, color="grey", alpha=0.15)
ax.set_xlabel("Raman shift (cm⁻¹)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Raw spectrum & baseline")
ax.legend(fontsize=9)

# --- middle: corrected + Lorentzian fit ---
ax = axes[1]
ax.plot(xw, ycorr, "o", ms=3, alpha=0.5, label="Baseline-corrected")
ax.plot(xw, yfit, "-", lw=2, color="tab:red",
        label=f"{profile.capitalize()} fit")
ax.axvline(peak_center, color="k", ls=":", lw=1.5,
           label=f"Lor. center = {peak_center:.2f} ± {peak_unc:.2f} cm⁻¹")
ax.set_xlabel("Raman shift (cm⁻¹)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title(f"{profile.capitalize()} fit  (red. χ² = {chi2_red:.2f})")
ax.legend(fontsize=9)

# --- right: parabolic apex fit ---
ax = axes[2]
ax.plot(xw, ycorr, "o", ms=3, alpha=0.3, color="grey", label="All data")
ax.plot(xp_used, yp_used, "s", ms=7, color="tab:green", zorder=5,
        label=f"Apex points ({n_apex_points} pts)")
if len(xp_dense) > 0:
    ax.plot(xp_dense, yp_dense, "-", lw=2, color="tab:green",
            label="Parabolic fit")
ax.axvline(para_center, color="k", ls=":", lw=1.5,
           label=f"Para. center = {para_center:.2f} ± {para_unc:.2f} cm⁻¹")
ax.axvline(peak_center, color="tab:red", ls=":", lw=1, alpha=0.5,
           label=f"Lor. center = {peak_center:.2f} cm⁻¹")
ax.set_xlabel("Raman shift (cm⁻¹)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Parabolic apex fit")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

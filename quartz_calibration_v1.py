# Determine peak position of the ~464 cm⁻¹ quartz band for Raman shift
# calibration. Applies a linear baseline correction and Lorentzian fit.
# Output can be used to calibrate laser wavelength using Fred's designated
# Excel.

# Author: Berke Santos
# Script developed with the help of Claude.AI
# Created: 10/02/2026

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_xy_txt(path):

    data = np.genfromtxt(path, comments="#", delimiter=None)

    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError("File must contain at least two columns.")

    x, y = data[:, 0], data[:, 1]

    # guarantee monotonic x order
    order = np.argsort(x)
    return x[order], y[order]


def linear_baseline(x, y, bl_regions):
    
    mask = np.zeros_like(x, dtype=bool)
    for lo, hi in bl_regions:
        mask |= (x >= lo) & (x <= hi)

    p = np.polyfit(x[mask], y[mask], 1)
    return np.polyval(p, x)


def lorentzian(x, x0, gamma, A, offset):
    
    return A * gamma / (4 * (x - x0) ** 2 + gamma**2) + offset


def fit_lorentzian(x, y, p0=None):
    
    if p0 is None:
        # sensible initial guesses
        idx_max = np.argmax(y)
        x0_guess = x[idx_max]
        gamma_guess = 10.0  # typical FWHM for quartz ~464 band
        offset_guess = np.median(np.concatenate([y[:5], y[-5:]]))
        A_guess = (y[idx_max] - offset_guess) * gamma_guess  # peak height = A / gamma
        p0 = [x0_guess, gamma_guess, A_guess, offset_guess]

    popt, pcov = curve_fit(lorentzian, x, y, p0=p0, maxfev=10000)
    yfit = lorentzian(x, *popt)
    return popt, pcov, yfit


def centroid(x, y):

    w = np.clip(y, 0, None)
    if w.sum() == 0:
        return np.nan
    
    return (x * w).sum() / w.sum()


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
infile = r"C:\Users\berke.santos\Documents\CEMHTI\RAMAN\Quartz-calibrations\2026-02-10-afterCalib.txt"

# spectral window around the ~464 cm-1 quartz band
xmin, xmax = 410, 530

# baseline regions: flat zones on either side of the peak
bl_regions = [(410, 420), (510, 530)]

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

# Lorentzian fit on corrected data
popt, pcov, yfit = fit_lorentzian(xw, ycorr)
perr = np.sqrt(np.diag(pcov))

peak_center = popt[0]
peak_unc = perr[0]
fwhm = abs(popt[1])
fwhm_unc = perr[1]

# centroid for comparison
xc = centroid(xw, ycorr)

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
print("─" * 45)
print("  Quartz calibration – Lorentzian fit")
print("─" * 45)
print(f"  Peak center : {peak_center:.3f} ± {peak_unc:.3f} cm⁻¹")
print(f"  FWHM        : {fwhm:.3f} ± {fwhm_unc:.3f} cm⁻¹")
print(f"  Centroid     : {xc:.3f} cm⁻¹  (for comparison)")
print(f"  Δ (fit−ctr)  : {peak_center - xc:.3f} cm⁻¹")
print("─" * 45)

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- left: raw + baseline ---
ax = axes[0]
ax.plot(xw, yw, "o", ms=3, alpha=0.5, label="Raw spectrum")
ax.plot(xw, baseline, "--", lw=2, label="Linear baseline")
for lo, hi in bl_regions:
    ax.axvspan(lo, hi, color="grey", alpha=0.15)
ax.set_xlabel("Raman shift (rel. cm⁻¹)")
ax.set_ylabel("CCD counts")
ax.set_title("Raw spectrum & baseline")
ax.legend(fontsize=9)

# --- right: corrected + Lorentzian fit ---
ax = axes[1]
ax.plot(xw, ycorr, "o", ms=3, alpha=0.5, label="Baseline-corrected")
ax.plot(xw, yfit, "-", lw=2, color="tab:red", label="Lorentzian fit")
ax.axvline(peak_center, color="k", ls=":", lw=1.5,
           label=f"Fit center = {peak_center:.2f} ± {peak_unc:.2f} cm⁻¹")
ax.axvline(xc, color="tab:blue", ls=":", lw=1.5,
           label=f"Centroid = {xc:.2f} cm⁻¹")
ax.set_xlabel("Raman shift (rel. cm⁻¹)")
ax.set_ylabel("CCD counts")
ax.set_title("Lorentzian fit")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

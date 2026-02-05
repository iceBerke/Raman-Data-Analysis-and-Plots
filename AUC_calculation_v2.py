## RAMAN SPECTROSCOPY AUC ANALYSIS TOOL

# PURPOSE:
#    Calculate Area Under Curve (AUC) for specific Raman bands (e.g., Carbon D and G)
#    This specific version is intended for calculating the AUC of the first-order carbon 
#    bands in order to determine the AUC_D/AUC_G ratio

# FEATURES:
#    - Load single or batch .txt files (single file or folder directory)
#    - Inward-snapping to actual measured data points
#    - Linear baseline correction
#    - Optional Savitzky-Golay smoothing for noisy data
#    - Automatic D/G ratio calculation
#    - Export results to CSV

# General Parameters:
#    d_range : tuple
#        D-band integration range in cm⁻¹ (typically 1300-1400)
#    g_range : tuple
#        G-band integration range in cm⁻¹ (typically 1550-1650)
#    baseline : str
#        "none" or "linear" (linear connects endpoints) or "polynomial"
#    baseline_poly_order : integer
#        specific to polynomial baseline correction
#    smooth : bool
#        Apply Savitzky-Golay filter (recommended for noisy data)
#    smooth_window : int
#        Window size for smoothing (11 recommended, must be odd)
#    smooth_polyorder : int
#        Polynomial order for smoothing (3 recommended)

# NOTES:
#    - Input files must be 2-column text: wavenumber (column 1), intensity (column 2)
#    - Assumes wavenumber is in ascending order (will sort if not)
#    - For highly defective samples, verify D-G bands don't overlap
#    - AUC values are in units of (cm⁻¹ × intensity)

# This version (v2) includes a visualization function of the raw spectrum with the 
# different integration regions highlighted
# It also plots the baseline-corrected spectrum

# AUTHOR: Berke Santos
# Developed with the help of Claude.AI 
# Created: 05/02/2026

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ============= DATA LOADING =============

def load_raman_data(filepath, skiprows=0, delimiter=None):
    
    filepath = Path(filepath)
    
    # Load data
    data = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter)
    
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected 2-column data, got shape {data.shape}")
    
    x = data[:, 0]
    y = data[:, 1]
    
    return x, y


def load_batch_raman(folder, pattern='*.txt', skiprows=0, delimiter=None):
    
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")
    
    data = {}
    failed = []
    
    print(f"Loading files from {folder}...")
    for filepath in files:
        try:
            x, y = load_raman_data(filepath, skiprows=skiprows, delimiter=delimiter)
            data[filepath.stem] = {'x': x, 'y': y}
            print(f"  Check:  {filepath.name}: {len(x)} points")
        except Exception as e:
            failed.append((filepath.name, str(e)))
            print(f"  Failed: {filepath.name}: {e}")
    
    print(f"\nLoaded {len(data)}/{len(files)} files successfully")
    
    return data


# ============= AUC FUNCTIONS =============

def snap_inward_bounds(x, a, b):

    x = np.asarray(x)

    if a > b:
        a, b = b, a
    
    if b < x[0] or a > x[-1]:
        raise ValueError(f"Range [{a}, {b}] is outside data range [{x[0]}, {x[-1]}].")
    
    i0 = np.searchsorted(x, a, side="left")
    #print(i0)
    i1 = np.searchsorted(x, b, side="right") - 1
    #print(i1)

    if i0 >= len(x) or i1 < 0 or i0 > i1:
        raise ValueError("No points remain after snapping bounds inward.")
    
    return i0, i1


def auc_inward(x, y, a, b, baseline, baseline_poly_order, smooth, smooth_window, smooth_polyorder, return_plot_data):
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Ensure sorted by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    
    # Apply smoothing before selecting region
    if smooth:
        if smooth_window % 2 == 0:
            smooth_window += 1
        smooth_window = min(smooth_window, len(y))
        if smooth_window < smooth_polyorder + 2:
            smooth_window = smooth_polyorder + 2
            if smooth_window % 2 == 0:
                smooth_window += 1
        y = savgol_filter(y, window_length=smooth_window, polyorder=smooth_polyorder)

    # Select region
    i0, i1 = snap_inward_bounds(x, a, b)
    xs = x[i0:i1+1]
    #print(x[i0])
    #print(x[i1])
    ys = y[i0:i1+1]
    #print(y[i0])
    #print(y[i1])

    ys_raw = ys.copy()  # Keep original for plotting
    base = np.zeros_like(ys)  # Default: no baseline

    # Baseline correction
    if baseline == "linear":
        x0, x1 = xs[0], xs[-1]
        y0, y1 = ys[0], ys[-1]
        base = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        ys = ys - base

        # Warning for improper baseline correction
        if np.trapezoid(ys, xs) < 0:
            warnings.warn(f"> Negative AUC detected - baseline may be inappropriate")

    elif baseline == "polynomial":
        # Use only the edges (first/last ~10% of points) to define baseline
        # Do not use more than 25%
        n_edge = max(5, len(xs) // 10)
        n_edge = min(n_edge, len(xs) // 4)
        
        x_base = np.concatenate([xs[:n_edge], xs[-n_edge:]])
        y_base = np.concatenate([ys[:n_edge], ys[-n_edge:]])
        
        coeffs = np.polyfit(x_base, y_base, deg=baseline_poly_order)
        base = np.polyval(coeffs, xs)
        ys = ys - base
        
        # Warning for improper baseline correction
        if np.trapezoid(ys, xs) < 0:
            warnings.warn(f"> Negative AUC detected - baseline may be inappropriate")
    
    auc = float(np.trapezoid(ys, xs))
    bounds = (xs[0], xs[-1])
    
    if return_plot_data:
        return {
            'auc': auc,
            'bounds': bounds,
            'xs': xs,
            'y_raw': y,
            'ys_raw': ys_raw,
            'ys_corrected': ys,
            'baseline': base
        }
    else:
        return auc, bounds

# ============= COMPLETE WORKFLOW =============

def analyze_raman_batch(folder, d_range, g_range, baseline, baseline_poly_order, 
                        smooth, smooth_window, smooth_polyorder, skiprows):
    
    # Validation for overlapping ranges (very uncommon)
    if d_range[1] > g_range[0]:
        print(" >  Warning: D and G ranges overlap. Results may not be meaningful.")

    # Load all files
    all_data = load_batch_raman(folder, skiprows=skiprows)
    
    # Process each spectrum
    results = {}
    print(f"\nAnalyzing {len(all_data)} spectra...")
    
    for name, spectrum in all_data.items():
        x = spectrum['x']
        y = spectrum['y']
        
        try:
            # Calculate D-band AUC
            d_auc, d_bounds = auc_inward(
                x, y, *d_range, 
                baseline=baseline,
                baseline_poly_order=baseline_poly_order,  
                smooth=smooth, 
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                return_plot_data=False
            )
            
            # Calculate G-band AUC
            g_auc, g_bounds = auc_inward(
                x, y, *g_range, 
                baseline=baseline,
                baseline_poly_order=baseline_poly_order, 
                smooth=smooth, 
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                return_plot_data=False
            )
            
            # Store results
            results[name] = {
                'D_AUC': d_auc,
                'D_range_used': f"{d_bounds[0]:.3f}-{d_bounds[1]:.3f}",
                'G_AUC': g_auc,
                'G_range_used': f"{g_bounds[0]:.3f}-{g_bounds[1]:.3f}",
                'AUC(D)/AUC(G)': d_auc / g_auc
            }
            
            print(f"  Check: {name}: AUC_D/AUC_G = {d_auc/g_auc:.3f}")
            
        except Exception as e:
            print(f"  Failed: {name}: Failed - {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    return results_df

# ============= VISUALIZATION =============

def plot_auc_analysis(x, y, d_range, g_range, baseline, baseline_poly_order, 
                     smooth, smooth_window, smooth_polyorder, title="Raman Spectrum"):
        
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Sort data
    order = np.argsort(x)
    x = x[order]
    y = y[order]
        
    # Get processed data from auc_inward (this ensures consistency!)
    d_data = auc_inward(x, y, *d_range, baseline, baseline_poly_order, 
                       smooth, smooth_window, smooth_polyorder, return_plot_data=True)
    g_data = auc_inward(x, y, *g_range, baseline, baseline_poly_order, 
                       smooth, smooth_window, smooth_polyorder, return_plot_data=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # ─── TOP PANEL: Raw spectrum with regions ───
    
    y_plot = d_data['y_raw']
    if not np.array_equal(d_data['y_raw'], g_data['y_raw']):
        raise ValueError(f"Y raw data of bands does not match!")

    ax1.plot(x, y_plot, 'k-', linewidth=1.2, label='Spectrum')
    
    # Highlight D-band region
    d_mask = (x >= d_range[0]) & (x <= d_range[1])
    if d_mask.any():
        ax1.fill_between(x[d_mask], 0, y_plot[d_mask], alpha=0.3, color='blue', label='D-band region')
    
    # Highlight G-band region
    g_mask = (x >= g_range[0]) & (x <= g_range[1])
    if g_mask.any():
        ax1.fill_between(x[g_mask], 0, y_plot[g_mask], alpha=0.3, color='red', label='G-band region')
    
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
    ax1.set_ylabel('Intensity (counts)', fontsize=10)
    ax1.set_title('Raw Spectrum with Integration Regions', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ─── BOTTOM PANEL: Baseline-corrected regions ───
    # Plot D-band using data from auc_inward
    ax2.plot(d_data['xs'], d_data['ys_raw'], 'b-', linewidth=1, alpha=0.5, label='D-band raw')
    if baseline != "none":
        ax2.plot(d_data['xs'], d_data['baseline'], 'b--', linewidth=1, alpha=0.7, label='D baseline')
    ax2.fill_between(d_data['xs'], 0, d_data['ys_corrected'], alpha=0.4, color='blue', label='D AUC')
    
    # Plot G-band using data from auc_inward
    ax2.plot(g_data['xs'], g_data['ys_raw'], 'r-', linewidth=1, alpha=0.5, label='G-band raw')
    if baseline != "none":
        ax2.plot(g_data['xs'], g_data['baseline'], 'r--', linewidth=1, alpha=0.7, label='G baseline')
    ax2.fill_between(g_data['xs'], 0, g_data['ys_corrected'], alpha=0.4, color='red', label='G AUC')
    
    # Use calculated AUCs (guaranteed to match!)
    ratio = d_data['auc'] / g_data['auc']
    
    ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
    ax2.set_ylabel('Intensity (baseline-corrected)', fontsize=10)
    ax2.set_title(f'Baseline-Corrected Bands  |  D/G Ratio = {ratio:.4f}', fontsize=11)
    ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    return fig

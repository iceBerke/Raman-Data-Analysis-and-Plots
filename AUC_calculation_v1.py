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
#        "none" or "linear" (linear connects endpoints)
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

# AUTHOR: Berke Santos
# Developed with the help of Claude.AI 
# Last updated: 03/02/2026

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

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
    print(i0)
    i1 = np.searchsorted(x, b, side="right") - 1
    print(i1)

    if i0 >= len(x) or i1 < 0 or i0 > i1:
        raise ValueError("No points remain after snapping bounds inward.")
    
    return i0, i1


def auc_inward(x, y, a, b, baseline, smooth, smooth_window, smooth_polyorder):
    
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

    # Baseline correction
    if baseline == "linear":
        x0, x1 = xs[0], xs[-1]
        y0, y1 = ys[0], ys[-1]
        base = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        ys = ys - base
    
    return float(np.trapezoid(ys, xs)), (xs[0], xs[-1])


# ============= COMPLETE WORKFLOW =============

def analyze_raman_batch(folder, d_range, g_range, baseline, smooth, 
                        smooth_window, smooth_polyorder, skiprows):
    
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
                smooth=smooth, 
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder
            )
            
            # Calculate G-band AUC
            g_auc, g_bounds = auc_inward(
                x, y, *g_range, 
                baseline=baseline, 
                smooth=smooth, 
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder
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


# ============= EXAMPLE USAGE =============

if __name__ == "__main__":
    
    # Example 1: Load single file
    # x, y = load_raman_data(r"C:\Users\berke.santos\Documents\CEMHTI\RAMAN\Rhynie\non-irr\RT\Rhynie-RT-LS-01-CM 125000 to 250000-Average-SubBkg-poly4.txt")
    # print(f"Loaded {len(x)} points from {x[0]:.1f} to {x[-1]:.1f} cm⁻¹")

    d_auc, d_bounds = auc_inward(x, y, 1100, 1500, baseline="linear", smooth=False, smooth_window=11, smooth_polyorder=3)
    g_auc, g_bounds = auc_inward(x, y, 1500, 1750, baseline="linear", smooth=False, smooth_window=11, smooth_polyorder=3)
    ratio = d_auc / g_auc
    
    print(f"\nD-band: AUC = {d_auc:.2f}, range = {d_bounds}")
    print(f"G-band: AUC = {g_auc:.2f}, range = {g_bounds}")
    print(f"AUC_D/AUC_G = {ratio:.3f}")
    
    # Example 2: Batch process entire folder
    # results = analyze_raman_batch(
    #    folder=r"",
    #    d_range=(1300, 1400),
    #    g_range=(1550, 1650),
    #    baseline='linear',
    #    smooth=True,          # Use smoothing if data is noisy
    #    smooth_window=11,
    #    smooth_polyorder=3,
    #    skiprows=0            # Set to 1 if files have header row
    #)
    
    # Display results
    #print("\n" + "="*60)
    #print(results)
    #results.to_csv("raman_results.csv")
    
    # Summary statistics
    #print("\nSummary Statistics:")
    #print(f"  Mean AUC_D/AUC_G: {results['AUC(D)/AUC(G)'].mean():.3f} ± {results['AUC(D)/AUC(G)'].std():.3f}")

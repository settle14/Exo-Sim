# tools/muscle_activation_analysis.py
# -*- coding: utf-8 -*-
"""
Muscle activation analysis with EMG-style comparison.
- Read muscle activation time series from CSV files
- Compare muscle group activations between exo ON vs OFF conditions
- Provide EMG-style analysis with time-series plots and statistical comparisons
- Extract r_m and work_proxy parameters for assistance effect analysis

Changes from previous power-focused version:
- Focus on muscle activation patterns (EMG-like analysis)
- Compare muscle groups rather than total power
- Add r_m and work_proxy analysis for assistance effectiveness
- Remove force and power analysis to focus on activation patterns
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# ---------- IO helpers ----------

def load_meta(d: str) -> Dict:
    p = os.path.join(d, "meta.json")
    with open(p, "r") as f:
        return json.load(f)

def read_time_series(csv_path: str, meta: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Returns (t, df). If CSV has a 'time' column, use it.
    Otherwise build time axis from meta['dt'] and row count.
    """
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        t = df["time"].to_numpy(dtype=float)
        df = df.drop(columns=["time"])
    else:
        dt = float(meta.get("dt", 1/60.0))
        t = np.arange(len(df)) * dt
    return t, df

def read_scalar_time_series(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read scalar time series (like r_m or work_proxy) with time column
    Returns (t, values)
    """
    if not os.path.exists(csv_path):
        return np.array([]), np.array([])
    
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        t = df["time"].to_numpy(dtype=float)
        # Get the data column (should be the second column)
        data_col = [col for col in df.columns if col != "time"][0]
        values = df[data_col].to_numpy(dtype=float)
    else:
        # Fallback if no time column
        t = np.arange(len(df)) * (1/60.0)  # Assume 60 FPS
        values = df.iloc[:, 0].to_numpy(dtype=float)
    return t, values

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = min(win, len(x))
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")

# ---------- Muscle grouping ----------

def group_muscles_by_function(muscle_names: List[str]) -> Dict[str, List[str]]:
    """
    Group muscles by functional categories for EMG-like analysis
    """
    groups = {
        'hip_flexors': [],
        'hip_extensors': [],
        'hip_abductors': [],
        'hip_adductors': [],
        'knee_extensors': [],
        'knee_flexors': [],
        'ankle_plantarflexors': [],
        'ankle_dorsiflexors': [],
        'other': []
    }
    
    for muscle in muscle_names:
        muscle_lower = muscle.lower()
        
        # Hip flexors
        if any(x in muscle_lower for x in ['psoas', 'iliacus', 'rectus_femoris', 'recfem', 'tfl', 'sart']):
            groups['hip_flexors'].append(muscle)
        # Hip extensors  
        elif any(x in muscle_lower for x in ['glmax', 'glut', 'semiten', 'semimem', 'bflh', 'bfsh']):
            groups['hip_extensors'].append(muscle)
        # Hip abductors
        elif any(x in muscle_lower for x in ['glmed', 'glmin', 'tfl']):
            groups['hip_abductors'].append(muscle)
        # Hip adductors
        elif any(x in muscle_lower for x in ['addbrev', 'addlong', 'addmag', 'grac', 'pect']):
            groups['hip_adductors'].append(muscle)
        # Knee extensors
        elif any(x in muscle_lower for x in ['vasmed', 'vaslat', 'vasint', 'recfem']):
            groups['knee_extensors'].append(muscle)
        # Knee flexors
        elif any(x in muscle_lower for x in ['semiten', 'semimem', 'bflh', 'bfsh', 'gaslat', 'gasmed']):
            groups['knee_flexors'].append(muscle)
        # Ankle plantarflexors
        elif any(x in muscle_lower for x in ['gaslat', 'gasmed', 'soleus', 'tibpost']):
            groups['ankle_plantarflexors'].append(muscle)
        # Ankle dorsiflexors
        elif any(x in muscle_lower for x in ['tibant']):
            groups['ankle_dorsiflexors'].append(muscle)
        else:
            groups['other'].append(muscle)
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}
    return groups

# ---------- alignment ----------

def intersection_grid(t1: np.ndarray, y1: np.ndarray,
                      t2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align by intersection of time ranges and resample both to a common grid.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    t0 = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    if t_end <= t0:
        raise ValueError("No temporal overlap between the two series.")

    dt1 = np.median(np.diff(t1)) if len(t1) > 1 else np.inf
    dt2 = np.median(np.diff(t2)) if len(t2) > 1 else np.inf
    dt = max(dt1, dt2)

    n = int(np.floor((t_end - t0) / dt)) + 1
    t_common = t0 + np.arange(n) * dt

    m1 = (t1 >= t0) & (t1 <= t_end)
    m2 = (t2 >= t0) & (t2 <= t_end)
    t1_cut, y1_cut = t1[m1], y1[m1]
    t2_cut, y2_cut = t2[m2], y2[m2]

    y1i = np.interp(t_common, t1_cut, y1_cut)
    y2i = np.interp(t_common, t2_cut, y2_cut)
    return t_common, y1i, y2i

def firstN_align(t1: np.ndarray, y1: np.ndarray,
                 t2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate both to the same number of samples using the later start.
    """
    t0  = max(t1.min(), t2.min())
    dt1 = np.median(np.diff(t1)) if len(t1) > 1 else np.inf
    dt2 = np.median(np.diff(t2)) if len(t2) > 1 else np.inf
    dt  = max(dt1, dt2)

    m1 = t1 >= t0
    m2 = t2 >= t0
    t1s, y1s = t1[m1], y1[m1]
    t2s, y2s = t2[m2], y2[m2]
    n = min(len(t1s), len(t2s))
    if n <= 1:
        raise ValueError("Not enough overlap after truncation.")
    t_common = t0 + np.arange(n) * dt
    y1i = np.interp(t_common, t1s, y1s)
    y2i = np.interp(t_common, t2s, y2s)
    return t_common, y1i, y2i

# ---------- metrics ----------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def integrate_trapz(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.trapz(y, t))

def pct_change(a: float, b: float) -> float:
    base = max(1e-8, abs(a))
    return float((b - a) / base * 100.0)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo_dir",   required=True, help="Directory with exo ON data")
    ap.add_argument("--human_dir", required=True, help="Directory with exo OFF data")
    ap.add_argument("--out_dir",   required=True, help="Output directory for analysis")
    ap.add_argument("--trim_start", type=float, default=0.0, help="seconds to cut from head")
    ap.add_argument("--trim_end",   type=float, default=0.0, help="seconds to cut from tail")
    ap.add_argument("--smooth_win", type=int,   default=5,   help="moving-average window for plotting")
    ap.add_argument("--align_mode", type=str,   default="intersection",
                    choices=["intersection", "firstN"],
                    help="alignment strategy for ON/OFF time series")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load meta
    meta_off = load_meta(args.human_dir)
    meta_on  = load_meta(args.exo_dir)

    # ---- read muscle activation series
    t_off_act, act_off_df = read_time_series(os.path.join(args.human_dir, "muscle_activations.csv"), meta_off)
    t_on_act,  act_on_df  = read_time_series(os.path.join(args.exo_dir,   "muscle_activations.csv"), meta_on)

    # ---- read r_m and work_proxy if available
    t_off_rm, rm_off = read_scalar_time_series(os.path.join(args.human_dir, "r_m_timeseries.csv"))
    t_on_rm,  rm_on  = read_scalar_time_series(os.path.join(args.exo_dir,   "r_m_timeseries.csv"))
    
    t_off_wp, wp_off = read_scalar_time_series(os.path.join(args.human_dir, "work_proxy_timeseries.csv"))
    t_on_wp,  wp_on  = read_scalar_time_series(os.path.join(args.exo_dir,   "work_proxy_timeseries.csv"))

    # ---- trim by time (per series)
    def trim_by_time(t: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        t0 = t.min() + args.trim_start
        t1 = t.max() - args.trim_end
        if t1 <= t0:
            t1 = t0 + 1e-8
        m = (t >= t0) & (t <= t1)
        return t[m], df.iloc[m]

    def trim_scalar_by_time(t: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(t) == 0 or len(values) == 0:
            return t, values
        t0 = t.min() + args.trim_start
        t1 = t.max() - args.trim_end
        if t1 <= t0:
            t1 = t0 + 1e-8
        m = (t >= t0) & (t <= t1)
        return t[m], values[m]

    t_off_act, act_off_df = trim_by_time(t_off_act, act_off_df)
    t_on_act,  act_on_df  = trim_by_time(t_on_act,  act_on_df)
    
    t_off_rm, rm_off = trim_scalar_by_time(t_off_rm, rm_off)
    t_on_rm, rm_on = trim_scalar_by_time(t_on_rm, rm_on)
    
    t_off_wp, wp_off = trim_scalar_by_time(t_off_wp, wp_off)
    t_on_wp, wp_on = trim_scalar_by_time(t_on_wp, wp_on)

    print(f"[INFO] OFF activation: len={len(t_off_act)}, range=[{t_off_act.min():.3f},{t_off_act.max():.3f}]")
    print(f"[INFO]  ON activation: len={len(t_on_act)}, range=[{t_on_act.min():.3f},{t_on_act.max():.3f}]")

    # ---- Group muscles by function
    muscle_names = list(act_off_df.columns)
    muscle_groups = group_muscles_by_function(muscle_names)
    
    print(f"[INFO] Muscle groups identified:")
    for group, muscles in muscle_groups.items():
        print(f"  {group}: {muscles}")

    # ---- Align activation data and compute group averages
    if args.align_mode == "intersection":
        # Align individual muscle activations
        aligned_data = {}
        t_common = None
        for muscle in muscle_names:
            if muscle in act_off_df.columns and muscle in act_on_df.columns:
                t_c, act_off_m, act_on_m = intersection_grid(
                    t_off_act, act_off_df[muscle].to_numpy(),
                    t_on_act, act_on_df[muscle].to_numpy()
                )
                aligned_data[muscle] = (act_off_m, act_on_m)
                if t_common is None:
                    t_common = t_c
    else:
        # firstN alignment
        aligned_data = {}
        t_common = None
        for muscle in muscle_names:
            if muscle in act_off_df.columns and muscle in act_on_df.columns:
                t_c, act_off_m, act_on_m = firstN_align(
                    t_off_act, act_off_df[muscle].to_numpy(),
                    t_on_act, act_on_df[muscle].to_numpy()
                )
                aligned_data[muscle] = (act_off_m, act_on_m)
                if t_common is None:
                    t_common = t_c

    # ---- Compute muscle group activations
    group_data = {}
    for group_name, muscle_list in muscle_groups.items():
        off_group = []
        on_group = []
        
        for muscle in muscle_list:
            if muscle in aligned_data:
                off_group.append(aligned_data[muscle][0])
                on_group.append(aligned_data[muscle][1])
        
        if off_group and on_group:
            # Average across muscles in the group
            group_off = np.mean(off_group, axis=0)
            group_on = np.mean(on_group, axis=0)
            group_data[group_name] = (group_off, group_on)

    # ---- Align r_m and work_proxy data
    rm_aligned = False
    wp_aligned = False
    
    if len(rm_off) > 0 and len(rm_on) > 0:
        try:
            if args.align_mode == "intersection":
                _, rm_off_aligned, rm_on_aligned = intersection_grid(t_off_rm, rm_off, t_on_rm, rm_on)
            else:
                _, rm_off_aligned, rm_on_aligned = firstN_align(t_off_rm, rm_off, t_on_rm, rm_on)
            rm_aligned = True
        except:
            print("[WARN] Could not align r_m data")
            rm_off_aligned = rm_on_aligned = np.array([])
    
    if len(wp_off) > 0 and len(wp_on) > 0:
        try:
            if args.align_mode == "intersection":
                _, wp_off_aligned, wp_on_aligned = intersection_grid(t_off_wp, wp_off, t_on_wp, wp_on)
            else:
                _, wp_off_aligned, wp_on_aligned = firstN_align(t_off_wp, wp_off, t_on_wp, wp_on)
            wp_aligned = True
        except:
            print("[WARN] Could not align work_proxy data")
            wp_off_aligned = wp_on_aligned = np.array([])

    print(f"[INFO] aligned data: len={len(t_common)}, range=[{t_common.min():.3f},{t_common.max():.3f}]")

    # ---- Compute group-level metrics
    group_results = []
    
    for group_name, (group_off, group_on) in group_data.items():
        # Apply smoothing for cleaner analysis
        group_off_s = moving_average(group_off, args.smooth_win)
        group_on_s = moving_average(group_on, args.smooth_win)
        
        # Compute metrics
        mean_off = float(np.mean(group_off_s))
        mean_on = float(np.mean(group_on_s))
        rms_off = rms(group_off_s)
        rms_on = rms(group_on_s)
        integral_off = integrate_trapz(group_off_s, t_common)
        integral_on = integrate_trapz(group_on_s, t_common)
        
        group_results.append({
            'group': group_name,
            'mean_off': mean_off,
            'mean_on': mean_on,
            'mean_change': pct_change(mean_off, mean_on),
            'rms_off': rms_off,
            'rms_on': rms_on,
            'rms_change': pct_change(rms_off, rms_on),
            'integral_off': integral_off,
            'integral_on': integral_on,
            'integral_change': pct_change(integral_off, integral_on)
        })

    # ---- Overall metrics
    overall_results = []
    
    # Overall muscle activation (mean across all muscles)
    all_off = np.mean([data[0] for data in aligned_data.values()], axis=0)
    all_on = np.mean([data[1] for data in aligned_data.values()], axis=0)
    all_off_s = moving_average(all_off, args.smooth_win)
    all_on_s = moving_average(all_on, args.smooth_win)
    
    overall_results.append({
        'metric': 'Overall Mean Activation',
        'off': float(np.mean(all_off_s)),
        'on': float(np.mean(all_on_s)),
        'change': pct_change(np.mean(all_off_s), np.mean(all_on_s))
    })
    
    overall_results.append({
        'metric': 'Overall RMS Activation', 
        'off': rms(all_off_s),
        'on': rms(all_on_s),
        'change': pct_change(rms(all_off_s), rms(all_on_s))
    })

    # r_m and work_proxy metrics
    if rm_aligned:
        rm_off_mean = float(np.mean(rm_off_aligned))
        rm_on_mean = float(np.mean(rm_on_aligned))
        overall_results.append({
            'metric': 'r_m (Mean Muscle Activation)',
            'off': rm_off_mean,
            'on': rm_on_mean,
            'change': pct_change(rm_off_mean, rm_on_mean)
        })
    
    if wp_aligned:
        wp_off_mean = float(np.mean(wp_off_aligned))
        wp_on_mean = float(np.mean(wp_on_aligned))
        overall_results.append({
            'metric': 'Work Proxy (Sum Sq. Activation)',
            'off': wp_off_mean,
            'on': wp_on_mean,
            'change': pct_change(wp_off_mean, wp_on_mean)
        })

    # ---- Print results
    print("\n=== Muscle Group Analysis (exo ON vs OFF) — negative Δ% indicates assistance ===")
    group_df = pd.DataFrame(group_results)
    if not group_df.empty:
        print(group_df[['group', 'mean_change', 'rms_change', 'integral_change']].to_string(index=False, float_format='%.2f'))

    print("\n=== Overall Metrics ===")
    overall_df = pd.DataFrame(overall_results)
    print(overall_df.to_string(index=False, float_format='%.6f'))

    # ---- Plotting with enhanced professional style ----
    
    # Set up publication-ready style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    colors = {
        'off': '#2E86AB',    # Professional blue
        'on': '#F24236',     # Professional red
        'neutral': '#A23B72',
        'positive': '#F18F01',
        'negative': '#C73E1D'
    }
    
    # 1. Muscle group time series - 2x5 grid layout
    n_groups = len(group_data)
    if n_groups > 0:
        # Calculate optimal grid layout
        if n_groups <= 4:
            rows, cols = 2, 2
        elif n_groups <= 6:
            rows, cols = 2, 3
        elif n_groups <= 8:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 4
            
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten() if n_groups > 1 else [axes]
        
        for i, (group_name, (group_off, group_on)) in enumerate(group_data.items()):
            if i >= len(axes):
                break
                
            group_off_s = moving_average(group_off, args.smooth_win)
            group_on_s = moving_average(group_on, args.smooth_win)
            
            axes[i].plot(t_common, group_off_s, color=colors['off'], 
                        linewidth=2, label='Exo OFF', alpha=0.8)
            axes[i].plot(t_common, group_on_s, color=colors['on'], 
                        linewidth=2, label='Exo ON', alpha=0.8)
            
            axes[i].set_ylabel('Activation (AU)', fontweight='bold')
            axes[i].set_title(f'{group_name.replace("_", " ").title()}', 
                            fontweight='bold', fontsize=11)
            
            # Add shaded region for major differences
            diff = group_off_s - group_on_s
            axes[i].fill_between(t_common, group_off_s, group_on_s, 
                               where=(diff > 0), alpha=0.2, color=colors['off'],
                               interpolate=True)
            axes[i].fill_between(t_common, group_off_s, group_on_s, 
                               where=(diff < 0), alpha=0.2, color=colors['on'],
                               interpolate=True)
            
            if i == 0:  # Only show legend on first subplot
                axes[i].legend(loc='upper right', fontsize=9)
            
            axes[i].set_xlim(t_common.min(), t_common.max())
        
        # Hide unused subplots
        for i in range(len(group_data), len(axes)):
            axes[i].set_visible(False)
        
        # Set x-label only for bottom row
        for i in range(len(axes)):
            if i >= len(axes) - cols or i >= len(group_data) - 1:
                axes[i].set_xlabel('Time (s)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "muscle_group_timeseries.png"), 
                   facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(args.out_dir, "muscle_group_timeseries.pdf"), 
                   facecolor='white', edgecolor='none')
        plt.close()

    # 2. Overall activation comparison - Enhanced style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left panel: Time series comparison
    ax1.plot(t_common, all_off_s, color=colors['off'], linewidth=2.5, 
             label="Exo OFF", alpha=0.9)
    ax1.plot(t_common, all_on_s, color=colors['on'], linewidth=2.5, 
             label="Exo ON", alpha=0.9)
    
    # Add confidence bands (using standard error)
    all_off_se = np.std([data[0] for data in aligned_data.values()], axis=0) / np.sqrt(len(aligned_data))
    all_on_se = np.std([data[1] for data in aligned_data.values()], axis=0) / np.sqrt(len(aligned_data))
    all_off_se_s = moving_average(all_off_se, args.smooth_win)
    all_on_se_s = moving_average(all_on_se, args.smooth_win)
    
    ax1.fill_between(t_common, all_off_s - all_off_se_s, all_off_s + all_off_se_s, 
                     color=colors['off'], alpha=0.2)
    ax1.fill_between(t_common, all_on_s - all_on_se_s, all_on_s + all_on_se_s, 
                     color=colors['on'], alpha=0.2)
    
    ax1.set_xlabel("Time (s)", fontweight='bold')
    ax1.set_ylabel("Mean Activation (AU)", fontweight='bold')
    ax1.set_title("A. Overall Muscle Activation Comparison", fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(t_common.min(), t_common.max())
    
    # Right panel: Difference plot
    diff = all_off_s - all_on_s
    ax2.plot(t_common, diff, color=colors['neutral'], linewidth=2, alpha=0.8)
    ax2.fill_between(t_common, diff, 0, where=(diff > 0), 
                     color=colors['off'], alpha=0.3, label='Exo reduces activation')
    ax2.fill_between(t_common, diff, 0, where=(diff < 0), 
                     color=colors['on'], alpha=0.3, label='Exo increases activation')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Time (s)", fontweight='bold')
    ax2.set_ylabel("Activation Difference (AU)", fontweight='bold')
    ax2.set_title("B. Activation Difference (OFF - ON)", fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(t_common.min(), t_common.max())
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "overall_activation_comparison.png"), 
               facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(args.out_dir, "overall_activation_comparison.pdf"), 
               facecolor='white', edgecolor='none')
    plt.close()

    # 3. r_m and work_proxy time series - Enhanced dual panel
    if rm_aligned and wp_aligned:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # r_m time series
        ax1.plot(t_common, rm_off_aligned, color=colors['off'], linewidth=2.5, 
                label="r_m OFF", alpha=0.9)
        ax1.plot(t_common, rm_on_aligned, color=colors['on'], linewidth=2.5, 
                label="r_m ON", alpha=0.9)
        ax1.set_ylabel("r_m (AU)", fontweight='bold')
        ax1.set_title("A. Mean Muscle Activation (r_m)", fontweight='bold', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_xlim(t_common.min(), t_common.max())
        
        # r_m difference
        rm_diff = rm_off_aligned - rm_on_aligned
        ax2.plot(t_common, rm_diff, color=colors['neutral'], linewidth=2)
        ax2.fill_between(t_common, rm_diff, 0, where=(rm_diff > 0), 
                        color=colors['positive'], alpha=0.3)
        ax2.fill_between(t_common, rm_diff, 0, where=(rm_diff < 0), 
                        color=colors['negative'], alpha=0.3)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel("r_m Difference", fontweight='bold')
        ax2.set_title("B. r_m Difference (OFF - ON)", fontweight='bold', fontsize=12)
        ax2.set_xlim(t_common.min(), t_common.max())
        
        # work_proxy time series
        ax3.plot(t_common, wp_off_aligned, color=colors['off'], linewidth=2.5, 
                label="Work Proxy OFF", alpha=0.9)
        ax3.plot(t_common, wp_on_aligned, color=colors['on'], linewidth=2.5, 
                label="Work Proxy ON", alpha=0.9)
        ax3.set_xlabel("Time (s)", fontweight='bold')
        ax3.set_ylabel("Work Proxy (AU²)", fontweight='bold')
        ax3.set_title("C. Sum of Squared Activations", fontweight='bold', fontsize=12)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.set_xlim(t_common.min(), t_common.max())
        
        # work_proxy difference
        wp_diff = wp_off_aligned - wp_on_aligned
        ax4.plot(t_common, wp_diff, color=colors['neutral'], linewidth=2)
        ax4.fill_between(t_common, wp_diff, 0, where=(wp_diff > 0), 
                        color=colors['positive'], alpha=0.3)
        ax4.fill_between(t_common, wp_diff, 0, where=(wp_diff < 0), 
                        color=colors['negative'], alpha=0.3)
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel("Time (s)", fontweight='bold')
        ax4.set_ylabel("Work Proxy Difference", fontweight='bold')
        ax4.set_title("D. Work Proxy Difference (OFF - ON)", fontweight='bold', fontsize=12)
        ax4.set_xlim(t_common.min(), t_common.max())
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "assistance_metrics_analysis.png"), 
                   facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(args.out_dir, "assistance_metrics_analysis.pdf"), 
                   facecolor='white', edgecolor='none')
        plt.close()

    # 4. Enhanced bar charts with statistical information
    if not group_df.empty:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        groups = group_df['group'].tolist()
        group_labels = [g.replace('_', ' ').title() for g in groups]
        mean_changes = group_df['mean_change'].tolist()
        rms_changes = group_df['rms_change'].tolist()
        integral_changes = group_df['integral_change'].tolist()
        
        # Color mapping based on change direction
        def get_bar_colors(changes):
            return [colors['positive'] if x > 5 else colors['negative'] if x < -5 
                   else colors['neutral'] for x in changes]
        
        # Mean activation changes
        bars1 = ax1.bar(range(len(groups)), mean_changes, 
                       color=get_bar_colors(mean_changes), alpha=0.8, edgecolor='black', linewidth=0.8)
        ax1.axhline(0, color='black', linewidth=1.2)
        ax1.set_ylabel('Change in Mean Activation (%)', fontweight='bold')
        ax1.set_title('A. Mean Activation Changes by Muscle Group', fontweight='bold', fontsize=12)
        ax1.set_xticks(range(len(groups)))
        ax1.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar, val in zip(bars1, mean_changes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=8)
        
        # RMS activation changes
        bars2 = ax2.bar(range(len(groups)), rms_changes, 
                       color=get_bar_colors(rms_changes), alpha=0.8, edgecolor='black', linewidth=0.8)
        ax2.axhline(0, color='black', linewidth=1.2)
        ax2.set_ylabel('Change in RMS Activation (%)', fontweight='bold')
        ax2.set_title('B. RMS Activation Changes by Muscle Group', fontweight='bold', fontsize=12)
        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=9)
        
        for bar, val in zip(bars2, rms_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=8)
        
        # Integral changes
        bars3 = ax3.bar(range(len(groups)), integral_changes, 
                       color=get_bar_colors(integral_changes), alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.axhline(0, color='black', linewidth=1.2)
        ax3.set_ylabel('Change in Integrated Activation (%)', fontweight='bold')
        ax3.set_title('C. Integrated Activation Changes by Muscle Group', fontweight='bold', fontsize=12)
        ax3.set_xticks(range(len(groups)))
        ax3.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=9)
        
        for bar, val in zip(bars3, integral_changes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=8)
        
        # Summary radar chart in the fourth subplot
        ax4.remove()  # Remove the fourth subplot
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        mean_changes_norm = [(x + 25) / 50 for x in mean_changes]  # Normalize to 0-1
        mean_changes_norm += mean_changes_norm[:1]
        
        ax4.plot(angles, mean_changes_norm, 'o-', linewidth=2, color=colors['neutral'])
        ax4.fill(angles, mean_changes_norm, alpha=0.25, color=colors['neutral'])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels([g.replace('_', ' ').title() for g in groups], fontsize=8)
        ax4.set_ylim(0, 1)
        ax4.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax4.set_yticklabels(['-15%', '-5%', '5%', '15%'], fontsize=7)
        ax4.set_title('D. Mean Activation Change Overview', fontweight='bold', fontsize=12, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "muscle_group_analysis_enhanced.png"), 
                   facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(args.out_dir, "muscle_group_analysis_enhanced.pdf"), 
                   facecolor='white', edgecolor='none')
        plt.close()

    # 5. Overall metrics summary with enhanced visualization
    if not overall_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left panel: Bar chart with enhanced styling
        metrics = overall_df['metric'].tolist()
        changes = overall_df['change'].tolist()
        metric_labels = [m.replace('(', '\n(') for m in metrics]  # Break long labels
        
        bars = ax1.bar(range(len(metrics)), changes, 
                      color=[colors['negative'] if x < 0 else colors['positive'] for x in changes], 
                      alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)
        
        ax1.axhline(0, color='black', linewidth=1.5)
        ax1.set_ylabel('Percentage Change (%)', fontweight='bold', fontsize=12)
        ax1.set_title('A. Overall Assistance Effectiveness\n(Negative = Better Assistance)', 
                     fontweight='bold', fontsize=12)
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metric_labels, fontsize=9, ha='center')
        
        # Add value labels on bars
        for bar, val in zip(bars, changes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                    f'{val:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=10)
        
        # Right panel: Assistance effectiveness gauge
        # Create a semi-circular gauge chart
        overall_improvement = np.mean([abs(x) for x in changes if x < 0])  # Mean of negative changes
        
        # Gauge parameters
        gauge_angles = np.linspace(np.pi, 0, 100)
        gauge_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
        
        for i in range(len(gauge_angles)-1):
            ax2.fill_between([gauge_angles[i], gauge_angles[i+1]], [0.8, 0.8], [1.0, 1.0], 
                           color=gauge_colors[i], alpha=0.8)
        
        # Add needle
        needle_angle = np.pi - (overall_improvement / 3) * np.pi  # Scale to 0-3% range
        needle_angle = max(0.1, min(np.pi - 0.1, needle_angle))
        
        ax2.plot([needle_angle, needle_angle], [0, 0.9], 'k-', linewidth=4)
        ax2.plot(needle_angle, 0.9, 'ko', markersize=8)
        
        # Gauge labels
        ax2.text(np.pi, -0.2, 'No Effect\n0%', ha='center', va='top', fontweight='bold')
        ax2.text(np.pi/2, 0.6, f'Current\n{overall_improvement:.2f}%', ha='center', va='center', 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax2.text(0, -0.2, 'Strong Effect\n3%', ha='center', va='top', fontweight='bold')
        
        ax2.set_xlim(-0.2, np.pi + 0.2)
        ax2.set_ylim(-0.3, 1.1)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('B. Overall Assistance Level', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "overall_metrics_enhanced.png"), 
                   facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(args.out_dir, "overall_metrics_enhanced.pdf"), 
                   facecolor='white', edgecolor='none')
        plt.close()

    # ---- Save CSV files ----
    if not group_df.empty:
        group_df.to_csv(os.path.join(args.out_dir, "muscle_group_analysis.csv"), index=False)
    
    overall_df.to_csv(os.path.join(args.out_dir, "overall_analysis.csv"), index=False)

    print(f"\n[OK] Enhanced analysis saved to {args.out_dir}")
    print("CSV Files:")
    print(" - muscle_group_analysis.csv")
    print(" - overall_analysis.csv")
    print("Enhanced Visualization Files (PNG + PDF):")
    print(" - muscle_group_timeseries.png/pdf (2x3 grid layout)")
    print(" - overall_activation_comparison.png/pdf (dual panel with confidence bands)")
    if rm_aligned and wp_aligned:
        print(" - assistance_metrics_analysis.png/pdf (2x2 analysis)")
    if not group_df.empty:
        print(" - muscle_group_analysis_enhanced.png/pdf (comprehensive analysis with radar chart)")
    print(" - overall_metrics_enhanced.png/pdf (bar chart + effectiveness gauge)")
    print("\nAll figures are publication-ready with 300 DPI resolution!")

if __name__ == "__main__":
    main()

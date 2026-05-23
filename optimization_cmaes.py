"""
optimization_cmaes.py - CMA-ES Optimization for 11-Element Yagi-Uda
====================================================================
Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
Optimizes 18 parameters at 300 MHz:
  - 9 director lengths  (d1_len ~ d9_len)
  - 9 director spacings (d1_sp  ~ d9_sp)

CMA-ES vs PSO:
  - CMA-ES learns the full covariance structure of the search space,
    automatically discovering parameter correlations (e.g. shorter directors
    need wider spacing). PSO treats each dimension independently.
  - CMA-ES is quasi-parameter-free: only sigma0 (initial step size) needs
    tuning. Population size, learning rates, etc. are derived from N_DIM.
  - CMA-ES is the gold standard for continuous black-box optimization
    in 10-100 dimensions.

Algorithm outline (Hansen & Ostermeier, 2001):
  1. Sample lambda offspring: x_k = mean + sigma * N(0, C)
  2. Evaluate fitness of each offspring
  3. Select mu best, recombine into new mean
  4. Update evolution paths (p_sigma, p_c)
  5. Adapt covariance matrix C via rank-1 and rank-mu updates
  6. Adapt step size sigma via cumulative step-size adaptation (CSA)

Usage:  python optimization_cmaes.py
Output: Q3_optimized_cmaes.nec, convergence plot, parameter summary
"""

import numpy as np
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== Paths ========================
ENGINE   = Path(r"C:\4nec2\exe\nec2dxs500.exe")
OUT_DIR  = Path(r"C:\4nec2\out")
SAVE_DIR = Path(r"E:\communication")

# ======================== Fixed Antenna Parameters ========================
FREQ        = 300.0
Z0          = 50.0
SEGS        = 11
CENTER_SEG  = 6
WIRE_RAD    = 0.004
DRIVEN_HALF = 0.24
REF_LEN     = 0.48
REF_X       = -0.2

# ======================== CMA-ES Configuration ========================
N_DIM       = 18
MAX_ITER    = 150
SIGMA0      = 0.15           # initial step size (fraction of search range)
N_WORKERS   = 8

# Bounds
LB = np.array([0.28]*9 + [0.08]*9)
UB = np.array([0.46]*9 + [0.45]*9)


# ======================== NEC2 Engine Interface ========================
# (identical to optimization.py)

def build_inp(params: np.ndarray) -> str:
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM Yagi 11el CMA-ES eval",
        "CE",
        f"GW 1 {SEGS} 0 0 {-DRIVEN_HALF} 0 0 {DRIVEN_HALF} {WIRE_RAD}",
        f"GW 2 {SEGS} {REF_X} 0 {-REF_LEN/2} {REF_X} 0 {REF_LEN/2} {WIRE_RAD}",
    ]
    x = 0.0
    for i in range(9):
        x += d_sps[i]
        h = d_lens[i] / 2
        lines.append(
            f"GW {i+3} {SEGS} {x:.6f} 0 {-h:.6f} {x:.6f} 0 {h:.6f} {WIRE_RAD}"
        )
    lines += [
        "GE 0", "GN -1", "EK",
        f"EX 0 1 {CENTER_SEG} 0 1 0",
        f"FR 0 1 0 0 {FREQ} 0",
        "RP 0 19 1 0 0 0 10 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def evaluate_one(args: tuple) -> tuple:
    idx, params = args
    stem = f"cma_{idx}"
    inp_path = OUT_DIR / f"{stem}.inp"
    out_path = OUT_DIR / f"{stem}.out"

    inp_path.write_text(build_inp(params), encoding="ascii", errors="replace")
    if out_path.exists():
        out_path.unlink()

    try:
        subprocess.run(
            [str(ENGINE)],
            input=f"{stem}.inp\n{stem}.out\n",
            capture_output=True, text=True, timeout=30,
            cwd=str(OUT_DIR),
        )
    except Exception:
        return -999.0, 0.0, 0.0

    if not out_path.exists():
        return -999.0, 0.0, 0.0

    content = out_path.read_text(encoding="utf-8", errors="replace")

    sec = re.search(
        r"ANTENNA INPUT PARAMETERS.*?(\d+\s+\d+\s+.*?E[+-]\d+.*)",
        content, re.DOTALL,
    )
    if not sec:
        return -999.0, 0.0, 0.0
    nums = re.findall(r"[+-]?\d+\.\d+E[+-]\d+", sec.group(1).split("\n")[0])
    if len(nums) < 6:
        return -999.0, 0.0, 0.0

    zr, zi = float(nums[4]), float(nums[5])
    gamma_abs = abs((complex(zr, zi) - Z0) / (complex(zr, zi) + Z0))
    s11 = 20 * np.log10(max(gamma_abs, 1e-10))

    gains = {}
    if "RADIATION PATTERNS" in content:
        for line in content[content.rindex("RADIATION PATTERNS"):].splitlines():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    theta = float(parts[0])
                    gain  = float(parts[4])
                    if 0 <= theta <= 180 and -100 < gain < 50:
                        gains[theta] = gain
                except ValueError:
                    pass

    if not gains:
        return -999.0, 0.0, 0.0

    forward_gain = gains.get(90.0, max(gains.values()))
    fitness = forward_gain
    if s11 > -10:
        fitness -= 5.0 * (s11 + 10)

    return fitness, forward_gain, s11


# ======================== CMA-ES Algorithm ========================

def run_cmaes():
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    All internal constants follow Hansen's canonical naming and recommended
    default values derived from dimensionality N = N_DIM.

    State variables:
      mean   (N,)     distribution mean
      sigma  scalar   global step size
      C      (N,N)    covariance matrix
      p_s    (N,)     conjugate evolution path (for sigma adaptation)
      p_c    (N,)     evolution path (for rank-1 C update)

    Per generation:
      1. Sample lambda offspring from N(mean, sigma^2 * C)
      2. Clamp to bounds
      3. Evaluate fitness in parallel
      4. Sort by fitness (descending, we maximize)
      5. Recombine top mu into new mean
      6. Update p_s, p_c, C, sigma
    """
    N = N_DIM

    # --- Strategy parameters (all derived from N, per Hansen 2016) ---
    lam   = 4 + int(3 * np.log(N))           # population size (offspring)
    mu    = lam // 2                          # number of parents for recombination

    # Recombination weights (log-linear, normalized)
    raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw_w / raw_w.sum()             # sum = 1
    mu_eff = 1.0 / np.sum(weights ** 2)       # variance effective selection mass

    # Step-size adaptation (CSA)
    c_s   = (mu_eff + 2) / (N + mu_eff + 5)
    d_s   = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + c_s
    E_chi = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))  # E[||N(0,I)||]

    # Covariance matrix adaptation
    c_c   = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
    c_1   = 2 / ((N + 1.3)**2 + mu_eff)
    c_mu  = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)**2 + mu_eff))
    h_threshold = (1.4 + 2 / (N + 1)) * E_chi

    # --- Initialize state ---
    rng   = np.random.default_rng(42)
    mean  = (LB + UB) / 2                     # start at center of bounds
    sigma = SIGMA0
    C     = np.eye(N)                         # covariance = identity
    p_s   = np.zeros(N)                       # sigma evolution path
    p_c   = np.zeros(N)                       # covariance evolution path

    # Eigendecomposition cache (updated lazily)
    eigen_C = np.eye(N)                       # C = eigen_C @ diag(eigen_D^2) @ eigen_C.T
    eigen_D = np.ones(N)
    need_eigen = True

    best_ever_fit   = -np.inf
    best_ever_x     = mean.copy()
    best_ever_gain  = 0.0
    best_ever_s11   = 0.0

    hist_best  = []
    hist_mean  = []
    hist_sigma = []

    print(f"{'='*70}")
    print(f"  CMA-ES Optimization: 11-Element Yagi-Uda at {FREQ:.0f} MHz")
    print(f"  N={N}, lambda={lam}, mu={mu}, mu_eff={mu_eff:.1f}")
    print(f"  sigma0={SIGMA0}, max_iter={MAX_ITER}")
    print(f"  Parallel workers: {N_WORKERS}")
    print(f"{'='*70}")

    t0 = time.time()

    for gen in range(MAX_ITER):

        # --- 1. Eigendecomposition of C (for sampling) ---
        if need_eigen:
            C = (C + C.T) / 2                 # enforce symmetry
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            eigen_D = np.sqrt(eigvals)         # D_i = sqrt(eigenvalue_i)
            eigen_C = eigvecs                  # columns = eigenvectors
            need_eigen = False
        invsqrtC = eigen_C @ np.diag(1.0 / eigen_D) @ eigen_C.T

        # --- 2. Sample lambda offspring ---
        z_all = rng.standard_normal((lam, N))  # N(0, I)
        y_all = z_all * eigen_D                # scaled by D
        y_all = (eigen_C @ y_all.T).T          # rotated by eigenvectors -> N(0, C)
        x_all = mean + sigma * y_all           # x_k = mean + sigma * y_k

        # Clamp to bounds
        x_all = np.clip(x_all, LB, UB)

        # --- 3. Evaluate fitness in parallel ---
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            results = list(pool.map(
                evaluate_one,
                [(i, x_all[i]) for i in range(lam)],
            ))

        fits  = np.array([r[0] for r in results])
        gains = np.array([r[1] for r in results])
        s11s  = np.array([r[2] for r in results])

        # --- 4. Sort by fitness (descending = best first) ---
        order = np.argsort(-fits)
        x_sorted = x_all[order]
        y_sorted = (x_sorted - mean) / sigma   # recover y for update

        # Track best ever
        if fits[order[0]] > best_ever_fit:
            best_ever_fit  = fits[order[0]]
            best_ever_x    = x_sorted[0].copy()
            best_ever_gain = gains[order[0]]
            best_ever_s11  = s11s[order[0]]

        # --- 5. Recombine: weighted mean of top mu ---
        mean_old = mean.copy()
        mean = weights @ x_sorted[:mu]          # new mean
        y_w  = (mean - mean_old) / sigma        # weighted mean step in y-space

        # --- 6. Update evolution path p_s (CSA) ---
        p_s = ((1 - c_s) * p_s
               + np.sqrt(c_s * (2 - c_s) * mu_eff) * (invsqrtC @ y_w))

        # --- 7. Heaviside function for stalling detection ---
        h_sig = 1.0 if np.linalg.norm(p_s) / np.sqrt(
            1 - (1 - c_s) ** (2 * (gen + 1))
        ) < h_threshold else 0.0

        # --- 8. Update evolution path p_c ---
        p_c = ((1 - c_c) * p_c
               + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w)

        # --- 9. Adapt covariance matrix C ---
        # Rank-1 update
        rank1 = np.outer(p_c, p_c)
        # Rank-mu update
        rank_mu = np.zeros((N, N))
        for i in range(mu):
            rank_mu += weights[i] * np.outer(y_sorted[i], y_sorted[i])

        delta_h = (1 - h_sig) * c_c * (2 - c_c)
        C = ((1 - c_1 - c_mu + delta_h * c_1) * C
             + c_1 * rank1
             + c_mu * rank_mu)

        need_eigen = True

        # --- 10. Adapt step size sigma ---
        sigma *= np.exp((c_s / d_s) * (np.linalg.norm(p_s) / E_chi - 1))
        sigma = min(sigma, 1.0)                # cap to prevent explosion

        # --- Logging ---
        valid = fits[fits > -900]
        avg = np.mean(valid) if len(valid) > 0 else -999
        elapsed = time.time() - t0

        hist_best.append(best_ever_fit)
        hist_mean.append(avg)
        hist_sigma.append(sigma)

        print(
            f"  [{gen+1:3d}/{MAX_ITER}]  "
            f"best={best_ever_fit:7.3f}  gain={best_ever_gain:.2f} dBi  "
            f"S11={best_ever_s11:.1f} dB  "
            f"avg={avg:7.3f}  sigma={sigma:.4f}  {elapsed:.0f}s"
        )

        # Early stop if sigma collapses
        if sigma < 1e-8:
            print("  sigma collapsed, stopping early.")
            break

    total = time.time() - t0
    print(f"\n  Completed in {total:.1f}s ({total/60:.1f} min)")

    return best_ever_x, best_ever_fit, best_ever_gain, best_ever_s11, \
           hist_best, hist_mean, hist_sigma


# ======================== Output Helpers ========================

def save_optimized_nec(params: np.ndarray, path: Path, tag: str = "CMA-ES"):
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM --------------------------------------------------------",
        f"CM Task 3: 11-Element Yagi-Uda ({tag} Optimized)",
        "CM 1 Driven + 1 Reflector + 9 Directors",
        f"CM Optimized for {FREQ:.0f} MHz by {tag}",
        "CM --------------------------------------------------------",
        "CE",
        "",
        "' --- Reflector (fixed) ---",
        f"SY ref_len={REF_LEN}",
        f"SY ref_space={REF_X}",
        "SY ref_z1=-ref_len/2",
        "SY ref_z2=ref_len/2",
        "",
        f"' --- Director lengths ({tag} optimized) ---",
    ]
    for i in range(9):
        lines.append(f"SY d{i+1}_len={d_lens[i]:.6f}")
    lines += ["", f"' --- Director relative spacings ({tag} optimized) ---"]
    for i in range(9):
        lines.append(f"SY d{i+1}_sp={d_sps[i]:.6f}")
    lines += ["", "' --- Cumulative X positions ---", "SY d1_x=d1_sp"]
    for i in range(1, 9):
        lines.append(f"SY d{i+1}_x=d{i}_x+d{i+1}_sp")
    lines += ["", "' --- Z symmetry coordinates ---"]
    for i in range(9):
        lines.append(f"SY d{i+1}_z1=-d{i+1}_len/2")
        lines.append(f"SY d{i+1}_z2=d{i+1}_len/2")
    lines += [
        "", "' --- Geometry (11 wires) ---",
        f"GW 1 {SEGS} 0 0 -0.24 0 0 0.24 {WIRE_RAD}",
        f"GW 2 {SEGS} ref_space 0 ref_z1 ref_space 0 ref_z2 {WIRE_RAD}",
    ]
    for i in range(9):
        lines.append(
            f"GW {i+3} {SEGS} d{i+1}_x 0 d{i+1}_z1 d{i+1}_x 0 d{i+1}_z2 {WIRE_RAD}"
        )
    lines += [
        "", "' --- Environment ---",
        "GE 0", "GN -1", "EK",
        f"EX 0 1 {CENTER_SEG} 0 1 0",
        f"FR 0 1 0 0 {FREQ:.0f} 0",
        "EN",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_verification(params: np.ndarray):
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM Yagi 11el CMA-ES final verification",
        "CE",
        f"GW 1 {SEGS} 0 0 {-DRIVEN_HALF} 0 0 {DRIVEN_HALF} {WIRE_RAD}",
        f"GW 2 {SEGS} {REF_X} 0 {-REF_LEN/2} {REF_X} 0 {REF_LEN/2} {WIRE_RAD}",
    ]
    x = 0.0
    for i in range(9):
        x += d_sps[i]
        h = d_lens[i] / 2
        lines.append(
            f"GW {i+3} {SEGS} {x:.6f} 0 {-h:.6f} {x:.6f} 0 {h:.6f} {WIRE_RAD}"
        )
    lines += [
        "GE 0", "GN -1", "EK",
        f"EX 0 1 {CENTER_SEG} 0 1 0",
        f"FR 0 1 0 0 {FREQ} 0",
        "RP 0 181 361 0 0 0 1 1",
        "EN",
    ]
    stem = "cma_final"
    inp_path = OUT_DIR / f"{stem}.inp"
    out_path = OUT_DIR / f"{stem}.out"
    inp_path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    if out_path.exists():
        out_path.unlink()

    subprocess.run(
        [str(ENGINE)],
        input=f"{stem}.inp\n{stem}.out\n",
        capture_output=True, text=True, timeout=120,
        cwd=str(OUT_DIR),
    )
    content = out_path.read_text(encoding="utf-8", errors="replace")

    sec = re.search(
        r"ANTENNA INPUT PARAMETERS.*?(\d+\s+\d+\s+.*?E[+-]\d+.*)",
        content, re.DOTALL,
    )
    zr = zi = 0.0
    if sec:
        nums = re.findall(r"[+-]?\d+\.\d+E[+-]\d+", sec.group(1).split("\n")[0])
        if len(nums) >= 6:
            zr, zi = float(nums[4]), float(nums[5])

    gamma_abs = abs((complex(zr, zi) - Z0) / (complex(zr, zi) + Z0))
    s11  = 20 * np.log10(max(gamma_abs, 1e-10))
    vswr = (1 + gamma_abs) / (1 - gamma_abs) if gamma_abs < 1 else 999

    max_gain = fwd_gain = bwd_gain = -999.0
    if "RADIATION PATTERNS" in content:
        for line in content[content.rindex("RADIATION PATTERNS"):].splitlines():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    theta = float(parts[0])
                    phi   = float(parts[1])
                    gain  = float(parts[4])
                    if gain > max_gain:
                        max_gain = gain
                    if abs(theta - 90) < 0.5 and abs(phi) < 0.5:
                        fwd_gain = gain
                    if abs(theta - 90) < 0.5 and abs(phi - 180) < 0.5:
                        bwd_gain = gain
                except ValueError:
                    pass

    fb = fwd_gain - bwd_gain if bwd_gain > -900 else 0
    return {
        "Z_real": zr, "Z_imag": zi, "S11": s11, "VSWR": vswr,
        "max_gain": max_gain, "fwd_gain": fwd_gain,
        "bwd_gain": bwd_gain, "F/B": fb,
    }


def plot_convergence(hist_best, hist_mean, hist_sigma, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    iters = range(1, len(hist_best) + 1)
    ax1.plot(iters, hist_best, "b-",  lw=2, label="Best Ever")
    ax1.plot(iters, hist_mean, "r--", lw=1, alpha=0.6, label="Generation Mean")
    ax1.set_ylabel("Fitness (dBi)", fontsize=13)
    ax1.set_title("CMA-ES Convergence - 11-Element Yagi-Uda",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, hist_sigma, "g-", lw=2)
    ax2.set_xlabel("Generation", fontsize=13)
    ax2.set_ylabel("Step Size (sigma)", fontsize=13)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close()


# ======================== Main ========================

if __name__ == "__main__":
    best_x, best_fit, best_gain, best_s11, h_best, h_mean, h_sigma = run_cmaes()

    d_lens = best_x[:9]
    d_sps  = best_x[9:]

    print("\nRunning final verification (full 3D pattern) ...")
    metrics = final_verification(best_x)

    print("\n" + "=" * 70)
    print("  CMA-ES OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"  Forward Gain : {metrics['fwd_gain']:.2f} dBi")
    print(f"  Max Gain     : {metrics['max_gain']:.2f} dBi")
    print(f"  F/B Ratio    : {metrics['F/B']:.2f} dB")
    print(f"  Impedance    : {metrics['Z_real']:.2f} + j({metrics['Z_imag']:.2f}) ohm")
    print(f"  S11          : {metrics['S11']:.2f} dB")
    print(f"  VSWR         : {metrics['VSWR']:.3f}")
    print()
    print("  Director Lengths (m):")
    for i in range(9):
        print(f"    d{i+1}_len = {d_lens[i]:.6f}  ({d_lens[i]*1000:.1f} mm)")
    print()
    print("  Director Spacings (m) and Cumulative X Positions:")
    x = 0.0
    for i in range(9):
        x += d_sps[i]
        print(f"    d{i+1}_sp = {d_sps[i]:.6f}  ->  x = {x:.4f} m")
    print(f"\n  Total boom length: {abs(REF_X) + x:.4f} m "
          f"({(abs(REF_X) + x) / (299.7925/FREQ):.2f} lambda)")

    nec_path = SAVE_DIR / "Q3_optimized_cmaes.nec"
    save_optimized_nec(best_x, nec_path, "CMA-ES")
    print(f"\n  Optimized .nec : {nec_path}")

    conv_path = SAVE_DIR / "fig_cmaes_convergence.png"
    plot_convergence(h_best, h_mean, h_sigma, conv_path)
    print(f"  Convergence    : {conv_path}")

    param_path = SAVE_DIR / "cmaes_best_params.txt"
    with open(param_path, "w") as f:
        f.write(f"# CMA-ES Best Parameters (fitness={best_fit:.4f})\n")
        f.write(f"# Forward Gain={metrics['fwd_gain']:.2f} dBi, "
                f"S11={metrics['S11']:.2f} dB, F/B={metrics['F/B']:.2f} dB\n")
        for i in range(9):
            f.write(f"d{i+1}_len={d_lens[i]:.6f}\n")
        for i in range(9):
            f.write(f"d{i+1}_sp={d_sps[i]:.6f}\n")
    print(f"  Parameters     : {param_path}")
    print("=" * 70)

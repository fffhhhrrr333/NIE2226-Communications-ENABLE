"""
optimization_hybrid.py - Hybrid DE -> CMA-ES for 11-Element Yagi-Uda
=====================================================================
Two-phase optimization at 300 MHz:

  Phase 1: Differential Evolution (DE/rand/1/bin)
    - Global search, explores multiple basins
    - Large population, high mutation factor
    - Finds the neighborhood of the global optimum

  Phase 2: CMA-ES (seeded from DE best)
    - Local refinement with covariance learning
    - Learns parameter correlations for fine-tuning
    - Polishes the solution to high precision

Why hybrid:
  DE alone: great at finding the right basin, slow at fine-tuning
  CMA-ES alone: fast convergence, but can miss the global basin
  Combined: DE handles multimodality, CMA-ES handles precision

Optimizes 18 parameters:
  - 9 director lengths  (d1_len ~ d9_len)
  - 9 director spacings (d1_sp  ~ d9_sp)

Usage:  python optimization_hybrid.py
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

# ======================== Antenna Constants ========================
FREQ        = 300.0
Z0          = 50.0
SEGS        = 11
CENTER_SEG  = 6
WIRE_RAD    = 0.004
DRIVEN_HALF = 0.24
REF_LEN     = 0.48
REF_X       = -0.2

# ======================== Search Space ========================
N_DIM = 18
LB = np.array([0.28]*9 + [0.08]*9)
UB = np.array([0.46]*9 + [0.45]*9)

# ======================== DE Configuration ========================
DE_POP       = 60            # population size (3~5x per dim is robust)
DE_ITER      = 80            # generations
DE_F         = 0.7           # mutation scale factor
DE_CR        = 0.9           # crossover probability
DE_STRATEGY  = "rand/1/bin"  # classic strategy

# ======================== CMA-ES Configuration ========================
CMA_ITER     = 100           # max generations for refinement
CMA_SIGMA0   = 0.03          # small initial sigma (already near optimum)

# ======================== Common ========================
N_WORKERS    = 8


# ======================== NEC2 Interface ========================

def build_inp(params: np.ndarray) -> str:
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM Yagi 11el hybrid eval",
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
    """Thread-safe NEC2 evaluation. Returns (fitness, gain, s11)."""
    idx, params, prefix = args
    stem = f"{prefix}_{idx}"
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


def batch_evaluate(pop: np.ndarray, prefix: str) -> list:
    """Evaluate entire population in parallel."""
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(
            evaluate_one,
            [(i, pop[i], prefix) for i in range(len(pop))],
        ))


# ======================== Phase 1: Differential Evolution ========================

def run_de():
    """
    DE/rand/1/bin — the most widely used DE variant.

    For each target vector x_i:
      1. Pick 3 distinct random vectors x_r1, x_r2, x_r3 (r1 != r2 != r3 != i)
      2. Mutant:  v = x_r1 + F * (x_r2 - x_r3)
      3. Crossover (binomial): for each dim j,
         trial_j = v_j  if rand < CR or j == j_rand
                   x_i_j  otherwise
      4. Selection: keep trial if fitness(trial) >= fitness(target)
    """
    rng = np.random.default_rng(42)

    # Initialize population uniformly in bounds
    pop = rng.uniform(LB, UB, (DE_POP, N_DIM))

    # Evaluate initial population
    results = batch_evaluate(pop, "de")
    fits  = np.array([r[0] for r in results])
    gains = np.array([r[1] for r in results])
    s11s  = np.array([r[2] for r in results])

    best_idx = np.argmax(fits)
    best_x   = pop[best_idx].copy()
    best_fit = fits[best_idx]
    best_gain = gains[best_idx]
    best_s11  = s11s[best_idx]

    hist_best = [best_fit]
    hist_avg  = [np.mean(fits[fits > -900]) if np.any(fits > -900) else -999]

    print(f"\n{'='*70}")
    print(f"  Phase 1: Differential Evolution (DE/rand/1/bin)")
    print(f"  pop={DE_POP}, F={DE_F}, CR={DE_CR}, iter={DE_ITER}")
    print(f"{'='*70}")
    print(f"  [init]  best={best_fit:7.3f}  gain={best_gain:.2f} dBi  S11={best_s11:.1f} dB")

    t0 = time.time()

    for gen in range(DE_ITER):
        # Generate trial vectors for all individuals
        trials = np.empty_like(pop)

        for i in range(DE_POP):
            # Pick 3 distinct random indices != i
            candidates = list(range(DE_POP))
            candidates.remove(i)
            r1, r2, r3 = rng.choice(candidates, 3, replace=False)

            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = pop[r1] + DE_F * (pop[r2] - pop[r3])

            # Binomial crossover
            j_rand = rng.integers(N_DIM)
            mask = rng.random(N_DIM) < DE_CR
            mask[j_rand] = True
            trial = np.where(mask, v, pop[i])

            # Clamp to bounds
            trials[i] = np.clip(trial, LB, UB)

        # Evaluate all trials in parallel
        trial_results = batch_evaluate(trials, "de")
        trial_fits  = np.array([r[0] for r in trial_results])
        trial_gains = np.array([r[1] for r in trial_results])
        trial_s11s  = np.array([r[2] for r in trial_results])

        # Greedy selection: keep trial if it's at least as good
        for i in range(DE_POP):
            if trial_fits[i] >= fits[i]:
                pop[i]   = trials[i]
                fits[i]  = trial_fits[i]
                gains[i] = trial_gains[i]
                s11s[i]  = trial_s11s[i]

        # Update global best
        gen_best = np.argmax(fits)
        if fits[gen_best] > best_fit:
            best_fit  = fits[gen_best]
            best_x    = pop[gen_best].copy()
            best_gain = gains[gen_best]
            best_s11  = s11s[gen_best]

        valid = fits[fits > -900]
        avg = np.mean(valid) if len(valid) > 0 else -999
        hist_best.append(best_fit)
        hist_avg.append(avg)

        elapsed = time.time() - t0
        print(
            f"  [{gen+1:3d}/{DE_ITER}]  "
            f"best={best_fit:7.3f}  gain={best_gain:.2f} dBi  "
            f"S11={best_s11:.1f} dB  avg={avg:7.3f}  {elapsed:.0f}s"
        )

    de_time = time.time() - t0
    de_evals = DE_POP + DE_POP * DE_ITER

    # Collect top-5 distinct individuals for CMA-ES seeding
    top_indices = np.argsort(-fits)[:5]
    top5 = pop[top_indices]

    print(f"\n  DE completed: {de_time:.1f}s, {de_evals} evals")
    print(f"  Best: gain={best_gain:.2f} dBi, S11={best_s11:.1f} dB")

    return best_x, best_fit, best_gain, best_s11, top5, hist_best, hist_avg


# ======================== Phase 2: CMA-ES Refinement ========================

def run_cmaes(seed_x: np.ndarray, seed_top5: np.ndarray):
    """
    CMA-ES seeded from DE best solution.
    Initial mean = DE best, small sigma for local refinement.
    Initial covariance seeded from top-5 DE spread for better starting shape.
    """
    N = N_DIM

    # Strategy parameters (Hansen defaults)
    lam   = 4 + int(3 * np.log(N))
    mu    = lam // 2
    raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw_w / raw_w.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)

    c_s   = (mu_eff + 2) / (N + mu_eff + 5)
    d_s   = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + c_s
    E_chi = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))

    c_c   = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
    c_1   = 2 / ((N + 1.3)**2 + mu_eff)
    c_mu  = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)**2 + mu_eff))
    h_threshold = (1.4 + 2 / (N + 1)) * E_chi

    rng = np.random.default_rng(123)

    # Seed from DE: mean = best, C from top-5 spread
    mean  = seed_x.copy()
    sigma = CMA_SIGMA0

    C = np.eye(N)

    p_s = np.zeros(N)
    p_c = np.zeros(N)

    best_fit  = -np.inf
    best_x    = mean.copy()
    best_gain = 0.0
    best_s11  = 0.0

    hist_best  = []
    hist_sigma = []

    print(f"\n{'='*70}")
    print(f"  Phase 2: CMA-ES Refinement")
    print(f"  lambda={lam}, mu={mu}, sigma0={CMA_SIGMA0}")
    print(f"  Seeded from DE best, C=I, sigma capped at 0.1")
    print(f"{'='*70}")

    t0 = time.time()

    for gen in range(CMA_ITER):
        # Eigendecomposition
        C = (C + C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        D = np.sqrt(eigvals)
        B = eigvecs
        invsqrtC = B @ np.diag(1.0 / D) @ B.T

        # Sample offspring
        z_all = rng.standard_normal((lam, N))
        y_all = (B @ (z_all * D).T).T
        x_all = mean + sigma * y_all
        x_all = np.clip(x_all, LB, UB)

        # Evaluate
        results = batch_evaluate(x_all, "cma2")
        fits_arr  = np.array([r[0] for r in results])
        gains_arr = np.array([r[1] for r in results])
        s11s_arr  = np.array([r[2] for r in results])

        # Sort descending
        order = np.argsort(-fits_arr)
        x_sorted = x_all[order]
        y_sorted = (x_sorted - mean) / sigma

        if fits_arr[order[0]] > best_fit:
            best_fit  = fits_arr[order[0]]
            best_x    = x_sorted[0].copy()
            best_gain = gains_arr[order[0]]
            best_s11  = s11s_arr[order[0]]

        # Recombine
        mean_old = mean.copy()
        mean = weights @ x_sorted[:mu]
        y_w  = (mean - mean_old) / sigma

        # Update p_s
        p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * mu_eff) * (invsqrtC @ y_w)

        h_sig = 1.0 if np.linalg.norm(p_s) / np.sqrt(
            1 - (1 - c_s) ** (2 * (gen + 1))
        ) < h_threshold else 0.0

        # Update p_c
        p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

        # Update C
        rank1 = np.outer(p_c, p_c)
        rank_mu = sum(weights[i] * np.outer(y_sorted[i], y_sorted[i]) for i in range(mu))
        delta_h = (1 - h_sig) * c_c * (2 - c_c)
        C = (1 - c_1 - c_mu + delta_h * c_1) * C + c_1 * rank1 + c_mu * rank_mu

        # Update sigma
        sigma *= np.exp((c_s / d_s) * (np.linalg.norm(p_s) / E_chi - 1))
        sigma = min(sigma, 0.1)

        hist_best.append(best_fit)
        hist_sigma.append(sigma)

        elapsed = time.time() - t0
        valid = fits_arr[fits_arr > -900]
        avg = np.mean(valid) if len(valid) > 0 else -999
        print(
            f"  [{gen+1:3d}/{CMA_ITER}]  "
            f"best={best_fit:7.3f}  gain={best_gain:.2f} dBi  "
            f"S11={best_s11:.1f} dB  avg={avg:7.3f}  "
            f"sigma={sigma:.5f}  {elapsed:.0f}s"
        )

        if sigma < 1e-8:
            print("  sigma collapsed, converged.")
            break

    cma_time = time.time() - t0
    print(f"\n  CMA-ES completed: {cma_time:.1f}s")

    return best_x, best_fit, best_gain, best_s11, hist_best, hist_sigma


# ======================== Output Helpers ========================

def save_optimized_nec(params: np.ndarray, path: Path):
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM --------------------------------------------------------",
        "CM Task 3: 11-Element Yagi-Uda (Hybrid DE+CMA-ES Optimized)",
        "CM 1 Driven + 1 Reflector + 9 Directors",
        f"CM Optimized for {FREQ:.0f} MHz",
        "CM --------------------------------------------------------",
        "CE",
        "",
        "' --- Reflector (fixed) ---",
        f"SY ref_len={REF_LEN}",
        f"SY ref_space={REF_X}",
        "SY ref_z1=-ref_len/2",
        "SY ref_z2=ref_len/2",
        "",
        "' --- Director lengths (Hybrid DE+CMA-ES optimized) ---",
    ]
    for i in range(9):
        lines.append(f"SY d{i+1}_len={d_lens[i]:.6f}")
    lines += ["", "' --- Director relative spacings ---"]
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
        "CM Yagi 11el hybrid final", "CE",
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
    stem = "hybrid_final"
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
                    theta, phi, gain = float(parts[0]), float(parts[1]), float(parts[4])
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


def plot_convergence(de_best, de_avg, cma_best, cma_sigma, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: fitness
    ax = axes[0]
    n_de = len(de_best)
    n_cma = len(cma_best)
    total = n_de + n_cma

    ax.plot(range(1, n_de+1), de_best, "b-", lw=2, label="DE Best")
    ax.plot(range(1, n_de+1), de_avg,  "b--", lw=1, alpha=0.4, label="DE Avg")
    ax.plot(range(n_de+1, total+1), cma_best, "r-", lw=2, label="CMA-ES Best")
    ax.axvline(n_de, color="gray", ls=":", lw=2, alpha=0.7)
    ax.text(n_de, ax.get_ylim()[0] if ax.get_ylim()[0] > -50 else -10,
            "  DE -> CMA-ES", fontsize=11, color="gray", va="bottom")
    ax.set_ylabel("Fitness (dBi)", fontsize=13)
    ax.set_title("Hybrid DE + CMA-ES Convergence - 11-Element Yagi-Uda",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Bottom: sigma (CMA-ES phase only)
    ax2 = axes[1]
    ax2.plot(range(n_de+1, total+1), cma_sigma, "g-", lw=2)
    ax2.set_xlabel("Generation (DE + CMA-ES)", fontsize=13)
    ax2.set_ylabel("CMA-ES sigma", fontsize=13)
    ax2.set_yscale("log")
    ax2.axvline(n_de, color="gray", ls=":", lw=2, alpha=0.7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close()


# ======================== Main ========================

if __name__ == "__main__":
    t_total = time.time()

    # Phase 1: DE global search
    de_x, de_fit, de_gain, de_s11, de_top5, de_hist_best, de_hist_avg = run_de()

    # Phase 2: CMA-ES local refinement
    cma_x, cma_fit, cma_gain, cma_s11, cma_hist_best, cma_hist_sigma = \
        run_cmaes(de_x, de_top5)

    # Pick overall best
    if cma_fit >= de_fit:
        best_x, best_gain, best_s11 = cma_x, cma_gain, cma_s11
        winner = "CMA-ES"
    else:
        best_x, best_gain, best_s11 = de_x, de_gain, de_s11
        winner = "DE"

    # Final verification
    print("\nRunning final verification (full 3D pattern) ...")
    metrics = final_verification(best_x)

    total_time = time.time() - t_total
    total_evals = DE_POP + DE_POP * DE_ITER + (4 + int(3*np.log(N_DIM))) * len(cma_hist_best)

    d_lens = best_x[:9]
    d_sps  = best_x[9:]

    print(f"\n{'='*70}")
    print(f"  HYBRID DE + CMA-ES RESULTS  (best from {winner})")
    print(f"{'='*70}")
    print(f"  Forward Gain : {metrics['fwd_gain']:.2f} dBi")
    print(f"  Max Gain     : {metrics['max_gain']:.2f} dBi")
    print(f"  F/B Ratio    : {metrics['F/B']:.2f} dB")
    print(f"  Impedance    : {metrics['Z_real']:.2f} + j({metrics['Z_imag']:.2f}) ohm")
    print(f"  S11          : {metrics['S11']:.2f} dB")
    print(f"  VSWR         : {metrics['VSWR']:.3f}")
    print(f"  Total evals  : ~{total_evals}")
    print(f"  Total time   : {total_time:.1f}s ({total_time/60:.1f} min)")
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
    boom = abs(REF_X) + x
    print(f"\n  Total boom: {boom:.4f} m ({boom / (299.7925/FREQ):.2f} lambda)")

    # Save
    nec_path = SAVE_DIR / "Q3_optimized_hybrid.nec"
    save_optimized_nec(best_x, nec_path)
    print(f"\n  Optimized .nec : {nec_path}")

    conv_path = SAVE_DIR / "fig_hybrid_convergence.png"
    plot_convergence(de_hist_best, de_hist_avg, cma_hist_best, cma_hist_sigma, conv_path)
    print(f"  Convergence    : {conv_path}")

    param_path = SAVE_DIR / "hybrid_best_params.txt"
    with open(param_path, "w") as f:
        f.write(f"# Hybrid DE+CMA-ES Best (winner={winner})\n")
        f.write(f"# Gain={metrics['fwd_gain']:.2f} dBi, "
                f"S11={metrics['S11']:.2f} dB, F/B={metrics['F/B']:.2f} dB\n")
        for i in range(9):
            f.write(f"d{i+1}_len={d_lens[i]:.6f}\n")
        for i in range(9):
            f.write(f"d{i+1}_sp={d_sps[i]:.6f}\n")
    print(f"  Parameters     : {param_path}")
    print("=" * 70)

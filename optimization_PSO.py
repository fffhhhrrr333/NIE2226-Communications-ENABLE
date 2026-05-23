"""
optimization.py - Particle Swarm Optimization for 11-Element Yagi-Uda
=====================================================================
Optimizes 18 parameters at 300 MHz:
  - 9 director lengths  (d1_len ~ d9_len)
  - 9 director spacings (d1_sp  ~ d9_sp)

Fixed elements:
  - Driven element: L=0.48m, center at origin, along Z-axis
  - Reflector:      L=0.48m, x=-0.2m

Objective: maximize forward gain while keeping S11 < -10 dB (50 ohm)

Usage:  python optimization.py
Output: Q3_optimized.nec, convergence plot, parameter summary
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
FREQ        = 300.0          # MHz
Z0          = 50.0           # reference impedance
SEGS        = 11
CENTER_SEG  = 6
WIRE_RAD    = 0.004          # m
DRIVEN_HALF = 0.24           # driven element half-length (m)
REF_LEN     = 0.48           # reflector total length (m)
REF_X       = -0.2           # reflector X position (m)

# ======================== PSO Configuration ========================
N_DIM        = 18            # 9 lengths + 9 spacings
N_PARTICLES  = 40
N_ITERATIONS = 100
W_START      = 0.9           # inertia weight (linearly decays)
W_END        = 0.4
C1           = 2.0           # cognitive coefficient (personal best)
C2           = 2.0           # social coefficient (global best)
N_WORKERS    = 8             # parallel NEC2 engine instances

# Parameter bounds: [d1_len..d9_len, d1_sp..d9_sp]
#   director length:  0.28 ~ 0.46 m  (shorter than driven 0.48m)
#   director spacing: 0.08 ~ 0.45 m  (relative gap to previous element)
LB = np.array([0.28]*9 + [0.08]*9)
UB = np.array([0.46]*9 + [0.45]*9)
V_MAX = 0.3 * (UB - LB)     # max velocity per dimension


# ======================== NEC2 Engine Interface ========================

def build_inp(params: np.ndarray) -> str:
    """
    From 18 PSO parameters, generate a complete .inp file string.

    params[0:9]  = director lengths  (d1_len .. d9_len)
    params[9:18] = director spacings (d1_sp  .. d9_sp)

    The .inp is pure NEC2 format (no SY variables).
    Wire layout:
      Tag 1:  driven element at x=0   (fixed)
      Tag 2:  reflector at x=-0.2     (fixed)
      Tag 3~11: directors at cumulative x positions
    """
    d_lens = params[:9]
    d_sps  = params[9:]

    lines = [
        "CM Yagi 11el PSO eval",
        "CE",
        # Driven element (fixed)
        f"GW 1 {SEGS} 0 0 {-DRIVEN_HALF} 0 0 {DRIVEN_HALF} {WIRE_RAD}",
        # Reflector (fixed)
        f"GW 2 {SEGS} {REF_X} 0 {-REF_LEN/2} {REF_X} 0 {REF_LEN/2} {WIRE_RAD}",
    ]

    # Directors: cumulative X position
    x = 0.0
    for i in range(9):
        x += d_sps[i]
        h = d_lens[i] / 2
        lines.append(
            f"GW {i+3} {SEGS} {x:.6f} 0 {-h:.6f} {x:.6f} 0 {h:.6f} {WIRE_RAD}"
        )

    lines += [
        "GE 0",
        "GN -1",
        "EK",
        f"EX 0 1 {CENTER_SEG} 0 1 0",
        f"FR 0 1 0 0 {FREQ} 0",
        # E-plane pattern: theta 0~180 every 10 deg, phi=0 (forward = theta 90)
        "RP 0 19 1 0 0 0 10 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def evaluate_particle(args: tuple) -> tuple:
    """
    Run NEC2 for one particle and return (fitness, forward_gain, s11).

    Each particle writes to its own pso_{idx}.inp/.out to avoid file conflicts
    when running in parallel with ThreadPoolExecutor.
    """
    idx, params = args
    stem = f"pso_{idx}"
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

    # --- Parse impedance ---
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

    # --- Parse radiation pattern ---
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

    # Forward gain: theta=90 is +X direction (boom axis)
    forward_gain = gains.get(90.0, max(gains.values()))

    # --- Fitness ---
    # Maximize gain; heavy penalty if S11 > -10 dB
    fitness = forward_gain
    if s11 > -10:
        fitness -= 5.0 * (s11 + 10)

    return fitness, forward_gain, s11


# ======================== PSO Algorithm ========================

def run_pso():
    """
    Standard PSO with linearly decreasing inertia weight.

    Each iteration:
      1. Evaluate all particles in parallel (ThreadPoolExecutor)
      2. Update personal best (pbest) and global best (gbest)
      3. Update velocity:  v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
      4. Update position:  x = x + v
      5. Clamp position to bounds
    """
    rng = np.random.default_rng(42)

    # Initialize swarm
    pos = rng.uniform(LB, UB, (N_PARTICLES, N_DIM))
    vel = rng.uniform(-V_MAX * 0.5, V_MAX * 0.5, (N_PARTICLES, N_DIM))

    pbest_pos = pos.copy()
    pbest_fit = np.full(N_PARTICLES, -np.inf)

    gbest_pos = pos[0].copy()
    gbest_fit = -np.inf
    gbest_gain = 0.0
    gbest_s11  = 0.0

    history_best = []
    history_avg  = []

    print(f"{'='*70}")
    print(f"  PSO Optimization: 11-Element Yagi-Uda at {FREQ:.0f} MHz")
    print(f"  {N_PARTICLES} particles x {N_ITERATIONS} iterations = "
          f"{N_PARTICLES * N_ITERATIONS} evaluations")
    print(f"  18 params: 9 director lengths + 9 director spacings")
    print(f"  Parallel workers: {N_WORKERS}")
    print(f"{'='*70}")

    t0 = time.time()

    for it in range(N_ITERATIONS):
        # Linearly decay inertia weight
        w = W_START - (W_START - W_END) * it / max(1, N_ITERATIONS - 1)

        # Evaluate all particles in parallel
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            results = list(pool.map(
                evaluate_particle,
                [(i, pos[i]) for i in range(N_PARTICLES)],
            ))

        # Update bests
        iter_fits = []
        for i, (fit, gain, s11) in enumerate(results):
            if fit > -900:
                iter_fits.append(fit)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = pos[i].copy()
            if fit > gbest_fit:
                gbest_fit  = fit
                gbest_pos  = pos[i].copy()
                gbest_gain = gain
                gbest_s11  = s11

        avg_fit = np.mean(iter_fits) if iter_fits else -999
        history_best.append(gbest_fit)
        history_avg.append(avg_fit)

        elapsed = time.time() - t0
        print(
            f"  [{it+1:3d}/{N_ITERATIONS}]  "
            f"best={gbest_fit:7.3f}  gain={gbest_gain:.2f} dBi  "
            f"S11={gbest_s11:.1f} dB  avg={avg_fit:7.3f}  "
            f"w={w:.2f}  {elapsed:.0f}s"
        )

        # Update velocity and position
        r1 = rng.random((N_PARTICLES, N_DIM))
        r2 = rng.random((N_PARTICLES, N_DIM))

        vel = (w * vel
               + C1 * r1 * (pbest_pos - pos)
               + C2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -V_MAX, V_MAX)

        pos = pos + vel
        pos = np.clip(pos, LB, UB)

    total = time.time() - t0
    print(f"\n  Completed in {total:.1f}s ({total/60:.1f} min)")

    return gbest_pos, gbest_fit, gbest_gain, gbest_s11, history_best, history_avg


# ======================== Output Helpers ========================

def save_optimized_nec(params: np.ndarray, path: Path):
    """Write optimized parameters into 4NEC2 .nec format with SY variables."""
    d_lens = params[:9]
    d_sps  = params[9:]

    lines = [
        "CM --------------------------------------------------------",
        "CM Task 3: 11-Element Yagi-Uda (PSO Optimized)",
        "CM 1 Driven + 1 Reflector + 9 Directors",
        f"CM Optimized for {FREQ:.0f} MHz by PSO",
        "CM --------------------------------------------------------",
        "CE",
        "",
        "' --- Reflector (fixed) ---",
        f"SY ref_len={REF_LEN}",
        f"SY ref_space={REF_X}",
        "SY ref_z1=-ref_len/2",
        "SY ref_z2=ref_len/2",
        "",
        "' --- Director lengths (PSO optimized) ---",
    ]
    for i in range(9):
        lines.append(f"SY d{i+1}_len={d_lens[i]:.6f}")

    lines += ["", "' --- Director relative spacings (PSO optimized) ---"]
    for i in range(9):
        lines.append(f"SY d{i+1}_sp={d_sps[i]:.6f}")

    lines += ["", "' --- Cumulative X positions ---"]
    lines.append("SY d1_x=d1_sp")
    for i in range(1, 9):
        lines.append(f"SY d{i+1}_x=d{i}_x+d{i+1}_sp")

    lines += ["", "' --- Z symmetry coordinates ---"]
    for i in range(9):
        lines.append(f"SY d{i+1}_z1=-d{i+1}_len/2")
        lines.append(f"SY d{i+1}_z2=d{i+1}_len/2")

    lines += [
        "",
        "' --- Geometry (11 wires) ---",
        f"GW 1 {SEGS} 0 0 -0.24 0 0 0.24 {WIRE_RAD}",
        f"GW 2 {SEGS} ref_space 0 ref_z1 ref_space 0 ref_z2 {WIRE_RAD}",
    ]
    for i in range(9):
        lines.append(
            f"GW {i+3} {SEGS} d{i+1}_x 0 d{i+1}_z1 d{i+1}_x 0 d{i+1}_z2 {WIRE_RAD}"
        )

    lines += [
        "",
        "' --- Environment ---",
        "GE 0",
        "GN -1",
        "EK",
        f"EX 0 1 {CENTER_SEG} 0 1 0",
        f"FR 0 1 0 0 {FREQ:.0f} 0",
        "EN",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_verification(params: np.ndarray):
    """Run high-resolution simulation on the optimized design."""
    d_lens = params[:9]
    d_sps  = params[9:]

    lines = [
        "CM Yagi 11el PSO final verification",
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

    stem = "pso_final"
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

    # Impedance
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

    # Full pattern
    max_gain = -999.0
    fwd_gain = -999.0
    bwd_gain = -999.0
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

    fb_ratio = fwd_gain - bwd_gain if bwd_gain > -900 else 0

    return {
        "Z_real": zr, "Z_imag": zi,
        "S11": s11, "VSWR": vswr,
        "max_gain": max_gain, "fwd_gain": fwd_gain,
        "bwd_gain": bwd_gain, "F/B": fb_ratio,
    }


def plot_convergence(hist_best, hist_avg, path):
    """Save convergence plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    iters = range(1, len(hist_best) + 1)
    ax.plot(iters, hist_best, "b-",  lw=2, label="Global Best")
    ax.plot(iters, hist_avg,  "r--", lw=1, alpha=0.6, label="Iteration Average")
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Fitness (dBi)", fontsize=13)
    ax.set_title("PSO Convergence - 11-Element Yagi-Uda Optimization",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close()


# ======================== Main ========================

if __name__ == "__main__":
    # --- Run PSO ---
    best_params, best_fit, best_gain, best_s11, hist_best, hist_avg = run_pso()

    d_lens = best_params[:9]
    d_sps  = best_params[9:]

    # --- Final high-res verification ---
    print("\nRunning final verification (full 3D pattern) ...")
    metrics = final_verification(best_params)

    # --- Print results ---
    print("\n" + "=" * 70)
    print("  PSO OPTIMIZATION RESULTS")
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

    # --- Save files ---
    nec_path = SAVE_DIR / "Q3_optimized.nec"
    save_optimized_nec(best_params, nec_path)
    print(f"\n  Optimized .nec : {nec_path}")

    desktop_path = Path(r"C:\Users\Lenovo\Desktop\Q3_optimized.nec")
    save_optimized_nec(best_params, desktop_path)
    print(f"  Desktop copy   : {desktop_path}")

    conv_path = SAVE_DIR / "fig_pso_convergence.png"
    plot_convergence(hist_best, hist_avg, conv_path)
    print(f"  Convergence    : {conv_path}")

    # --- Save params to text ---
    param_path = SAVE_DIR / "pso_best_params.txt"
    with open(param_path, "w") as f:
        f.write(f"# PSO Best Parameters (fitness={best_fit:.4f})\n")
        f.write(f"# Forward Gain={metrics['fwd_gain']:.2f} dBi, "
                f"S11={metrics['S11']:.2f} dB, F/B={metrics['F/B']:.2f} dB\n")
        for i in range(9):
            f.write(f"d{i+1}_len={d_lens[i]:.6f}\n")
        for i in range(9):
            f.write(f"d{i+1}_sp={d_sps[i]:.6f}\n")
    print(f"  Parameters     : {param_path}")
    print("=" * 70)

"""
optimization_bayes.py - Bayesian Optimization for 11-Element Yagi-Uda
=====================================================================
Gaussian Process surrogate model + Expected Improvement acquisition.

Why Bayesian Optimization:
  - Sample efficient: builds a probabilistic model to guide search
  - Each evaluation is chosen to maximize expected improvement
  - Much fewer NEC2 evaluations than population-based methods
  - Naturally balances exploration (uncertain regions) vs exploitation

Algorithm:
  1. Latin Hypercube Sampling for initial space-filling design
  2. Fit GP with isotropic RBF kernel to all observations
  3. Maximize Expected Improvement (EI) to select next batch
  4. Evaluate batch in parallel via NEC2 engine
  5. Repeat until evaluation budget is exhausted

Optimizes 18 parameters:
  - 9 director lengths  (d1_len ~ d9_len)
  - 9 director spacings (d1_sp  ~ d9_sp)

Dependencies:  numpy, matplotlib, scipy
Usage:         python optimization_bayes.py
Output:        Q3_optimized_bayes.nec, convergence plot, parameter summary
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
from scipy.optimize import minimize as sp_minimize
from scipy.special import erf as sp_erf

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

# ======================== BO Configuration ========================
N_INIT      = 100       # Latin Hypercube initial samples (need more in 18D)
N_ITER      = 100       # BO iterations
BATCH       = 8         # parallel acquisitions per iteration
N_CAND      = 10000     # random candidates for acquisition maximization
N_WORKERS   = 8
HP_EVERY    = 10        # re-optimize GP hyperparams every K iterations
MIN_DIST    = 0.06      # diversity radius in [0,1] unit space
UCB_BETA    = 2.5       # exploration weight for UCB acquisition
N_RANDOM    = 2         # pure random exploration slots per batch
LOCAL_STD   = 0.10      # std for local candidate perturbation around best


# ======================== NEC2 Interface ========================

def build_inp(params: np.ndarray) -> str:
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM Yagi 11el bayes eval",
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
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(
            evaluate_one,
            [(i, pop[i], prefix) for i in range(len(pop))],
        ))


# ======================== Gaussian Process ========================

def _norm_cdf(z):
    return 0.5 * (1.0 + sp_erf(z / np.sqrt(2.0)))


def _norm_pdf(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)


class GaussianProcess:
    """
    GP with isotropic RBF kernel.

    All training inputs are in [0,1]^N_DIM (unit-normalized).
    Targets are internally standardized to zero-mean, unit-variance.

    Hyperparameters (optimized via marginal likelihood):
      - lengthscale (l): controls smoothness
      - signal variance (sigma_f^2): controls output scale
      - noise variance (sigma_n^2): observation noise
    """

    def __init__(self):
        self._log_ls    = 0.0     # log(lengthscale)
        self._log_var   = 0.0     # log(signal variance)
        self._log_noise = -4.0    # log(noise variance)
        self.X      = None
        self.y      = None
        self.y_mean = 0.0
        self.y_std  = 1.0
        self.L      = None        # Cholesky of K
        self.alpha  = None        # K^{-1} y

    # ---------- kernel ----------

    def _K(self, X1, X2):
        ls  = np.exp(self._log_ls)
        var = np.exp(self._log_var)
        d2  = (np.sum(X1 ** 2, 1, keepdims=True)
               + np.sum(X2 ** 2, 1)
               - 2.0 * X1 @ X2.T)
        return var * np.exp(-0.5 * np.maximum(d2, 0.0) / (ls ** 2))

    def _cholesky(self, M):
        jitter = 1e-6
        for _ in range(8):
            try:
                return np.linalg.cholesky(M + jitter * np.eye(len(M)))
            except np.linalg.LinAlgError:
                jitter *= 10
        raise np.linalg.LinAlgError("Cholesky failed")

    # ---------- fit / predict ----------

    def fit(self, X, y, optimize=True):
        self.X = X.copy()
        self.y_mean = np.mean(y)
        self.y_std  = max(np.std(y), 1e-8)
        self.y = (y - self.y_mean) / self.y_std

        if optimize:
            self._optimize_hp()
        self._cache()

    def _cache(self):
        K = self._K(self.X, self.X) + np.exp(self._log_noise) * np.eye(len(self.X))
        self.L = self._cholesky(K)
        self.alpha = np.linalg.solve(
            self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, Xnew):
        Ks = self._K(Xnew, self.X)                # (m, n)
        mu = Ks @ self.alpha                       # (m,)
        V  = np.linalg.solve(self.L, Ks.T)        # (n, m)
        var = np.exp(self._log_var) - np.sum(V ** 2, axis=0)
        var = np.maximum(var, 1e-10)

        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std

    # ---------- hyperparameter optimization ----------

    def _nlml(self, theta):
        self._log_ls, self._log_var, self._log_noise = theta
        n = len(self.X)
        K = self._K(self.X, self.X) + np.exp(self._log_noise) * np.eye(n)
        try:
            L = self._cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return (0.5 * self.y @ a
                + np.sum(np.log(np.diag(L)))
                + 0.5 * n * np.log(2 * np.pi))

    def _optimize_hp(self):
        best_val   = np.inf
        best_theta = np.array([self._log_ls, self._log_var, self._log_noise])
        rng = np.random.default_rng()

        starts = [best_theta.copy()]
        for _ in range(5):
            starts.append(np.array([
                rng.uniform(-2, 3),
                rng.uniform(-2, 3),
                rng.uniform(-6, -1),
            ]))

        for th0 in starts:
            try:
                res = sp_minimize(
                    self._nlml, th0, method="L-BFGS-B",
                    bounds=[(-4, 5), (-4, 5), (-8, 1)],
                    options={"maxiter": 80, "ftol": 1e-7},
                )
                if res.fun < best_val:
                    best_val   = res.fun
                    best_theta = res.x.copy()
            except Exception:
                pass

        self._log_ls, self._log_var, self._log_noise = best_theta


# ======================== Acquisition ========================

def ucb_acquisition(X_cand, gp, beta=UCB_BETA):
    mu, std = gp.predict(X_cand)
    return mu + beta * std


def select_batch(gp, f_best, best_xu, rng):
    """
    Pick BATCH points using UCB with local+global candidates and diversity.

    Candidate pool:
      - 50% global random in [0,1]^18
      - 50% local perturbation around current best (Gaussian, std=LOCAL_STD)

    Selection:
      - N_RANDOM slots reserved for pure random exploration
      - Remaining slots filled by top UCB with diversity filtering
    """
    n_half = N_CAND // 2

    X_global = rng.random((n_half, N_DIM))
    X_local  = best_xu + rng.normal(0, LOCAL_STD, (n_half, N_DIM))
    X_local  = np.clip(X_local, 0, 1)

    X_cand = np.vstack([X_global, X_local])
    scores = ucb_acquisition(X_cand, gp)

    order = np.argsort(-scores)
    batch = []
    used  = np.zeros(len(X_cand), dtype=bool)

    n_acq = BATCH - N_RANDOM

    for idx in order:
        if len(batch) >= n_acq:
            break
        if used[idx]:
            continue

        x = X_cand[idx]
        batch.append(x)
        used[idx] = True

        dists = np.sqrt(np.sum((X_cand - x) ** 2, axis=1))
        used |= (dists < MIN_DIST)

    for _ in range(N_RANDOM):
        batch.append(rng.random(N_DIM))

    while len(batch) < BATCH:
        batch.append(rng.random(N_DIM))

    return np.array(batch)


# ======================== Latin Hypercube Sampling ========================

def latin_hypercube(n, rng):
    X = np.zeros((n, N_DIM))
    for j in range(N_DIM):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return X


def to_orig(X_unit):
    return LB + X_unit * (UB - LB)


def to_unit(X_orig):
    return (X_orig - LB) / (UB - LB)


# ======================== Main BO Loop ========================

def run_bo():
    rng = np.random.default_rng(42)
    gp  = GaussianProcess()

    total_budget = N_INIT + N_ITER * BATCH

    print(f"\n{'='*70}")
    print(f"  Bayesian Optimization for 11-Element Yagi-Uda")
    print(f"  init={N_INIT} LHS, iter={N_ITER}, batch={BATCH}")
    print(f"  Total budget: ~{total_budget} evaluations")
    print(f"{'='*70}")

    # ---------- Phase 1: LHS initialization ----------
    print(f"\n  Generating {N_INIT} Latin Hypercube samples ...")
    t0 = time.time()

    X_unit = latin_hypercube(N_INIT, rng)
    X_orig = to_orig(X_unit)
    results = batch_evaluate(X_orig, "bo_init")

    fits  = np.array([r[0] for r in results])
    gains = np.array([r[1] for r in results])
    s11s  = np.array([r[2] for r in results])

    valid = fits > -900
    X_data = X_unit[valid]
    y_data = fits[valid]

    best_idx  = np.argmax(y_data)
    best_fit  = y_data[best_idx]
    best_xu   = X_data[best_idx].copy()
    best_gain = gains[valid][best_idx]
    best_s11  = s11s[valid][best_idx]

    n_evals   = N_INIT
    hist_best = [best_fit]
    hist_ei   = []

    init_time = time.time() - t0
    print(f"  [init] {np.sum(valid)}/{N_INIT} valid  "
          f"best={best_fit:7.3f}  gain={best_gain:.2f} dBi  "
          f"S11={best_s11:.1f} dB  ({init_time:.1f}s)")

    # ---------- Phase 2: BO iterations ----------
    t_bo = time.time()

    for it in range(N_ITER):
        opt_hp = (it % HP_EVERY == 0)
        gp.fit(X_data, y_data, optimize=opt_hp)

        if opt_hp:
            ls  = np.exp(gp._log_ls)
            var = np.exp(gp._log_var)
            nse = np.exp(gp._log_noise)
            print(f"         GP hyperparams: ls={ls:.4f} var={var:.4f} noise={nse:.6f}")

        batch_u = select_batch(gp, best_fit, best_xu, rng)
        batch_o = to_orig(batch_u)

        results = batch_evaluate(batch_o, f"bo_{it}")
        b_fits  = np.array([r[0] for r in results])
        b_gains = np.array([r[1] for r in results])
        b_s11s  = np.array([r[2] for r in results])

        for j in range(BATCH):
            if b_fits[j] > -900:
                X_data = np.vstack([X_data, batch_u[j]])
                y_data = np.append(y_data, b_fits[j])

                if b_fits[j] > best_fit:
                    best_fit  = b_fits[j]
                    best_xu   = batch_u[j].copy()
                    best_gain = b_gains[j]
                    best_s11  = b_s11s[j]

        n_evals += BATCH
        hist_best.append(best_fit)

        elapsed = time.time() - t_bo
        tag = "HP" if opt_hp else "  "
        print(
            f"  [{it+1:3d}/{N_ITER}]  "
            f"best={best_fit:7.3f}  gain={best_gain:.2f} dBi  "
            f"S11={best_s11:.1f} dB  "
            f"data={len(y_data):4d}  {tag}  {elapsed:.0f}s"
        )

    bo_time  = time.time() - t_bo
    tot_time = time.time() - t0

    best_xo = to_orig(best_xu)

    print(f"\n  BO completed: {bo_time:.1f}s  (total {tot_time:.1f}s)")
    print(f"  GP dataset: {len(y_data)} valid points, {n_evals} evaluations")

    return best_xo, best_fit, best_gain, best_s11, hist_best, n_evals, tot_time


# ======================== Output Helpers ========================

def save_optimized_nec(params: np.ndarray, path: Path):
    d_lens = params[:9]
    d_sps  = params[9:]
    lines = [
        "CM --------------------------------------------------------",
        "CM 11-Element Yagi-Uda (Bayesian Optimization)",
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
        "' --- Director lengths (Bayesian Optimization) ---",
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
        "CM Yagi 11el bayes final", "CE",
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
    stem = "bayes_final"
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


def plot_convergence(hist_best, n_init, path):
    fig, ax = plt.subplots(figsize=(12, 5))

    x_init = [0]
    x_bo   = list(range(1, len(hist_best)))

    ax.axvline(1, color="gray", ls=":", lw=2, alpha=0.5)
    ax.text(1.5, hist_best[0] - 0.3, "LHS -> BO", fontsize=10, color="gray")

    ax.plot(0, hist_best[0], "s", color="tab:blue", ms=8, label=f"LHS init ({n_init} pts)")
    ax.plot(x_bo, hist_best[1:], "-o", color="tab:red", ms=3, lw=2, label="BO iterations")

    ax.set_xlabel("BO Iteration", fontsize=13)
    ax.set_ylabel("Best Fitness (dBi)", fontsize=13)
    ax.set_title("Bayesian Optimization Convergence - 11-Element Yagi-Uda",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close()


# ======================== Main ========================

if __name__ == "__main__":
    best_x, best_fit, best_gain, best_s11, hist_best, n_evals, tot_time = run_bo()

    print("\nRunning final verification (full 3D pattern) ...")
    metrics = final_verification(best_x)

    d_lens = best_x[:9]
    d_sps  = best_x[9:]

    print(f"\n{'='*70}")
    print(f"  BAYESIAN OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"  Forward Gain : {metrics['fwd_gain']:.2f} dBi")
    print(f"  Max Gain     : {metrics['max_gain']:.2f} dBi")
    print(f"  F/B Ratio    : {metrics['F/B']:.2f} dB")
    print(f"  Impedance    : {metrics['Z_real']:.2f} + j({metrics['Z_imag']:.2f}) ohm")
    print(f"  S11          : {metrics['S11']:.2f} dB")
    print(f"  VSWR         : {metrics['VSWR']:.3f}")
    print(f"  Total evals  : {n_evals}")
    print(f"  Total time   : {tot_time:.1f}s ({tot_time/60:.1f} min)")
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

    # Save outputs
    nec_path = SAVE_DIR / "Q3_optimized_bayes.nec"
    save_optimized_nec(best_x, nec_path)
    print(f"\n  Optimized .nec : {nec_path}")

    conv_path = SAVE_DIR / "fig_bayes_convergence.png"
    plot_convergence(hist_best, N_INIT, conv_path)
    print(f"  Convergence    : {conv_path}")

    param_path = SAVE_DIR / "bayes_best_params.txt"
    with open(param_path, "w") as f:
        f.write(f"# Bayesian Optimization Best\n")
        f.write(f"# Gain={metrics['fwd_gain']:.2f} dBi, "
                f"S11={metrics['S11']:.2f} dB, F/B={metrics['F/B']:.2f} dB\n")
        for i in range(9):
            f.write(f"d{i+1}_len={d_lens[i]:.6f}\n")
        for i in range(9):
            f.write(f"d{i+1}_sp={d_sps[i]:.6f}\n")
    print(f"  Parameters     : {param_path}")
    print("=" * 70)

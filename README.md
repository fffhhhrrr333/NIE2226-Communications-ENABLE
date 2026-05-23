# 📡 11-Element Yagi-Uda Antenna Optimization

> Optimizing an 11-element Yagi-Uda antenna (300 MHz) using four optimization algorithms based on the **4NEC2** simulation engine

---

## 📂 Project Structure

```
├── 🧬 optimization_PSO.py           # PSO (Particle Swarm Optimization)
├── 🔬 optimization_cmaes.py         # CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
├── ⚡ optimization_hybrid.py         # Hybrid DE→CMA-ES Optimization
├── 🧠 optimization_bayes.py          # Bayesian Optimization (GP+UCB)
├── 📊 fig_*_convergence.png          # Convergence plots
└── 📝 *_best_params.txt              # Best parameter records
```

---

## 🛠️ Environment Setup

### 1️⃣ Install 4NEC2

1. Download [4NEC2](https://www.qsl.net/4nec2/) and install to `C:\4nec2`
2. Verify the following paths exist:
   ```
   C:\4nec2\exe\nec2dxs500.exe   ← NEC2 computation engine
   C:\4nec2\out\                  ← Simulation output directory
   ```

### 2️⃣ Install Python Dependencies

Requires **Python 3.10+**:

```bash
pip install numpy matplotlib scipy
```

> 💡 `scipy` is only required for Bayesian Optimization (GP hyperparameter optimization + erf function). The other three algorithms only need numpy + matplotlib

---

## 🔌 4NEC2 Integration

This project **does not use an API**. Instead, it communicates with the NEC2 engine via **file I/O + subprocess**:

```
┌──────────────────┐   Write .inp file    ┌──────────────────┐
│  Python script   │ ───────────────────▶  │  nec2dxs500.exe  │
│                  │   Pass filename via   │  (Fortran engine) │
│                  │ ◀───────────────────  │                  │
│  Parse .out      │   Read .out file      │  Generate .out    │
└──────────────────┘                      └──────────────────┘
```

1. 🔄 **Generate .inp** — Write antenna parameters into a standard NEC2 format file
2. 🚀 **Invoke engine** — `subprocess.run([engine], input="stem.inp\nstem.out\n")` passes filenames via stdin
3. 📖 **Parse output** — Regex-match Fortran fixed-width output to extract impedance, gain, and other data

---

## 🎯 Optimization Objective

Optimizing **18 parameters** (9 director lengths + 9 director spacings) with the fitness function:

```
fitness = forward_gain - 5 × max(0, S11 + 10)
```

Maximizes forward gain with a penalty when S11 > -10 dB, ensuring antenna impedance matching (50Ω).

All algorithms use `ThreadPoolExecutor(max_workers=8)` for parallel NEC2 engine invocations to accelerate evaluation.

---

## 🏃 How to Run

### ⚙️ Algorithm 1: PSO (Particle Swarm Optimization)

```bash
python optimization_PSO.py
```

| Parameter | Value |
|---|---|
| Particles | 40 |
| Iterations | 100 |
| Inertia weight | 0.9 → 0.4 (linear decay) |
| Parallel threads | 8 |

📦 Output: `Q3_optimized.nec` · `fig_pso_convergence.png` · `pso_best_params.txt`

---

### ⚙️ Algorithm 2: CMA-ES (Covariance Matrix Adaptation)

```bash
python optimization_cmaes.py
```

| Parameter | Value |
|---|---|
| λ (offspring) | 12 |
| μ (selected) | 6 |
| σ₀ (initial step size) | 0.1 |
| Reference | Hansen & Ostermeier (2001) |

📦 Output: `Q3_optimized_cmaes.nec` · `fig_cmaes_convergence.png` · `cmaes_best_params.txt`

---

### ⚙️ Algorithm 3: Hybrid DE→CMA-ES ⭐ Recommended

```bash
python optimization_hybrid.py
```

| Phase | Algorithm | Configuration |
|---|---|---|
| 🌍 Phase 1 | DE/rand/1/bin | 60 individuals × 80 generations, F=0.7, CR=0.9 |
| 🎯 Phase 2 | CMA-ES | Starting from DE best solution, σ₀=0.03, 100 generations |

📦 Output: `Q3_optimized_hybrid.nec` · `fig_hybrid_convergence.png` · `hybrid_best_params.txt`

---

### ⚙️ Algorithm 4: Bayesian Optimization (GP + UCB)

```bash
python optimization_bayes.py
```

| Parameter | Value |
|---|---|
| Initial samples | 100 (Latin Hypercube) |
| BO iterations | 100 |
| Batch parallelism | 8 (6 UCB + 2 random exploration) |
| Surrogate model | Gaussian Process (isotropic RBF kernel) |
| Acquisition function | UCB (β=2.5) |
| Candidate generation | 50% global random + 50% local perturbation |

📦 Output: `Q3_optimized_bayes.nec` · `fig_bayes_convergence.png` · `bayes_best_params.txt`

> ⚠️ **18 dimensions is challenging for standard GP-BO** — the GP surrogate model struggles to build accurate mappings in high-dimensional sparse spaces.
> The noise variance continuously increases (0.0003→0.4), indicating the model treats the objective function as nearly random.
> BO is better suited for **low-dimensional (5-10D) + very expensive evaluation** scenarios.

---

## 📊 Results Comparison

| Algorithm | Gain (dBi) | S11 (dB) | F/B (dB) | VSWR | Evaluations | Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🧬 PSO | 14.49 | -10.2 | 11.35 | — | ~4000 | 37s |
| 🔬 CMA-ES | 14.45 | -13.1 | — | 1.571 | ~1200 | 19s |
| ⚡ **Hybrid DE→CMA-ES** | **15.10** | **-13.5** | **11.82** | **1.535** | **~6060** | **61s** |
| 🧠 Bayesian (GP+UCB) | 12.70 | -11.1 | 13.48 | 1.768 | 900 | 546s |

> 🏆 **Hybrid DE→CMA-ES achieved the best result: 15.10 dBi**
> The CMA-ES refinement phase improved the DE result by approximately 1.1 dB

### 💡 Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|---|---|
| ⚡ Best performance | Hybrid DE→CMA-ES |
| 🕐 Quick and good results | CMA-ES (19s, 14.45 dBi) |
| 🔰 Simple and easy to understand | PSO |
| 🔬 Low-dimensional + expensive evaluations | Bayesian Optimization |

---

## 📋 Notes

- ⚠️ 4NEC2 must be installed at `C:\4nec2`. If your path differs, modify `ENGINE` and `OUT_DIR` in the scripts
- ⚠️ Temporary files are generated in `C:\4nec2\out\` during runtime and can be cleaned up afterwards
- 💻 Windows only (NEC2 engine is a .exe binary)
- 🔢 Optimization results are stochastic and may vary slightly between runs

---

<p align="center">
  <i>✨ Built with Python + 4NEC2 + Metaheuristic Optimization Algorithms ✨</i>
</p>

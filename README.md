# 📡 11-Element Yagi-Uda Antenna Optimization

> 基于 **4NEC2** 仿真引擎，使用四种优化算法优化 11 元八木天线 (300 MHz)

---

## 📂 项目结构

```
E:\communication\
├── 🧬 optimization_PSO.py           # PSO 粒子群优化
├── 🔬 optimization_cmaes.py         # CMA-ES 协方差矩阵自适应优化
├── ⚡ optimization_hybrid.py         # Hybrid DE→CMA-ES 混合优化
├── 🧠 optimization_bayes.py          # Bayesian Optimization (GP+UCB)
├── 📊 fig_*_convergence.png          # 收敛曲线图
└── 📝 *_best_params.txt              # 最优参数记录
```

---

## 🛠️ 环境安装

### 1️⃣ 安装 4NEC2

1. 下载 [4NEC2](https://www.qsl.net/4nec2/) 并安装到 `C:\4nec2`
2. 确认以下路径存在：
   ```
   C:\4nec2\exe\nec2dxs500.exe   ← NEC2 计算引擎
   C:\4nec2\out\                  ← 仿真输出目录
   ```

### 2️⃣ 安装 Python 依赖

需要 **Python 3.10+**：

```bash
pip install numpy matplotlib scipy
```

> 💡 `scipy` 仅贝叶斯优化需要（GP 超参数优化 + erf 函数），其余三个算法仅需 numpy + matplotlib

---

## 🔌 4NEC2 连接原理

本项目**不使用 API**，而是通过 **文件 I/O + 子进程** 与 NEC2 引擎通信：

```
┌──────────────┐     写入 .inp 文件     ┌──────────────────┐
│  Python 脚本  │ ──────────────────────▶ │  nec2dxs500.exe  │
│              │     stdin 传文件名      │   (Fortran 引擎)  │
│              │ ◀────────────────────── │                  │
│  解析 .out   │     读取 .out 文件      │  生成 .out 输出   │
└──────────────┘                        └──────────────────┘
```

1. 🔄 **生成 .inp** — 将天线参数写入纯 NEC2 格式文件
2. 🚀 **调用引擎** — `subprocess.run([engine], input="stem.inp\nstem.out\n")` 通过 stdin 传入文件名
3. 📖 **解析输出** — 正则匹配 Fortran 固定宽度输出，提取阻抗、增益等数据

---

## 🎯 优化目标

优化 **18 个参数**（9 个引向器长度 + 9 个引向器间距），目标函数：

```
fitness = forward_gain - 5 × max(0, S11 + 10)
```

最大化前向增益，S11 > -10 dB 时施加惩罚，确保天线阻抗匹配（50Ω）。

所有算法使用 `ThreadPoolExecutor(max_workers=8)` 并行调用 NEC2 引擎加速评估。

---

## 🏃 运行方式

### ⚙️ 算法 1：PSO 粒子群优化

```bash
python optimization_PSO.py
```

| 参数 | 值 |
|---|---|
| 粒子数 | 40 |
| 迭代数 | 100 |
| 惯性权重 | 0.9 → 0.4（线性递减） |
| 并行线程 | 8 |

📦 输出：`Q3_optimized.nec` · `fig_pso_convergence.png` · `pso_best_params.txt`

---

### ⚙️ 算法 2：CMA-ES 协方差矩阵自适应

```bash
python optimization_cmaes.py
```

| 参数 | 值 |
|---|---|
| λ (子代数) | 12 |
| μ (选择数) | 6 |
| σ₀ (初始步长) | 0.1 |
| 算法来源 | Hansen & Ostermeier (2001) |

📦 输出：`Q3_optimized_cmaes.nec` · `fig_cmaes_convergence.png` · `cmaes_best_params.txt`

---

### ⚙️ 算法 3：Hybrid DE→CMA-ES 混合优化 ⭐ 推荐

```bash
python optimization_hybrid.py
```

| 阶段 | 算法 | 配置 |
|---|---|---|
| 🌍 Phase 1 | DE/rand/1/bin | 60 个体 × 80 代, F=0.7, CR=0.9 |
| 🎯 Phase 2 | CMA-ES | 从 DE 最优解出发, σ₀=0.03, 100 代 |

📦 输出：`Q3_optimized_hybrid.nec` · `fig_hybrid_convergence.png` · `hybrid_best_params.txt`

---

### ⚙️ 算法 4：Bayesian Optimization (GP + UCB)

```bash
python optimization_bayes.py
```

| 参数 | 值 |
|---|---|
| 初始采样 | 100 (Latin Hypercube) |
| BO 迭代 | 100 |
| 每批并行 | 8 (6 UCB + 2 随机探索) |
| 代理模型 | Gaussian Process (isotropic RBF kernel) |
| 采集函数 | UCB (β=2.5) |
| 候选生成 | 50% 全局随机 + 50% 局部扰动 |

📦 输出：`Q3_optimized_bayes.nec` · `fig_bayes_convergence.png` · `bayes_best_params.txt`

> ⚠️ **18 维对标准 GP-BO 是挑战**——GP 代理模型在高维稀疏空间中难以建立精确映射，
> 噪声方差持续上升(0.0003→0.4)表明模型将目标函数视为近乎随机。
> BO 更适合 **低维(5-10D) + 评估代价极高** 的场景。

---

## 📊 结果对比

| 算法 | 增益 (dBi) | S11 (dB) | F/B (dB) | VSWR | 评估次数 | 耗时 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🧬 PSO | 14.49 | -10.2 | 11.35 | — | ~4000 | 37s |
| 🔬 CMA-ES | 14.45 | -13.1 | — | 1.571 | ~1200 | 19s |
| ⚡ **Hybrid DE→CMA-ES** | **15.10** | **-13.5** | **11.82** | **1.535** | **~6060** | **61s** |
| 🧠 Bayesian (GP+UCB) | 12.70 | -11.1 | 13.48 | 1.768 | 900 | 546s |

> 🏆 **Hybrid DE→CMA-ES 取得最佳结果：15.10 dBi**
> CMA-ES 精炼阶段在 DE 基础上提升了约 1.1 dB

### 💡 算法选型建议

| 场景 | 推荐算法 |
|---|---|
| ⚡ 追求最优性能 | Hybrid DE→CMA-ES |
| 🕐 快速得到较好结果 | CMA-ES（19s，14.45 dBi） |
| 🔰 实现简单、易理解 | PSO |
| 🔬 低维 + 评估代价高 | Bayesian Optimization |

---

## 📋 注意事项

- ⚠️ 4NEC2 必须安装在 `C:\4nec2`，如路径不同请修改脚本中的 `ENGINE` 和 `OUT_DIR`
- ⚠️ 运行时会在 `C:\4nec2\out\` 产生临时文件，可在完成后清理
- 💻 仅支持 Windows（NEC2 引擎为 .exe）
- 🔢 优化结果具有随机性，每次运行可能略有不同

---

<p align="center">
  <i>✨ Built with Python + 4NEC2 + 元启发式优化算法 ✨</i>
</p>

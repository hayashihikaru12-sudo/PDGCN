# 温度动态权重 — 方案 C：硬阈值

## 1. 动机

热源动态权重仅覆盖 $q^{*}>0$ 的直照节点，热尾迹区域（$q^{*}=0$ 但温度已升高）权重退化。引入温度作为权重信号：温度超过阈值的节点认定为"热区"，赋予统一的高权重，直接对抗惰性效应。

## 2. 加权损失

$$
\boxed{
\mathcal L_{\mathrm{pde}}
=
\frac{
\sum_{i\in\Omega_{\mathrm{int}}}
w_i^n\,
\big(R_i^{n,*}\big)^2
}{
\sum_{i\in\Omega_{\mathrm{int}}}
w_i^n + \varepsilon
}}
$$

## 3. 权重公式

### 3.1 温度变量选择

选用 $\tilde T^n$（热源注入后、PD-GCN 前的 `source_temperature`），因其包含的 $\Delta T_Q^n$ 仅依赖输入热流，与模型预测无关——切断反馈循环。

### 3.2 权重定义

$$
\boxed{
w_i^n =
\begin{cases}
W_{\mathrm{high}}, & \tilde T_i^{n,*} > \tau \\[4pt]
1, & \tilde T_i^{n,*} \le \tau
\end{cases}}
$$

| 参数 | 推荐值 | 含义 |
| :-- | :-- | :-- |
| $\tau$ | $1.0$ | 无量纲温度阈值（对应真实温度 $T_{\mathrm{amb}}+\Delta T_0=350^\circ\text{C}$） |
| $W_{\mathrm{high}}$ | $4.0$ | 热区节点统一权重 |

### 3.3 设计意图

- $\tau = 1.0$ 的物理锚点：超过一个特征温差 $\Delta T_0$ 即认定为"被显著加热"；
- $W_{\mathrm{high}} = 4$ 表示热区节点对损失的贡献是冷节点的 4 倍；
- 二值化权重天然免疫极端温度（$T_{\max} \approx 8000^\circ\text{C}$，$\tilde T_{\max}^{*} \approx 34$）的权重爆炸问题。

## 4. 反馈循环分析

硬阈值天然阻尼反馈循环：热源加热区几乎永远满足 $\tilde T^{*} \gg \tau = 1.0$，即使模型预测偏低导致 $\tilde T^{*}$ 下降 $10\%\sim20\%$，只要仍高于 $\tau$，权重保持 $W_{\mathrm{high}}$ 不变——零衰减。

$$
\text{预测偏低 } \Delta\% \;\not\Rightarrow\; \text{权重降低}
\qquad(\text{只要 }\tilde T^{*} > \tau)
$$

Teacher Forcing 模式下风险完全不存在。

## 5. Warmup

$$
w_i^n =
\begin{cases}
1.0, & e < E_{\mathrm{warmup}} \\[6pt]
1.0 + \lambda(e) \cdot (W_{\mathrm{high}} - 1) \cdot \mathbb{I}[\tilde T_i^{n,*} > \tau], & E_{\mathrm{warmup}} \le e < 2E_{\mathrm{warmup}} \\[6pt]
W_{\mathrm{high}} \text{ if } \tilde T_i^{n,*} > \tau \text{ else } 1, & e \ge 2E_{\mathrm{warmup}}
\end{cases}
$$

$\lambda(e)=\dfrac{e-E_{\mathrm{warmup}}}{E_{\mathrm{warmup}}}$，$E_{\mathrm{warmup}}=50$。

## 6. 验证

### 6.1 参数扫描

| 实验 | $\tau$ | $W_{\mathrm{high}}$ | 目的 |
| :-- | :-- | :-- | :-- |
| 基线 | — | — | 仅热源权重基准 |
| C1 | $0.5$ | $4$ | 低阈值（含尾迹边缘） |
| C2 | $1.0$ | $4$ | 推荐配置 |
| C3 | $2.0$ | $4$ | 高阈值（仅热源核心） |
| C4 | $1.0$ | $3$ | 温和增强 |
| C5 | $1.0$ | $6$ | 激进增强 |

### 6.2 核心指标

1. **分位数 RMSE**：节点按 $\tilde T_i^{*}$ 分为 5 个区间，分别统计 RMSE。预期 $\tilde T^{*} > \tau$ 的区间在引入权重后显著下降；
2. **峰值追踪**：沿扫描路径对比 $\max_i T_i$ 的预测值与 FEM；
3. **有效节点占比**：监控 $\tilde T^{*} > \tau$ 的节点比例，确保热区节点数量足够支持稳定训练（建议 $>2\%$）。

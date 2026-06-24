# 温度动态权重 — 方案 B：连续截断

## 1. 动机

热源动态权重仅覆盖 $q^{*}>0$ 的直照节点，热尾迹区域（$q^{*}=0$ 但温度已升高）权重退化。引入温度作为权重信号，使高温区获得更高损失权重，直接对抗惰性效应。

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
w_i^n = \operatorname{clamp}\!\Big(1 + \beta \cdot \max(0,\tilde T_i^{n,*}),\;\;1,\;W_{\max}\Big)
}
$$

| 参数 | 推荐值 | 含义 |
| :-- | :-- | :-- |
| $\beta$ | $0.5$ | 温度→权重映射斜率 |
| $W_{\max}$ | $8.0$ | 权重上限，防止极端高温独霸梯度 |

### 3.3 设计意图

- **线性段**（$\tilde T_i^{*} \in [0,\;14]$，$\beta=0.5$ 时）：温度越高权重越大，热区内保留细粒度区分
- **截断段**（$\tilde T_i^{*} > 14$）：所有极端高温节点统一赋予 $W_{\max}$，防止 $8000^\circ\text{C}$ 量级的节点权重爆炸

## 4. 反馈循环分析

Rollout 模式下预测偏低会导致 $\tilde T^{*}$ 偏低、权重偏低。缓解因素：

- $\Delta T_Q^n$ 来自输入热流，为热源节点提供独立锚点；
- $W_{\max}$ 截断限制了大偏差的放大效应；
- Teacher Forcing 模式下风险完全不存在。

## 5. Warmup

$$
w_i^n =
\begin{cases}
1.0, & e < E_{\mathrm{warmup}} \\[6pt]
1.0 + \lambda(e) \cdot \beta \cdot \max(0,\tilde T_i^{n,*}), & E_{\mathrm{warmup}} \le e < 2E_{\mathrm{warmup}} \\[6pt]
\operatorname{clamp}(1 + \beta \cdot \max(0,\tilde T_i^{n,*}),\,1,\,W_{\max}), & e \ge 2E_{\mathrm{warmup}}
\end{cases}
$$

$\lambda(e)=\dfrac{e-E_{\mathrm{warmup}}}{E_{\mathrm{warmup}}}$，$E_{\mathrm{warmup}}=50$。

## 6. 验证

### 6.1 参数扫描

| 实验 | $\beta$ | $W_{\max}$ | 目的 |
| :-- | :-- | :-- | :-- |
| 基线 | — | — | 仅热源权重基准 |
| B1 | $0.25$ | $8$ | 温和斜率 |
| B2 | $0.5$ | $8$ | 推荐配置 |
| B3 | $1.0$ | $8$ | 陡峭斜率 |
| B4 | $0.5$ | $4$ | 紧缩上限 |
| B5 | $0.5$ | $12$ | 宽松上限 |

### 6.2 核心指标

1. **分位数 RMSE**：节点按 $\tilde T_i^{*}$ 分为 5 个区间，分别统计 RMSE；
2. **峰值追踪**：沿扫描路径对比 $\max_i T_i$ 的预测值与 FEM；
3. **权重分布监控**：记录每 epoch 的 $w_i$ 直方图，确认截断生效。

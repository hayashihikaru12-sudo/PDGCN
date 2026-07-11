# FDM 后热源补偿方案

## 1. 问题定位

当前算子顺序为 **PD-GCN → FDM**（原始顺序）：

```
Step A — 显式热源（仅顶层）:
  T_src[0] = T_cur[0] + ΔT_Q

Step B — PD-GCN 面内输运:
  T_inplane = T_src + PDGCN(T_src, graph)

Step C — 厚度方向隐式 FDM:
  (I − C_n·D) · T_next[0:L−1] = T_inplane[0:L−1]
  T_next[L−1] = T_bottom
```

PD-GCN 在 Step B 看到的是"未经厚度散热"的温度场 $T_{\mathrm{src}}$。它基于此预测面内输运增量，但不知道 Step C 的 FDM 会将部分热量从顶层抽走。该冷却效应 PD-GCN 完全不可见，导致每步存在一个**系统性的热损失**。

## 2. FDM 的冷却效应

以三层为例。FDM 三对角系统的第一行：

$$
(1 + C_n) \cdot T_{\mathrm{next}}[0] \;-\; C_n \cdot T_{\mathrm{next}}[1] \;=\; T_{\mathrm{inplane}}[0]
$$

顶层最终温度 $T_{\mathrm{next}}[0]$ 是 $T_{\mathrm{inplane}}[0]$ 和 $T_{\mathrm{next}}[1]$ 的加权折衷，权重由 $C_n$ 控制。铺设初期下层仍为冷模温（$T_{\mathrm{next}}[1] \ll T_{\mathrm{next}}[0]$），近似有：

$$
T_{\mathrm{next}}[0] \;\approx\; \frac{1}{1 + C_n} \cdot T_{\mathrm{inplane}}[0]
$$

即 FDM 一步就将顶层的热增量稀释了约 $\frac{C_n}{1+C_n}$。

## 3. 定量估算

以示例配置参数：

$$
\begin{aligned}
C_n &= \frac{k_{\mathrm{ratio}} \cdot dt^* \cdot \mathrm{Pe}^{-1}}{dz^{*2}} \approx 0.62 \\[6pt]
\text{顶层保留比例} &= \frac{1}{1 + C_n} \approx 61.7\% \\[6pt]
\text{每步损失比例} &= \frac{C_n}{1 + C_n} \approx 38.3\%
\end{aligned}
$$

设热源在热点处的无量纲温升 $\Delta T_Q^* \approx 0.3$（对应 $\Delta T_0 = 900\ \mathrm{K}$ 时约 $270\ \mathrm{°C}$）：

$$
\text{每步被 FDM 抽走} \;\approx\; 0.383 \times 0.3 = 0.115 \quad (\text{无量纲}) \;\approx\; 103\ \mathrm{°C}
$$

## 4. 补偿方案

在 FDM **之后**对顶层追加补偿热源，直接修正 $T_{\mathrm{next}}$：

```
Step A — 显式热源:
  T_src[0] = T_cur[0] + ΔT_Q

Step B — PD-GCN 面内输运:
  T_inplane = T_src + PDGCN(T_src, graph)

Step C — 厚度方向隐式 FDM:
  T_fdm = ImplicitFDM(T_inplane)

Step D — 后补偿（新增）:
  T_fdm[0] += α · ΔT_Q              ← 补回被 FDM 抽走的热量
  T_next = ApplyBoundary(T_fdm)     ← 重新钳制面内和底层边界
```

其中 $\alpha \in [0, 1]$ 为补偿系数，默认 $0.0$（与旧行为一致）。

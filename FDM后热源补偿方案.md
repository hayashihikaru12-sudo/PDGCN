# FDM 后热源补偿方案

## 1. 问题定位

当前算子顺序已改为 **FDM → PD-GCN**，消除了 $O(dt^2)$ 分裂误差项。但 FDM 的 Backward Euler 隐式步会将顶层热量稀释到下层——这一过程 PD-GCN 完全不可见，导致每步存在一个**固定的热损失**。

## 2. 单步热量流动分析

当前每步的算子执行顺序：

```
Step A — 显式热源（仅顶层）:
  T_src[0] = T_cur[0] + ΔT_Q

Step B — 厚度方向隐式 FDM:
  (I − C_n·D) · T_fdm[0:L−1] = T_src[0:L−1]
  T_fdm[L−1] = T_bottom

Step C — PD-GCN 面内输运:
  T_next = T_fdm + PDGCN(T_fdm, graph)
```

关键在 Step B。FDM 三对角系统的第一行：

$$
(1 + C_n) \cdot T_{\mathrm{fdm}}[0] \;-\; C_n \cdot T_{\mathrm{fdm}}[1] \;=\; T_{\mathrm{src}}[0]
$$

在铺设初期（下层仍为冷模温），$T_{\mathrm{fdm}}[1] \ll T_{\mathrm{fdm}}[0]$，近似有：

$$
T_{\mathrm{fdm}}[0] \;\approx\; \frac{1}{1 + C_n} \cdot T_{\mathrm{src}}[0]
$$

即热源注入后，FDM 一步就抽走了顶层约 $\frac{C_n}{1+C_n}$ 的热增量。

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

在 FDM 之后、PD-GCN 之前，对顶层追加一份**补偿热源**：

```
Step A — 显式热源:
  T_src[0] = T_cur[0] + ΔT_Q

Step B — FDM 厚度扩散:
  T_fdm = ImplicitFDM(T_src)

Step C — 补偿热源（新增）:
  T_fdm[0] += α · ΔT_Q              ← 补回被 FDM 抽走的热量
  T_fdm = ApplyBoundary(T_fdm)      ← 重新钳制面内和底层边界

Step D — PD-GCN 面内输运:
  T_next = T_fdm + PDGCN(T_fdm, graph)
```

其中 $\alpha \in [0, 1]$ 为补偿系数，对应推理配置 `post_fdm_source_compensation_alpha`。代码默认值为 `0.0`，保证旧配置结果不变；理论热损失估算约为 `0.383`，但示例配置经 50 个多层 FEM 案例实测标定为 `0.3`。该系数是独立校准参数，不根据 $C_n$ 自动推导或关闭。

补偿只改变进入 PD-GCN 的顶层温度 `T*`。若模型启用了热源节点特征，`q*` 和 `ΔT_Q*` 仍表示原始显式热源，不乘以 $(1+\alpha)$；非顶层热源特征继续置零。即使关闭 PD-GCN 面内增量，补偿仍属于外部热源路径并正常生效。

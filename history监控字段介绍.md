# History 监控字段介绍

`history.json` 是 PDGCN 训练过程中自动输出的监控文件，位于 `runs/<run_name>/history.json`。它记录每个 epoch 的损失分量、温度统计、模型参数和训练状态，用于训练过程追踪、收敛判断和事后分析。

## 文件结构

```json
{
  "history": [ ... ],   // 每个 epoch 一条记录，按训练顺序排列
  "metadata": { ... }   // 本次训练的完整运行配置（run_config）
}
```

- `history`：数组，每个元素为一个 epoch 的监控记录，长度等于实际训练的 epoch 数（含提前停止的情况）。
- `metadata`：训练时使用的完整运行配置，包含数据集路径、模型超参数、物理损失权重、训练参数、监督配置、监控参数等。可用于复现实验或对比不同实验的配置差异。

---

## 各字段详解

### 1. 基础标识与学习率

| 字段 | 类型 | 含义 |
|------|------|------|
| `epoch` | int | 训练轮次索引，从 0 开始。epoch=0 表示第 1 轮。 |
| `lr` | float | 当前 epoch 使用的学习率。受 `lr_scheduler` 策略（`warmup_cosine` / `plateau` / `none`）控制，会随训练动态调整。 |

### 2. 总损失与顶层分量

训练的总损失由物理损失和监督损失两部分加权组成：

$$L_{\text{total}} = L_{\text{physics}} + \lambda_T \cdot L_{\text{temperature}}$$

| 字段 | 类型 | 含义 |
|------|------|------|
| `loss` | float | 当前 epoch 的总损失，等价于 `loss_total`，用于终端快速显示。 |
| `loss_total` | float | 总损失 = `loss_physics` + `loss_supervised`。 |
| `loss_physics` | float | 物理损失：PDE 残差 + 出流边界 + 图平滑的加权和 = $\lambda_{\text{pde}} \cdot$ `loss_pde` + $\lambda_{\text{outflow}} \cdot$ `loss_outflow` + $\lambda_{\text{smooth}} \cdot$ `loss_smooth`。 |
| `loss_supervised` | float | 监督损失：$\lambda_T \cdot$ `loss_temperature`。未启用 FEM 监督时为 0。 |

### 3. 温度监督损失（FEM 监督相关）

仅在 `supervision.enabled = true` 时有非零值；未启用时全部为 0。

| 字段 | 类型 | 含义 |
|------|------|------|
| `loss_temperature` | float | 温度监督 MSE 损失（当前监督模式下的主温度损失）。 |
| `loss_teacher_forcing_temperature` | float | Teacher Forcing 模式的单步温度监督损失。在 teacher forcing 路径中，每步输入取 FEM 真实温度，约束单步预测偏离 FEM 标签的 MSE。 |
| `loss_rollout_temperature` | float | Rollout 模式的自回归温度监督损失。在 rollout 路径中，模型用自身预测作为下一步输入进行自回归推演，约束推演结果偏离 FEM 标签的 MSE。 |

**监督模式说明**：
- `teacher_forcing`：仅启用 `loss_teacher_forcing_temperature`，`loss_rollout_temperature` 为 0。
- `rollout`：仅启用 `loss_rollout_temperature`，`loss_teacher_forcing_temperature` 为 0。
- `mixed`：同时启用两者，各自独立记录。

### 4. PDE 物理损失分量

这些分量共同构成物理损失 $L_{\text{physics}}$，是 PD-GCN 的核心约束。

| 字段 | 类型 | 含义 |
|------|------|------|
| `loss_pde` | float | **PDE 残差损失**。在内部节点（去除迎风、侧边、尾迹边界）上计算无源曲面内输运 PDE 残差的均方值。残差包含对流项、各向异性扩散项和时间导数项，用 `residual_time_scheme` 控制时间离散方式（`explicit` 用当前温度，`backward` 用预测温度）。该项约束预测温度满足曲面内输运物理方程。 |
| `loss_outflow` | float | **出流边界 Neumann 软约束损失**。在尾迹（downwind）边界节点上计算法向温度梯度平方的均值，约束出流边界满足绝热/自然边界条件。 |
| `loss_beta` | float | **热损失项**（保留字段）。当前版本中始终为 0，为向后兼容而保留。 |
| `loss_smooth` | float | **图梯度平滑正则化损失**。在内部边上计算一阶温度梯度（温度差/边距）的均方值，用于抑制预测温度场的高频非物理振荡。由 `gradient_regularization` 权重控制强度，通常取很小值（如 1e-4）。 |

### 5. FEM 温度监督指标

当 `supervision.enabled = true` 时，这些指标衡量模型预测温度与 FEM 参考温度的偏差。所有指标均以**真实温度（K）**为单位，已从无量纲温度还原。

| 字段 | 类型 | 含义 |
|------|------|------|
| `fem_temperature_rmse` | float | 预测温度与 FEM 标签温度的**均方根误差**（RMSE），单位 K。RMSE 对大误差更敏感，是衡量整体预测精度的核心指标。 |
| `fem_temperature_mae` | float | 预测温度与 FEM 标签温度的**平均绝对误差**（MAE），单位 K。MAE 对异常值不如 RMSE 敏感，反映典型偏差水平。 |
| `fem_temperature_max_error` | float | 预测温度与 FEM 标签温度的**最大绝对误差**，单位 K。用于检测是否存在局部严重偏离的"热点"区域。 |
| `rollout_fem_temperature_rmse` | float | Rollout 路径上的 FEM 温度 RMSE（K）。在 `rollout` 或 `mixed` 监督模式下记录自回归推演路径的温度偏差。 |
| `rollout_fem_temperature_mae` | float | Rollout 路径上的 FEM 温度 MAE（K）。 |
| `rollout_fem_temperature_max_error` | float | Rollout 路径上的 FEM 温度最大绝对误差（K）。 |

> **注意**：在 `teacher_forcing` 模式下，`fem_temperature_*` 记录 teacher forcing 路径的单步误差，`rollout_fem_temperature_*` 为 0。在 `rollout` 模式下，`fem_temperature_*` 与 `rollout_fem_temperature_*` 通常相同。在 `mixed` 模式下，`fem_temperature_*` 取 teacher forcing 路径，`rollout_fem_temperature_*` 取 rollout 路径。

### 6. 温度统计量

记录当前 epoch 所有预测节点温度（还原为真实温度 K）的统计特征，用于监控温度场的整体分布和数值稳定性。

| 字段 | 类型 | 含义 |
|------|------|------|
| `temperature_mean` | float | 所有预测节点温度的平均值（K）。反映整体温度水平。 |
| `temperature_max` | float | 所有预测节点温度的最大值（K）。用于检测是否出现异常高温（发散信号）。 |
| `temperature_min` | float | 所有预测节点温度的最小值（K）。用于检测是否出现异常低温。 |
| `temperature_var` | float | 所有预测节点温度的方差（$\sigma^2$，除以 N 而非 N-1）。反映温度场的空间不均匀程度。 |

> **警告**：若 `temperature_max` 持续快速增长或 `temperature_min` 出现极端负值，通常表明训练发散，需检查学习率、梯度裁剪或 PDE 权重。

### 7. 可学习物理参数

| 字段 | 类型 | 含义 |
|------|------|------|
| `gamma_upwind` | float | PD-GCN 各 **EdgeBlock** 中可学习迎风权重参数 $\gamma_{\text{upwind}}$ 的均值。$\gamma_{\text{upwind}}$ 控制对流项中迎风方向与各向同性扩散的比例，范围通常靠近初始值（如 0.8）。 |
| `gamma_upwind_std` | float | $\gamma_{\text{upwind}}$ 在各 EdgeBlock 间的标准差。标准差小表示各消息传递层的迎风权重趋于一致；标准差大则表示不同层学到了差异化的对流-扩散混合策略。 |

### 8. 窗口级损失分布

| 字段 | 类型 | 含义 |
|------|------|------|
| `window_losses` | array[float] | 当前 epoch 中每个 TBPTT 窗口的总损失列表。长度 = 该 epoch 所有 HDF5 数据文件的 TBPTT 窗口总和。用于分析不同时间窗口的损失分布：若靠后窗口的损失显著高于靠前窗口，说明模型在长时间 rollout 中误差累积严重。 |
| `file_window_counts` | array[int] | 每个 HDF5 数据文件产生的 TBPTT 窗口数量。长度 = 该 epoch 使用的 HDF5 文件数。用于了解数据在各个文件间的分布情况。 |

例如，若 `file_window_counts = [18, 36, 36]`，则 `window_losses` 前 18 个值来自第一个 HDF5 文件的窗口，接着 36 个来自第二个，以此类推。

### 9. 提前停止标记（仅最后一个 epoch）

| 字段 | 类型 | 含义 |
|------|------|------|
| `stopped_early` | bool | 是否因触发提前停止条件而终止训练。仅出现在最后一个 epoch 的记录中。 |
| `stop_reason` | string | 提前停止的原因。当前可能的值：<br>• `"loss_threshold"`：总损失已降至配置的 `training.loss_threshold` 以下，训练自动终止。 |

若训练正常完成所有 epoch（未提前停止），则最后一条记录中不包含这两个字段。

---

## 使用建议

1. **追踪收敛**：观察 `loss_total` 和 `loss_physics` 随 epoch 的下降趋势。若长时间不下降，考虑调整学习率调度器。
2. **诊断发散**：若 `temperature_max` 急剧上升或出现 `NaN`，训练可能发散。检查 `loss_pde` 是否异常增大，必要时降低学习率或增大 `gradient_regularization`。
3. **评估监督效果**：对比 `loss_physics` 和 `loss_supervised` 的大小关系，确保 $\lambda_T$ 权重设置合理。观察 `fem_temperature_rmse` 和 `fem_temperature_max_error` 的下降趋势。
4. **分析 rollout 质量**：比较 `loss_teacher_forcing_temperature` 和 `loss_rollout_temperature`，后者通常更大。两者差距反映模型在自回归路径上的误差累积程度。
5. **窗口分析**：`window_losses` 若呈明显上升趋势（靠后窗口损失更大），表明 TBPTT 窗口长度可能需要调整或模型需要更强的长时间稳定性约束。
6. **跨实验对比**：`metadata` 中包含完整的运行配置，可直接对比不同实验的超参数设置。

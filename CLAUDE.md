# 项目说明

本代码库用于构建曲面内 PDGCN 神经网络训练架构和曲面差分技术，实现多层堆叠铺放曲面温度场预测。PDGCN 架构改造自 PIGNN，面向曲面铺放热场预测应用场景。

## 当前物理职责划分

总体更新式为：

```text
T_next = T_current + delta_T_source + delta_T_inplane + delta_T_thickness
```

- `delta_T_source`：由显式表面热源模块计算，读取 HDF5 `dynamic/Q`，只默认作用于顶层。
- `delta_T_inplane`：由无源 PD-GCN 计算，只负责曲面内对流、曲面扩散和纤维各向异性扩散。
- `delta_T_thickness`：由厚度方向 Backward Euler 隐式 1D FDM 计算，负责层间导热和底层恒温边界。

PD-GCN 节点输入默认兼容为：

```text
[x*, y*, z*, fx, fy, fz, T*]
```

若配置 `include_q_in_features=true` 和/或 `include_delta_t_source_in_features=true`，节点特征会在 `T*` 后追加 `q*` 和/或当前步 `ΔT_Q*`。热源 `dynamic/Q` 始终按表面热流 `q''` 读取，转换为 `W/m^2` 后由显式表面热源模块计算温升；新增热源节点特征只作为 PD-GCN 输入信息，不重新进入 PDE residual 源项。

## FEM 温度监督

当前代码已支持单层 FEM 温度监督训练，配置位于顶层 `supervision`，默认关闭。v1 只支持：

```json
{
  "enabled": true,
  "temperature_dataset": "fem/temperature",
  "valid_mask_dataset": "fem/valid_mask",
  "lambda_temperature": 1.0,
  "mode": "teacher_forcing"
}
```

实现约定：

- `fem/temperature` 形状必须为 `[T, N, 1]`，与 `dynamic/Q` 对齐。
- `fem/valid_mask` 可缺失；缺失时使用全 1 mask。
- FEM 温度读取为真实温度，训练时按

  $$
  T_{\mathrm{FEM}}^*
  =
  \frac{T_{\mathrm{FEM}}-T_{\mathrm{amb}}}{\Delta T_0}
  $$

  转为无量纲温度。
- `fem/temperature_unit` 可为 `degC`、`C` 或 `K`；代码只校验元数据，不自动转换温标。
- FEM 温度只作为监督标签，不作为 PD-GCN 额外节点特征；监督路径只替换节点输入中的 `T*` 列。
- 监督启用时训练 transition 为 `n = 0 ... T-2`，输入温度取 `T_FEM,n*`，不使用 `warmup_steps` 生成训练输入。

监督损失为：

$$
\mathcal L_T
=
\frac{
\sum_i m_i
\left(
T_{\mathrm{pred},i}^*
-
T_{\mathrm{FEM},i}^*
\right)^2
}{
\sum_i m_i+\varepsilon
},
\qquad
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{physics}}
+
\lambda_T\mathcal L_T.
$$

history 和 monitor 会记录 `loss_physics`、`loss_supervised`、`loss_temperature`、`fem_temperature_rmse`、`fem_temperature_mae` 和 `fem_temperature_max_error`。

## 参考资料

1. `DesignPlan/` 中说明研究背景、PDGCN+FDM 温度场预测方案、FEM 监督训练、无量纲化和损失函数策略。
2. `DesignPlan/局部窗口定拓扑采样器.md` 说明单层定拓扑计算域构建过程。
3. `configs/README.md` 说明训练、监督和推理配置。
4. `PIGNN/` 为 PIGNN 参考仓库源码。

除非 prompt 显式要求，`DesignPlan/` 和 `PIGNN/` 均作为参考资料目录处理。本次用户已显式要求同步更新 `DesignPlan/`。

## Python 环境

本代码库的 Python 环境是 conda 中的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```

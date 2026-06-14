# 项目说明

默认回答使用中文，除非显式指定英文。输出文本采用 Markdown 源码格式，确保 Typora 可直接渲染。

本代码库用于构建曲面内 PDGCN 神经网络训练架构和曲面差分技术，实现多层堆叠铺放曲面温度场预测。PDGCN 架构改造自 PIGNN，面向曲面铺放热场预测应用场景。

## 当前物理职责划分

总体更新式为：

```text
T_next = T_current + delta_T_source + delta_T_inplane + delta_T_thickness
```

- `delta_T_source`：由显式表面热源模块计算，读取 HDF5 `dynamic/Q`，只默认作用于顶层。
- `delta_T_inplane`：由无源 PD-GCN 计算，只负责曲面内对流、曲面扩散和纤维各向异性扩散。
- `delta_T_thickness`：由厚度方向 Backward Euler 隐式 1D FDM 计算，负责层间导热和底层恒温边界。

PD-GCN 节点输入固定为：

```text
[x*, y*, z*, fx, fy, fz, T*]
```

热源 `dynamic/Q` 不进入节点特征；它按表面热流 `q''` 读取，转换为 `W/m^2` 后由显式表面热源模块处理。

## FEM 温度监督

当前代码已支持单层 FEM 温度监督训练，配置位于顶层 `supervision`，默认关闭。v1 只支持 `teacher_forcing`。

监督数据约定：

```text
fem/temperature          [T, N, 1]
fem/valid_mask           [T, N, 1]  可选
fem/temperature_unit     degC/C/K   可选
```

FEM 温度读取为真实温度，训练时按以下公式无量纲化：

$$
T_{\mathrm{FEM}}^*
=
\frac{T_{\mathrm{FEM}}-T_{\mathrm{amb}}}{\Delta T_0}.
$$

监督启用时：

- `HDF5FrameReader` 要求 `fem/temperature` 存在，并提供 `has_fem_temperature`、`read_fem_temperature(frame_idx)` 和 `read_fem_valid_mask(frame_idx)`。
- 训练 transition 为 `n = 0 ... T-2`。
- 输入温度取 $T_{\mathrm{FEM},n}^*$，经过显式热源和边界钳制后送入 PD-GCN。
- 预测 $T_{\mathrm{pred},n+1}^*$ 与 $T_{\mathrm{FEM},n+1}^*$ 计算 masked MSE。
- FEM 温度不进入 PD-GCN 额外节点特征，节点输入维度仍为 7。

监督损失为：

$$
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

本次架构改造已要求同步修改 `DesignPlan/`。后续若没有显式要求，仍将 `DesignPlan/` 和 `PIGNN/` 作为参考资料目录，避免随意改动。

## Python 环境

本代码库的 Python 环境是 conda 中的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```

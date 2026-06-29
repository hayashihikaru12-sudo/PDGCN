# pde 模块说明

`pde` 目录实现 PDGCN 训练和推理所需的物理算子，包括无源曲面内输运残差、显式表面热源温升、厚度方向 FDM、Dirichlet 边界处理、出流边界软约束和总损失函数。

## 主要文件

- `residual.py`：实现 `compute_pde_residual`，根据源项已显式处理后的当前温度、下一步温度、扫描速度、时间步长、图拓扑和边特征计算每个节点的无源输运残差。
- `source.py`：实现显式表面热源温升，将 `q''` 或 `q_surface*` 转换为 `delta_T_Q*`。
- `fdm.py`：对外导出厚度方向 FDM 系数、显式层间温度增量诊断函数和 Backward Euler 隐式 FDM 步进函数。
- `loss.py`：实现边界条件和损失聚合，包括 `apply_dirichlet_boundary`、`compute_outflow_loss` 和 `total_loss`。
- `tests/`：包含 PDE 残差和损失函数的单元测试，用于验证形状广播、边界处理和损失分量。
- `__init__.py`：对外导出 PDE 模块的核心接口。

## 物理建模约定

边特征布局遵循 `data` 模块生成的 `[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]`。其中 `cos_theta` 用于扫描方向相关的守恒 FVM 上风对流通量：正值表示通量沿 `sender -> receiver`，sender 对流残差加 `+F`、receiver 加 `-F`；负值表示反向流动。`cos_phi_sq` 和 `k_ratio` 用于构造沿纤维方向的各向异性导热权重。

`upwind` 和 `side` 节点会被钳制到 Dirichlet 温度，默认无量纲值为 `0.0`。`downwind` 节点用于出流边界 Neumann 软约束。PDE 残差支持单步张量和 TBPTT 窗口张量，返回形状会与输入温度保持一致；残差中不再包含热源项或单层等效热汇项。

## 与训练模块的关系

`training/tbptt.py` 和 `training/static_topology.py` 在每个时间步先调用显式热源，再调用 PD-GCN，随后用 `total_loss` 约束无源输运增量。FEM 温度监督损失不在 `pde.total_loss` 中实现，而是在固定拓扑训练循环中与物理损失合并。多层推理通过 `inference/fdm.py` 使用 Backward Euler 隐式厚度 FDM。

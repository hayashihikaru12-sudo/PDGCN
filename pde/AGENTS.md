# pde 模块说明

`pde` 目录实现 PDGCN 训练所需的曲面内热传导物理约束，包括无量纲 PDE 残差、Dirichlet 边界处理、出流边界软约束和总损失函数。

## 主要文件

- `residual.py`：实现 `compute_pde_residual`，根据当前温度、下一步温度、扫描速度、热源、时间步长、图拓扑和边特征计算每个节点的无量纲 PDE 残差。
- `loss.py`：实现边界条件和损失聚合，包括 `apply_dirichlet_boundary`、`compute_outflow_loss` 和 `total_loss`。
- `tests/`：包含 PDE 残差和损失函数的单元测试，用于验证形状广播、边界处理和损失分量。
- `__init__.py`：对外导出 PDE 模块的核心接口。

## 物理建模约定

边特征布局遵循 `data` 模块生成的 `[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]`。其中 `cos_theta` 用于扫描方向相关的迎风对流项，`cos_phi_sq` 和 `k_ratio` 用于构造沿纤维方向的各向异性导热权重。

`upwind` 和 `side` 节点会被钳制到 Dirichlet 温度，默认无量纲值为 `0.0`。`downwind` 节点用于出流边界 Neumann 软约束。PDE 残差支持单步张量和 TBPTT 窗口张量，返回形状会与输入温度保持一致。

## 与训练模块的关系

`training/tbptt.py` 和 `training/static_topology.py` 在每个时间步或窗口中调用 `total_loss`。模型只预测温度增量，最终物理一致性主要由本模块中的残差项和边界项约束。

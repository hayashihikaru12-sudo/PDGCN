# data 模块说明

`data` 目录负责把曲面铺放热场 HDF5 数据转换为 PDGCN 可直接使用的图数据，并提供无量纲化、初温生成和固定拓扑缓存等数据层能力。

## 主要职责

- 读取 `DesignPlan/1.h5` 这类预生成 HDF5 数据，提取节点坐标、纤维方向、热源、边索引和边界节点。
- 将真实物理量转换为无量纲量，保证模型输入和 PDE 损失使用一致的尺度。
- 构建 PyTorch Geometric `Data` 对象，包括节点特征、边特征、全局工艺条件和边界索引。
- 为固定拓扑训练生成 memmap 缓存，减少训练时重复读取和重复构图的开销。
- 在没有模型 warmup 初温时，提供 legacy 图扩散式初温 fallback。

## 主要文件

- `loader.py`：定义 `GraphRawData` 和 `HDF5Loader`，负责按时间帧读取并校验 HDF5 图数据。
- `dimensionless.py`：定义 `ScaleParams`，实现坐标、温度、热源、速度的无量纲化和反变换，并派生 PDE 常数。
- `feature_builder.py`：把原始帧数据组装为 PDGCN 图输入，节点特征布局为 `[x*, y*, z*, fx, fy, fz, T*, Q*]`，边特征布局为 `[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]`。
- `initial_condition.py`：基于图扩散松弛生成无量纲初始温度，主要作为数据层 fallback。
- `static_cache.py`：从 HDF5 生成固定拓扑缓存，并提供 `FrameMemmapReader` 按帧读取动态基础特征。
- `__init__.py`：对外导出数据模块的核心接口。

## 数据约定

HDF5 输入需要包含 `dynamic/xyz`、`dynamic/fiber`、`dynamic/Q`、`edge_index` 以及 `boundary_nodes/upwind`、`boundary_nodes/downwind`、`boundary_nodes/side`。其中 `upwind` 和 `side` 通常用于 Dirichlet 边界，`downwind` 用于出流边界约束。

本模块不负责模型训练循环，只提供训练前和训练中需要的数据表示。训练端的固定拓扑高速路径会优先复用 `static_cache.py` 生成的缓存。

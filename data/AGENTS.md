# data 模块说明

`data` 目录负责把曲面铺放热场 HDF5 数据转换为 PDGCN 可直接使用的图数据，并提供无量纲化、初温生成和固定拓扑缓存等数据层能力。

## 主要职责

- 读取预生成 HDF5 数据，提取节点坐标、纤维方向、表面热流、边索引和边界节点。
- 将真实物理量转换为无量纲量；表面热流不进入 PD-GCN 节点特征，而是保存为图对象字段供显式热源模块使用。
- 构建 PyTorch Geometric `Data` 对象。
- 为固定拓扑训练生成共享静态缓存，只保存拓扑、边界节点、节点类型和特征维度等静态信息。
- 训练时由 `HDF5FrameReader` 按切片文件读取动态基础特征。

## 主要文件

- `loader.py`：定义 `GraphRawData` 和 `HDF5Loader`，负责按时间帧读取并校验 HDF5 图数据。
- `dimensionless.py`：定义 `ScaleParams`，实现无量纲化和 PDE 常数派生。
- `feature_builder.py`：把原始帧数据组装为 PDGCN 图输入。
- `initial_condition.py`：提供 legacy 图扩散式初温 fallback。
- `static_cache.py`：生成共享静态缓存，并提供 `HDF5FrameReader` 按帧读取动态基础特征。

## 数据约定

用于生成静态缓存的首个 HDF5 文件需要包含 `dynamic/xyz`、`dynamic/fiber`、`dynamic/normal`、`dynamic/Q`、`edge_index` 以及 `boundary_nodes/upwind`、`boundary_nodes/downwind`、`boundary_nodes/side`。其中 `dynamic/Q` 为表面热流，读取后转换为 `W/m^2`。

后续同目录 HDF5 文件只校验动态数据集、节点数和动态形状，并复用共享静态缓存中的拓扑信息。

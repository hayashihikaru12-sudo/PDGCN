# PDGCN

PDGCN 是面向曲面铺放热场预测的图神经网络训练框架。项目目标是构建曲面内 PDGCN 神经网络与曲面差分技术，用于预测多层堆叠铺放曲面的温度场。

## 项目特点

- 使用 PyTorch Geometric 表示曲面局部窗口图。
- 节点特征包含坐标、纤维方向、当前温度和热源强度。
- 边特征编码局部几何、扫描方向和纤维各向异性关系。
- 使用物理约束损失训练，包括 PDE residual 和出流边界约束。
- 支持目录级 HDF5 切片训练：同一目录内多个 `.h5` 文件按自然升序训练，每个文件是独立序列。
- 支持共享静态缓存：一个训练集目录复用同一份拓扑缓存，动态帧数据从各 HDF5 文件读取。

## 环境

本项目使用 conda 中已配置好的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```

建议训练和测试命令都使用该 Python 解释器。

## 数据约定

用于生成静态缓存的首个 HDF5 文件需要包含：

```text
dynamic/xyz
dynamic/fiber
dynamic/Q
edge_index
boundary_nodes/upwind
boundary_nodes/downwind
boundary_nodes/side
path/heat_center_step_distance
```

HDF5 原始切片采用生成程序的原生单位：几何为 `mm`，速度为 `mm/s`，`dynamic/Q` 为面热流 `W/mm^2`。PDGCN 预处理阶段会强制转换为 SI 后再无量纲化：

- `dynamic/xyz`: `mm -> m`
- `velocity_speed`: `mm/s -> m/s`
- `path/heat_center_step_distance`、`path/slice_path_length`: `mm -> m`
- `dynamic/Q`: `W/mm^2 -> W/m^3`，转换公式为 `q''' = q'' * 1e6 / heat_source_effective_thickness`

因此训练配置中的 `datasets[].scale` 必须使用 SI：`L0` 为 `m`，`v0` 为 `m/s`，`Q0` 为 `W/m^3`，`K0` 为 `W/(m·°C)`，`rho` 为 `kg/m^3`，`Cp` 为 `J/(kg·°C)`。`heat_source_effective_thickness` 为必填字段，单位 `m`，应按实际工况填写。

详细配置说明见 [configs/README.md](configs/README.md)。

## 训练配置

示例配置位于：

```text
configs/pdgcn_train.example.json
```

关键字段：

- `datasets[0].h5_dir`：训练输入 HDF5 目录。
- `datasets[0].cache_dir`：该训练集共享的静态缓存目录。
- `datasets[0].scale`：SI 无量纲化标尺与 PDE 系数派生参数。

静态缓存默认启用。缓存缺失时，训练入口会使用目录内排序后的第一个 HDF5 文件生成；缓存存在时直接复用。

## 快速开始

运行测试：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe -m pytest
```

使用示例配置训练：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\train_entry.py --config configs\pdgcn_train.example.json
```

使用训练后的 checkpoint 执行多层 PD-GCN + 1D FDM 推理：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py --config configs\pdgcn_train.example.json
```

默认输出：

```text
runs/pdgcn/checkpoint.pt
runs/pdgcn/history.json
runs/pdgcn/cache/case_1/
runs/pdgcn/multilayer_prediction.h5
runs/pdgcn/multilayer_prediction_vtk/
```

多层推理输出 HDF5 包含：

- `temperature`：真实温度，形状 `[time, layer, node, 1]`。
- `temperature_star`：无量纲温度，形状 `[time, layer, node, 1]`。
- `metadata`：JSON 字符串数据集，同时写入根属性副本；记录 checkpoint、源 HDF5、层数、层间距、`dt_star`、FDM 系数和无量纲化参数。

层索引约定为 `layer=0` 是顶层，`layer=L-1` 是底层恒温模具边界。默认仅顶层保留热源，下层热源置零。

推理默认同时输出 ParaView 可读取的 legacy `.vtk` 文件，文件名形如
`temperature_step_000000_layer_000.vtk`。VTK 使用曲面节点三维坐标和计算图 `edge_index`
作为 `POINTS + LINES`，可在 ParaView 中用 `temperature` 或 `temperature_star` 着色。

训练监控快照不再生成 PNG 热力图。需要查看单层曲面温度/残差时，运行：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\visualize_monitor.py --monitor-data runs\pdgcn\metrics\monitor_data.h5
```

该命令会导出 VTK 文件，默认位于 `runs/pdgcn/figures/vtk/`。

## 训练语义

1. 读取 JSON 配置并发现 `h5_dir` 下所有 `.h5`/`.hdf5` 文件。
2. 按文件名自然升序遍历切片，保证训练顺序可复现。
3. 缓存缺失时用第一个文件生成共享静态缓存；后续文件不读取或比较拓扑数据。
4. 每个 HDF5 文件作为独立样本序列训练，不与前一个文件共享温度状态。
5. 每个文件开始时重新初始化温度；若 `warmup_steps > 0`，使用当前最新 PDGCN 参数做连续前向 warmup，不反向传播、不更新参数。
6. 同一 run 内模型参数和优化器状态持续更新。
7. epoch loss 统计为该 epoch 内所有文件所有 TBPTT 窗口损失的平均值。

## 目录说明

```text
configs/      训练配置示例与说明
data/         HDF5 数据读取、SI 转换、无量纲化、特征构建、静态缓存
models/       PDGCN 模型、编码器、处理器、解码器
pde/          PDE residual、边界条件和物理损失
training/     训练入口、TBPTT、单层 rollout、checkpoint、监控和 VTK 快照导出
inference/    独立多层 PD-GCN + 1D FDM 推理入口、HDF5/VTK 输出和单元测试
visualization/ 通用 VTK 导出工具
DesignPlan/   参考资料，只读
PIGNN/        PIGNN 参考仓库，只读
runs/         训练输出，不进入 Git
```

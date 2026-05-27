# inference 模块说明

`inference` 目录负责训练完成后的多层 PDGCN + 厚度方向 1D FDM 推理。该模块只承担推理、输出和推理配置校验职责；单层训练逻辑、TBPTT、checkpoint 保存等仍属于 `training`。

## 主要职责

- 从推理 JSON 配置读取 `inference` 段，并通过 `training_config` 复用训练阶段的尺度参数、模型超参数和 checkpoint。
- 读取单个 HDF5 输入文件，按时间帧构造单层曲面 PyG 图。
- 将训练好的单层 PDGCN 虚拟复制到多层同拓扑曲面图上，执行逐步滚动预测。
- 在厚度方向叠加显式 1D FDM 层间导热，底层强制恒温。
- 输出多层温度场 HDF5；VTK 文件只由离线渲染入口生成。

## 主要文件

- `config.py`：定义 `InferenceRunConfig`，校验层数、层间距、步数、warmup、VTK 输出等配置。
- `fdm.py`：实现厚度方向显式 FDM 系数和层间温度增量，系数为 `k_ratio * dt_star * inverse_pe / layer_spacing_star^2`。
- `multilayer.py`：实现 `rollout_multilayer_fdm(...)`，输入单层图或图工厂，输出形状为 `[time, layer, node, 1]` 的温度序列。
- `io.py`：负责从配置运行推理、加载 checkpoint、构造图、写 HDF5、离线写 VTK 和 metadata。
- `infer_entry.py`：推理命令行入口，只输出 HDF5，默认读取 `configs/pdgcn_infer.example.json`。
- `render_entry.py`：离线渲染入口，从已生成的多层 HDF5 输出合并三维云图 VTK。
- `tests/`：覆盖配置校验、FDM 公式、多层 rollout 和推理输出。

## 推理约定

- `layer=0` 为顶层，`layer=num_layers-1` 为底层模具恒温边界。
- 默认仅顶层保留热源；下层热源置零。
- 多层状态张量形状固定为 `[layer, node, 1]`。
- 输出序列形状固定为 `[time, layer, node, 1]`。
- CUDA 推理默认按较小层批量前向，避免 30 层等大规模场景一次性构造完整多层图导致显存溢出。
- 显式 FDM 稳定性默认要求 `C_n <= 0.5`；如确需跳过检查，配置 `allow_unstable_fdm=true`。
- VTK 输出从真实 `edge_index` 恢复 Gmsh 三角网格面，并写成相邻层连接的 `UNSTRUCTURED_GRID` wedge 体单元；拓扑渲染必须使用全节点，不支持按节点数降采样。

## 与其他模块关系

- 依赖 `training.run_config` 读取训练配置和派生 `PDGCNConfig`。
- 依赖 `training.train_entry` 中的 HDF5 时间步推导与文件发现工具。
- 依赖 `data` 读取 HDF5 并构造图。
- 依赖 `models.PDGCN` 恢复训练好的单层网络。
- 依赖 `visualization` 写 ParaView VTK 文件。

本代码库的 Python 环境为 conda 中的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```

# inference 模块说明

`inference` 目录负责训练完成后的多层 PDGCN + 厚度方向 1D FDM 推理。该模块只承担推理、输出和推理配置校验职责；单层训练逻辑、TBPTT、checkpoint 保存等仍属于 `training`。

## 主要职责

- 从训练 JSON 配置读取 `inference` 段，并复用训练阶段的尺度参数、模型超参数和 checkpoint。
- 读取单个 HDF5 输入文件，按时间帧构造单层曲面 PyG 图。
- 将训练好的单层 PDGCN 虚拟复制到多层同拓扑曲面图上，执行逐步滚动预测。
- 在厚度方向叠加显式 1D FDM 层间导热，底层强制恒温。
- 输出多层温度场 HDF5，并可同步输出 ParaView 可读取的 VTK 文件。

## 主要文件

- `config.py`：定义 `InferenceRunConfig`，校验层数、层间距、步数、warmup、VTK 输出等配置。
- `fdm.py`：实现厚度方向显式 FDM 系数和层间温度增量，系数为 `k_ratio * dt_star * inverse_pe / layer_spacing_star^2`。
- `multilayer.py`：实现 `rollout_multilayer_fdm(...)`，输入单层图或图工厂，输出形状为 `[time, layer, node, 1]` 的温度序列。
- `io.py`：负责从配置运行推理、加载 checkpoint、构造图、写 HDF5、写 VTK 和 metadata。
- `infer_entry.py`：命令行入口，默认读取 `configs/pdgcn_train.example.json`。
- `tests/`：覆盖配置校验、FDM 公式、多层 rollout 和推理输出。

## 推理约定

- `layer=0` 为顶层，`layer=num_layers-1` 为底层模具恒温边界。
- 默认仅顶层保留热源；下层热源置零。
- 多层状态张量形状固定为 `[layer, node, 1]`。
- 输出序列形状固定为 `[time, layer, node, 1]`。
- 显式 FDM 稳定性默认要求 `C_n <= 0.5`；如确需跳过检查，配置 `allow_unstable_fdm=true`。
- VTK 输出使用曲面节点三维坐标和计算图 `edge_index` 写成 `POINTS + LINES`，不构造伪三角曲面。

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

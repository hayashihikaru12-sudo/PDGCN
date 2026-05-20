# `pdgcn_train.example.json` 配置说明

`pdgcn_train.example.json` 是 PDGCN 固定拓扑训练入口的示例配置。运行命令：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\train_entry.py --config configs\pdgcn_train.example.json
```

配置中的相对路径均以配置文件所在目录 `configs/` 为基准解析。

## 顶层结构

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `outputs` | object | 训练产物输出位置，包含 checkpoint 和训练历史。 |
| `datasets` | array | 训练数据集列表。当前训练入口一次运行一个数据集；一个目录可包含多个 `.h5` 切片。 |
| `hyperparameters` | object | 模型、物理损失和训练超参数。 |
| `inference` | object | 可选多层 PD-GCN + 1D FDM 推理配置。 |

## `datasets[]`

| 参数 | 类型 | 示例 | 含义 |
| --- | --- | --- | --- |
| `name` | string | `case_1` | 数据集名称。 |
| `h5_dir` | string | `../HDF5_outputs` | 训练输入 HDF5 目录。目录内 `.h5`/`.hdf5` 文件按自然升序遍历。 |
| `cache_dir` | string | `../runs/pdgcn/cache/case_1` | 共享静态缓存目录。缓存缺失时由排序后的第一个 HDF5 文件生成。 |
| `scale` | object | 见下节 | 无量纲化和 PDE 系数派生所需的 SI 标尺参数。 |
| `scan_velocity` | number 或 null | `null` | 可选真实扫描速度，单位 `m/s`。若设置，必须与 HDF5 根属性 `velocity_speed` 转换到 SI 后一致。 |

静态缓存只保存拓扑、边界节点、节点类型、节点数、边数和特征维度等静态信息。动态帧数据始终从各 HDF5 切片文件读取，并在读取阶段转换为 SI。

## 单位约定

HDF5 原始数据固定使用生成程序的原生单位：

| HDF5 字段 | 原始单位 | 进入模型/PDE 前的转换 |
| --- | --- | --- |
| `dynamic/xyz` | `mm` | 乘 `1e-3` 转为 `m` |
| `dynamic/fiber` | 无量纲方向向量 | 代码内归一化 |
| `dynamic/Q` | 面热流 `W/mm^2` | 乘 `1e6` 转为 `W/m^2`，再除以 `heat_source_effective_thickness` 得到 `W/m^3` |
| `path/heat_center_step_distance` | `mm` | 乘 `1e-3` 转为 `m` |
| `path/slice_path_length` | `mm` | 乘 `1e-3` 转为 `m` |
| 根属性 `velocity_speed` | `mm/s` | 乘 `1e-3` 转为 `m/s` |

`datasets[].scale` 必须按 SI 填写。PDE loss 和无量纲化不接触任何 HDF5 原始 mm 数值。

| 参数 | 含义 | 单位 |
| --- | --- | --- |
| `L0` | 特征长度 | `m` |
| `v0` | 特征扫描速度 | `m/s` |
| `T_amb` | 环境温度基准 | `°C` |
| `delta_T0` | 特征温升；数值上与 K 温差相同 | `°C` |
| `Q0` | 特征体积热源强度 | `W/m^3` |
| `K0` | 特征导热系数 | `W/(m·°C)` |
| `rho` | 密度 | `kg/m^3` |
| `Cp` | 比热容 | `J/(kg·°C)` |
| `heat_source_effective_thickness` | 面热流等效作用厚度，用于 `W/mm^2 -> W/m^3` | `m` |
| `eps` | 可选数值下界，默认 `1e-12` | 无量纲 |

`heat_source_effective_thickness` 是必填工况参数，示例中的 `0.001 m` 仅用于演示，训练前必须替换为实际铺放热源等效作用厚度。

不建议手动配置 `inverse_pe`、`pi_q` 和 `dt_star`。训练入口会根据 `scale` 与首个 HDF5 文件中的路径步长自动派生这些量，即使写入模型配置也会被覆盖。

## 训练语义

- 目录内切片按文件名自然升序训练，保证顺序可复现。
- 每个 HDF5 文件是独立序列，文件之间不继承温度状态。
- 每个文件开始时重新初始化温度；若 `warmup_steps > 0`，使用当前最新模型参数连续前向传播 warmup 帧，不反向传播、不更新参数。
- 模型参数和优化器状态在同一次 run 内持续更新。
- epoch loss 是该 epoch 内所有文件所有 TBPTT 窗口损失的平均值。

## 常用训练参数

| 参数 | 含义 |
| --- | --- |
| `lr` | Adam 学习率。 |
| `epochs` | 最大训练轮数。 |
| `tbptt_window` | TBPTT 时间窗口长度。 |
| `warmup_steps` | 每个文件开头的模型伪时间 warmup 步数。 |
| `grad_clip_norm` | 梯度裁剪阈值，`null` 表示不裁剪。 |
| `loss_threshold` | 提前停止阈值，`null` 表示不按 loss 提前停止。 |
| `device` | 训练设备，`null` 表示自动选择。 |

## 多层推理参数

运行命令：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py --config configs\pdgcn_train.example.json
```

| 参数 | 含义 |
| --- | --- |
| `num_layers` | 多层堆叠层数，必须 `>= 2`。 |
| `layer_spacing` | 层间距，单位 `m`。内部转换为 `layer_spacing / L0`。 |
| `output_path` | 输出 HDF5 路径，保存 `temperature`、`temperature_star` 和 `metadata`。 |
| `dataset_index` | 使用 `datasets[]` 中的哪个数据集，默认 `0`。 |
| `h5_path` | 可选输入 HDF5 文件；为 `null` 时使用数据目录中自然升序的第一个文件。 |
| `steps` | 可选推理步数；为 `null` 时使用输入 HDF5 的全部帧。 |
| `warmup_steps` | 可选推理 warmup 步数；为 `null` 时沿用训练配置。 |
| `bottom_temperature_star` | 底层恒温边界的无量纲温度，默认 `0.0`。 |
| `top_heat_source_only` | 是否仅顶层保留热源，默认 `true`。 |
| `allow_unstable_fdm` | 是否允许显式 FDM 系数 `C_n > 0.5`，默认 `false`。 |
| `write_vtk` | 是否同时写出 ParaView legacy `.vtk` 文件，默认 `true`。 |
| `vtk_output_dir` | 可选 VTK 输出目录；为 `null` 时使用 `<output_path stem>_vtk/`。 |

VTK 文件按时间步和层输出，例如 `temperature_step_000000_layer_000.vtk`。文件使用曲面节点三维坐标和计算图 `edge_index` 写为 `POINTS + LINES`，点标量包含 `temperature` 和 `temperature_star`。

FDM 系数定义为：

```text
C_n = dt_star * inverse_pe * k_ratio / layer_spacing_star^2
```

其中 `k_ratio = K_perp / K_parallel`，对应厚度方向法向导热能力。

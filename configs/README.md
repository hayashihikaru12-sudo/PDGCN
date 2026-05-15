# `pdgcn_train.example.json` 配置说明

`pdgcn_train.example.json` 是 PDGCN 固定拓扑训练入口的示例配置。运行命令：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\train_entry.py --config configs\pdgcn_train.example.json
```

配置中的相对路径均以配置文件所在目录 `configs/` 为基准解析。

## 顶层结构

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `outputs` | object | 训练产物输出位置，包括 checkpoint 和训练历史。 |
| `datasets` | array | 训练数据集列表。当前训练入口一次运行一个数据集；一个数据集目录可包含多个 `.h5` 切片。 |
| `hyperparameters` | object | 模型、物理损失和训练超参数。 |

## `datasets[]`

| 参数 | 类型 | 示例 | 含义 |
| --- | --- | --- | --- |
| `name` | string | `case_1` | 数据集名称。 |
| `h5_dir` | string | `../HDF5_outputs` | 训练输入 HDF5 目录。目录内所有 `.h5`/`.hdf5` 文件按自然升序遍历，每个文件作为独立序列训练。 |
| `cache_dir` | string | `../runs/pdgcn/cache/case_1` | 该训练集共享的静态缓存目录。缓存缺失时用目录内排序后的第一个 HDF5 文件生成；缓存存在时直接复用。 |
| `scale` | object | 见下节 | 无量纲化和 PDE 系数派生所需尺度参数。 |
| `scan_velocity` | number 或 null | `null` | 可选真实扫描速度。若设置，必须与用于派生时间步的 HDF5 根属性 `velocity_speed` 一致。 |

静态缓存只保存拓扑、边界节点、节点类型、节点数、边数和特征维度等静态信息。动态帧数据始终从各 HDF5 切片文件读取。

## `datasets[].scale`

| 参数 | 含义 |
| --- | --- |
| `L0` | 特征长度。 |
| `v0` | 特征扫描速度。 |
| `T_amb` | 环境温度基准。 |
| `delta_T0` | 特征温升尺度。 |
| `Q0` | 特征热源强度。 |
| `K0` | 特征导热系数。 |
| `rho` | 密度。 |
| `Cp` | 比热容。 |
| `eps` | 可选数值下界，默认 `1e-12`。 |

不建议手动配置 `inverse_pe`、`pi_q` 和 `dt_star`。训练入口会根据 `scale` 与首个 HDF5 文件中的路径步长自动派生这些量，即使写入模型配置也会被覆盖。

## 训练语义

- 目录内切片按文件名自然升序训练，保证可复现。
- 每个 HDF5 文件是独立序列，文件之间不继承温度状态。
- 每个文件开始时重新初始化温度；若 `warmup_steps > 0`，使用当前最新模型参数连续前向传播 warmup 帧，不反向传播、不更新参数。
- 模型参数和优化器状态在同一次训练 run 内持续更新。
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

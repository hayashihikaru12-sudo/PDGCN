# inference 推理模块

`inference` 模块用于在训练完成后执行多层曲面热场预测。它复用训练好的单层 PDGCN checkpoint，在推理阶段把同一曲面拓扑虚拟堆叠为多层，并叠加厚度方向 1D FDM 层间传热，最终输出 HDF5 和 ParaView 可视化用 VTK。

## 推理流程

1. 读取推理 JSON 配置，并通过 `training_config` 引用的训练配置解析 `datasets`、`outputs` 和 `hyperparameters`。
2. 选择输入 HDF5：
   - 若 `inference.h5_path` 非空，使用该文件；
   - 否则使用 `datasets[inference.dataset_index].h5_dir` 下自然排序的第一个 HDF5 文件。
3. 读取 HDF5 中的路径步长和扫描速度，结合 `scale` 自动派生 `dt_star`、`inverse_pe` 和 `source_coefficient`。
4. 加载训练 checkpoint：
   - 优先使用 checkpoint metadata 中的 `model_config`；
   - 若 metadata 不包含模型配置，则使用当前 JSON 配置派生的模型配置。
5. 按时间帧构造单层曲面图，节点特征默认包含坐标、纤维方向和当前温度；若 checkpoint 配置启用热源特征，则追加 `q*` 和/或当前步 `ΔT_Q*`。表面热流始终作为独立图字段供显式热源模块读取。
6. 在每个时间步执行多层滚动：
   - 按 `layer_spacing` 沿节点曲面法向偏移下层节点坐标；
   - 按 `layer_fiber_angles_deg` 绕节点法向旋转各层纤维方向；
   - 对顶层施加显式表面热源温升；
   - 使用同一个无源单层 PDGCN 对每层预测面内温度增量；
   - 对网络温度增量执行可选图低通平滑，抑制自回归高频误差；
   - 使用 1D FDM 基于面内更新后的温度计算厚度方向层间传热；
   - 对迎风/侧边节点施加 Dirichlet 边界；
   - 对底层全节点施加恒温边界。
7. 写出多层温度序列 HDF5。VTK 云图不由推理入口生成，需使用 `render_entry.py` 从 HDF5 结果离线渲染。

## 使用方法

在仓库根目录运行：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py --config configs\pdgcn_infer.example.json
```

可选覆盖参数：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py `
  --config configs\pdgcn_infer.example.json `
  --checkpoint ..\runs\pdgcn\checkpoint.pt `
  --h5 ..\case_3_HDF\xxx.h5 `
  --output ..\runs\pdgcn\multilayer_prediction.h5
```

兼容入口仍可使用：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\infer_entry.py --config configs\pdgcn_infer.example.json
```

## 配置字段

推理配置 JSON 顶层需要包含 `training_config` 和 `inference` 段：

```json
{
  "training_config": "pdgcn_train.example.json",
  "inference": {
    "num_layers": 4,
    "layer_spacing": 0.00015,
    "output_path": "../runs/pdgcn/multilayer_prediction.h5",
    "dataset_index": 0,
    "h5_path": null,
    "steps": null,
    "warmup_steps": null,
    "bottom_temperature_star": 0.0,
    "allow_unstable_fdm": false,
    "layer_fiber_angles_deg": [0.0, 45.0, -45.0, 90.0],
    "normal_offset_sign": -1,
    "write_vtk": false,
    "use_pdgcn_inplane": true,
    "pdgcn_inplane_top_layer_only": false,
    "cloud_interval": 20,
    "layer_batch_size": null,
    "delta_smoothing_alpha": 0.2,
    "delta_smoothing_steps": 1,
    "cloud_max_nodes_per_layer": null,
    "vtk_output_dir": null
  }
}
```

字段说明：

- `training_config`：训练配置路径，推理入口会从中读取数据集、尺度参数、模型超参和默认 checkpoint 路径。
- `num_layers`：多层堆叠层数，必须 `>= 2`。
- `layer_spacing`：真实层间距，单位与 `scale.L0` 一致，当前约定为 `m`。
- `output_path`：输出 HDF5 路径。
- `dataset_index`：使用 `datasets[]` 中的哪个数据集。
- `h5_path`：可选输入 HDF5 文件；为 `null` 时自动选取数据目录中的第一个 HDF5。
- `steps`：推理步数；为 `null` 时使用输入 HDF5 的全部帧。
- `warmup_steps`：推理初温 warmup 步数；为 `null` 时沿用训练配置。
- `bottom_temperature_star`：底层恒温边界的无量纲温度，默认 `0.0`。
- `allow_unstable_fdm`：兼容旧显式 FDM 配置的保留字段；当前厚度方向使用 Backward Euler 隐式 FDM，不再依赖该字段跳过稳定性检查。
- `layer_fiber_angles_deg`：每层相对第 0 层纤维方向的旋转角，单位为度；长度需等于 `num_layers`，第 0 项必须为 `0.0`。
- `normal_offset_sign`：法向偏移方向，只能为 `-1` 或 `1`；默认 `-1` 表示 `pos_i = pos_0 - i * layer_spacing * normal`。
- `write_vtk`：兼容旧配置的保留字段；`infer_entry.py` 不再根据该字段生成 VTK。
- `use_pdgcn_inplane`：是否启用无源 PD-GCN 面内输运增量。设为 `false` 时执行“显式热源 + 厚度 FDM only”对照推理。
- `pdgcn_inplane_top_layer_only`：当 `use_pdgcn_inplane=true` 时，是否只在第 0 层启用 PD-GCN 面内输运；下层面内增量置零，仅由厚度 FDM 传热。
- `cloud_interval`：合并三维云图输出间隔，默认 `20`，即输出第 `0, 20, 40, ...` 帧。
- `layer_batch_size`：每次模型前向处理的层数；为 `null` 时 CUDA 默认自动按较小层批量推理，降低 30 层等大规模工况显存占用。
- `delta_smoothing_alpha`：推理端网络增量图低通强度，范围 `[0, 1]`；默认 `0.2`，设为 `0` 可关闭。
- `delta_smoothing_steps`：对 `delta_T_net` 执行的图低通迭代次数，必须非负；默认 `1`，设为 `0` 可关闭。
- `cloud_max_nodes_per_layer`：兼容旧配置的保留字段；拓扑 wedge 渲染必须使用全节点，该字段不会被自动应用。
- `vtk_output_dir`：VTK 输出目录；为 `null` 时使用 `<output_path stem>_vtk/`。

## 隐式 FDM 更新公式

层间 FDM 系数为：

```text
C_n = k_ratio * dt_star * inverse_pe / layer_spacing_star^2
layer_spacing_star = layer_spacing / L0
```

多层温度更新为：

```text
T_src[0] = T_curr[0] + delta_T_source
T_src[k>0] = T_curr[k]
T_inplane = T_src + delta_T_inplane
T_next = implicit_fdm_step(T_inplane)
```

其中：

- `delta_T_source` 来自显式表面热源模块，只作用于顶层；
- `delta_T_inplane` 来自训练好的无源单层 PDGCN，并在 FDM 前按层内图拓扑进行可选低通平滑；
- `implicit_fdm_step` 来自厚度方向 Backward Euler 隐式 1D FDM；
- 默认 `layer=0` 为顶层，`layer=num_layers-1` 为底层。

增量低通只更新内部节点的网络增量，迎风、侧边界和出流边界节点保持原增量；它不会直接平滑最终温度场，也不会作用于 warmup。

隐式厚度步对活动层求解：

```text
(I - C_n D) u_next = u_current
```

其中 `u = T - T_bottom`，`D` 为厚度方向二阶差分算子。该格式对厚度方向导热无条件稳定；`C_n` 仍写入 metadata 作为诊断指标。

## 输出文件

HDF5 输出包含：

- `temperature`：真实温度，形状 `[time, layer, node, 1]`。
- `temperature_star`：无量纲温度，形状 `[time, layer, node, 1]`。
- `metadata`：JSON 字符串数据集，同时写入根属性副本，记录 checkpoint、源 HDF5、层数、层间距、纤维旋转角、法向偏移方向、FDM 系数、增量平滑参数、合并三维云图输出间隔、尺度参数、总推理/渲染耗时和逐帧推理耗时统计。

VTK 输出默认目录为：

```text
<output_path stem>_vtk/
```

VTK 仅由 `render_entry.py` 从已生成的 HDF5 结果离线生成，并按 `cloud_interval` 写出合并三维拓扑云图快照。默认文件名格式：

```text
temperature_step_000000.vtk
```

每个 VTK 文件合并包含所有层：

- 法向偏移后的三维节点坐标；
- 从真实 `edge_index` 恢复的 Gmsh 三角网格面，以及相邻层之间生成的 `UNSTRUCTURED_GRID` wedge 体单元；
- 点标量 `temperature`、`temperature_star`、`layer_index`、`time_step`。

在 ParaView 中打开 `.vtk` 后，可选择 `temperature` 或 `temperature_star` 进行着色。

从已生成的 HDF5 结果离线渲染：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\render_entry.py --cloud-interval 20
```

若未传入 `--prediction`，渲染入口会读取默认配置 `configs/pdgcn_infer.example.json`，并使用其中的 `inference.output_path` 作为 HDF5 输入。可通过 `--config` 指定其他推理配置，或继续用 `--prediction` 手动覆盖。

## 测试

运行推理模块测试：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe -m unittest discover inference\tests
```

运行相关回归测试：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe -m unittest discover training\tests
D:\ProgramData\CondaEnv\PIGNN\python.exe -m unittest discover pde\tests
D:\ProgramData\CondaEnv\PIGNN\python.exe -m unittest discover visualization\tests
```

## 单层推理诊断

当需要排查单层 PD-GCN 本身的训练效果、暂时不考虑多层厚度方向 FDM 时，可使用单层推理入口：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\single_layer_infer_entry.py --config configs\pdgcn_single_layer_infer.example.json
```

单层入口读取 `training_config` 中的 checkpoint、数据集、尺度参数和 warmup 默认值，只推进一层曲面温度：

```text
T_next = T_current + delta_T_source + delta_T_inplane
```

输出 HDF5 默认写到：

```text
runs/pdgcn/single_layer_prediction.h5
```

输出文件会先复制原始输入 HDF5，再在 `prediction/pdgcn_single_layer/temperature` 下按帧写入预测真实温度，因此原始 HDF5 不会被原地修改。该数据集形状为 `[time, node, 1]`，组织方式与 `fem/temperature` 保持一致；不再额外写入无量纲温度、误差场、metrics 或 metadata。

若 `write_vtu=true`，入口会按 `vtu_interval` 直接输出 ParaView 可读取的单层 `.vtu` 曲面文件，默认目录为：

```text
<output_path stem>_vtu/
```

文件名形如：

```text
temperature_step_000000.vtu
```

批量推理可设置 `single_layer_inference.batch_mode=true`，或命令行传入 `--batch --h5-dir <输入目录> --output-dir <输出目录>`。批量模式会按自然升序遍历输入目录直属 `.h5`/`.hdf5` 文件，增强 HDF5 写入 `<output_dir>/pre_<原文件名>`，VTU 写入 `<output_dir>/pre_<原stem>_vtu/`。单个文件失败时不会中止整个批次，入口会在最后打印成功和失败清单。

可选覆盖参数：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\single_layer_infer_entry.py `
  --config configs\pdgcn_single_layer_infer.example.json `
  --checkpoint ..\runs\pdgcn\checkpoint.pt `
  --h5 ..\case_3_HDF\xxx.h5 `
  --output ..\runs\pdgcn\single_layer_prediction.h5 `
  --vtu-interval 5 `
  --mode both
```

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
    "h5_dir": "../case_fastsmall",
    "output_dir": "../runs/pdgcn/multilayer_batch",
    "output_prefix": "pre_",
    "batch_mode": false,
    "prediction_group_path": "prediction/pdgcn_multilayer",
    "steps": null,
    "warmup_steps": null,
    "bottom_temperature_star": 0.0,
    "allow_unstable_fdm": false,
    "layer_fiber_angles_deg": [0.0, 45.0, -45.0, 90.0],
    "normal_offset_sign": -1,
    "write_vtk": false,
    "use_pdgcn_inplane": true,
    "pdgcn_inplane_top_layer_only": false,
    "use_alternating_order_average": false,
    "fdm_k_ratio_scale": 0.42,
    "fdm_layer_interface_scales": [0.7, 0.7, 2.0, 2.5, 2.5, 2.0, 2.0, 2.0, 1.0],
    "fdm_top_surface_loss_gamma_dt": 0.014,
    "fdm_top_surface_loss_velocity_exponent": 3.4,
    "fdm_top_surface_loss_reference_velocity_star": 1.0,
    "fin_cooling_enabled": true,
    "fin_cooling_mode": "r_char",
    "fin_cooling_r_char_star": 0.034,
    "fin_cooling_gamma_star": null,
    "fin_cooling_beta_h": 3.0,
    "fin_cooling_skip_top_layers": 0,
    "fin_cooling_layer_profile": "uniform",
    "fin_cooling_layer_profile_strength": 0.0,
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
- `h5_dir`：批量模式输入 HDF5 目录；为 `null` 时使用训练配置中的数据集目录。
- `output_dir`：批量模式输出目录。
- `output_prefix`：批量输出文件名前缀，默认 `pre_`。
- `batch_mode`：是否启用批处理模式。
- `prediction_group_path`：多层预测结果组路径，默认 `prediction/pdgcn_multilayer`。
- `steps`：推理步数；为 `null` 时使用输入 HDF5 的全部帧。
- `warmup_steps`：推理初温 warmup 步数；为 `null` 时沿用训练配置。
- `bottom_temperature_star`：底层恒温边界的无量纲温度，默认 `0.0`。
- `allow_unstable_fdm`：兼容旧显式 FDM 配置的保留字段；当前厚度方向使用 Backward Euler 隐式 FDM，不再依赖该字段跳过稳定性检查。
- `layer_fiber_angles_deg`：每层相对第 0 层纤维方向的旋转角，单位为度；长度需等于 `num_layers`，第 0 项必须为 `0.0`。
- `normal_offset_sign`：法向偏移方向，只能为 `-1` 或 `1`；默认 `-1` 表示 `pos_i = pos_0 - i * layer_spacing * normal`。
- `write_vtk`：是否在推理后立即生成 VTK；为 `false` 时只保存 HDF5。
- `use_pdgcn_inplane`：是否启用无源 PD-GCN 面内输运增量。设为 `false` 时执行“显式热源 + 厚度 FDM only”对照推理。
- `pdgcn_inplane_top_layer_only`：当 `use_pdgcn_inplane=true` 时，是否只在第 0 层启用 PD-GCN 面内输运；下层面内增量置零，仅由厚度 FDM 传热。
- `use_alternating_order_average`：是否启用方案 B。设为 `true` 时，同一步分别计算“面内 PD-GCN → 厚度 FDM”和“厚度 FDM → 面内 PD-GCN”两条路径并取算术平均；默认 `false` 保持旧顺序分裂行为。
- `fdm_k_ratio_scale`：推理阶段厚度 FDM 使用的 `k_ratio` 缩放系数，默认代码值为 `1.0`。当前多层示例采用 `0.42`，只作用于推理厚度 FDM，不修改 checkpoint 中的模型参数。
- `fdm_layer_interface_scales`：层间界面导热倍率，标量或长度为 `num_layers - 1` 的数组；第 `i` 项对应 layer `i` 到 layer `i+1` 的界面，最后一项对应倒数第二层到底部恒温层。
- `fdm_top_surface_loss_gamma_dt`：厚度步后的顶层表面对流损失分裂项，阻尼 `layer=0` 相对底温的温升；默认 `0.0` 保持旧行为。
- `fdm_top_surface_loss_velocity_exponent`：顶层表面对流损失的速度缩放指数。大于 `0` 时按 `(fdm_top_surface_loss_reference_velocity_star / v_scan*)^exponent` 缩放 `fdm_top_surface_loss_gamma_dt`，当前多层示例采用 `3.4`。
- `fdm_top_surface_loss_reference_velocity_star`：顶层表面对流损失速度缩放的参考无量纲速度，必须为正；当前多层示例采用 `1.0`。
- `fin_cooling_enabled`：是否启用瞬态散热片等效横向冷却项；设为 `false` 时退化为原隐式 FDM。
- `fin_cooling_mode`：等效横向冷却强度来源，支持 `r_char`、`beta_h` 和 `direct`。当前示例使用 `r_char`。
- `fin_cooling_r_char_star`：`r_char` 模式下的无量纲横向特征尺度，按 `gamma_star = inverse_pe / r_char_star^2` 计算冷却强度。
- `fin_cooling_gamma_star`：`direct` 模式下直接给定的无量纲冷却强度，可为标量或 active layer 数组。
- `fin_cooling_beta_h`：`beta_h` 模式下的散热片剖面弯曲度；在 `r_char` 模式下主要用于缺省 `r_char_star` 时的兼容推导。
- `fin_cooling_skip_top_layers`：保留瞬态散热片模型时必须为 `0`，即所有 active layers 都施加等效横向冷却项；非零值会在配置解析时报错。
- `cloud_interval`：合并三维云图输出间隔，默认 `20`，即输出第 `0, 20, 40, ...` 帧。
- `layer_batch_size`：每次模型前向处理的层数；为 `null` 时 CUDA 默认自动按较小层批量推理，降低 30 层等大规模工况显存占用。
- `delta_smoothing_alpha`：推理端网络增量图低通强度，范围 `[0, 1]`；默认 `0.2`，设为 `0` 可关闭。
- `delta_smoothing_steps`：对 `delta_T_net` 执行的图低通迭代次数，必须非负；默认 `1`，设为 `0` 可关闭。
- `cloud_max_nodes_per_layer`：兼容旧配置的保留字段；拓扑 wedge 渲染必须使用全节点，该字段不会被自动应用。
- `vtk_output_dir`：VTK 输出目录；为 `null` 时使用 `<output_path stem>_vtk/`。

## 隐式 FDM 更新公式

层间 FDM 系数为：

```text
effective_k_ratio = model_k_ratio * fdm_k_ratio_scale
C_n = effective_k_ratio * dt_star * inverse_pe / layer_spacing_star^2
layer_spacing_star = layer_spacing / L0
gamma_star = inverse_pe / fin_cooling_r_char_star^2          # fin_cooling_mode = "r_char"
gamma_star * dt_star = (fin_cooling_beta_h / (num_layers - 1))^2 * C_n  # fin_cooling_mode = "beta_h"
```

多层温度更新为：

```text
T_src[0] = T_curr[0] + delta_T_source
T_src[k>0] = T_curr[k]
T_inplane = T_src + delta_T_inplane
T_next = implicit_fdm_step(T_inplane)
```

若 `use_alternating_order_average=true`，则同一步改为：

```text
T_A = implicit_fdm_step(T_src + delta_T_inplane(T_src))
T_B = implicit_fdm_step(T_src)
T_B = T_B + delta_T_inplane(T_B)
T_next = 0.5 * (T_A + T_B)
```

其中：

- `delta_T_source` 来自显式表面热源模块，只作用于顶层；
- `delta_T_inplane` 来自训练好的无源单层 PDGCN，并在叠加到对应路径温度前按层内图拓扑进行可选低通平滑；
- `implicit_fdm_step` 来自厚度方向 Backward Euler 隐式 1D FDM；
- 默认 `layer=0` 为顶层，`layer=num_layers-1` 为底层。

增量低通只更新内部节点的网络增量，迎风、侧边界和出流边界节点保持原增量；它不会直接平滑最终温度场，也不会作用于 warmup。

隐式厚度步对活动层求解：

```text
(I - C_n D) u_next = u_current
```

启用 `fin_cooling_enabled=true` 时，活动层矩阵改为：

```text
(I - C_n D + gamma_star * dt_star * I) u_next = u_current
```

其中 `u = T - T_bottom`，`D` 为厚度方向二阶差分算子。该格式对厚度方向导热无条件稳定；`C_n`、`model_k_ratio`、`fdm_k_ratio_scale`、`fdm_effective_k_ratio`、`beta_h`、`fin_cooling_skip_top_layers`、`gamma_star`、`gamma_star * dt_star` 和 `thickness_model` 会写入 metadata 作为诊断指标。当前保留瞬态散热片模型时要求 `fin_cooling_skip_top_layers=0`，所有 active layers 的对角线都加入 `gamma_star * dt_star`。

若设置 `fdm_layer_interface_scales`，矩阵中的每个层间界面使用 `C_n * scale_i`，序列长度为 `num_layers - 1`，最后一项对应倒数第二层到底部恒温层的界面。`fdm_top_surface_loss_gamma_dt` 是厚度步后的顶层表面对流损失分裂项，只阻尼 `layer=0` 相对底温的温升，默认 `0.0` 保持旧行为；若 `fdm_top_surface_loss_velocity_exponent > 0`，该损失会按无量纲扫描速度做缩放。

## 输出文件

多层 HDF5 输出会先复制输入 HDF5，再在副本中新增 `prediction/pdgcn_multilayer`。输入中已有的 `fem/`、根级 `multilayer/` 和 `mesh/` 会原样保留，不参与校验。

预测结果组包含：

- `prediction/pdgcn_multilayer/temperature`：真实温度，形状 `[time, layer, node, 1]`。
- `prediction/pdgcn_multilayer/temperature_star`：无量纲温度，形状 `[time, layer, node, 1]`。
- `prediction/pdgcn_multilayer/top_temperature`：第 0 层真实温度，形状 `[time, node, 1]`。
- `prediction/pdgcn_multilayer/time`：物理时间，单位秒。
- `prediction/pdgcn_multilayer/timing/*`：逐步推理、渲染和总耗时。
- `prediction/pdgcn_multilayer/multilayer/*`：与样例根级 `multilayer/` 对齐的多层坐标、层数、层间距、纤维角、法向偏移方向和底温。
- `prediction/pdgcn_multilayer/metadata_json`：JSON 字符串，记录 checkpoint、源 HDF5、配置、尺度参数和时间统计摘要。

VTK 输出默认目录为：

```text
<output_path stem>_vtk/
```

当 `write_vtk=true` 时，`infer_entry.py` 会在 HDF5 写出后立即按 `cloud_interval` 生成合并三维拓扑云图快照。也可以继续用 `render_entry.py` 对已生成的 HDF5 离线渲染。默认文件名格式：

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

输出文件会先复制原始输入 HDF5，再在 `prediction/pdgcn_single_layer/temperature` 下按帧写入预测真实温度，因此原始 HDF5 不会被原地修改。该数据集形状为 `[time, node, 1]`，组织方式与 `fem/temperature` 保持一致；同组还会写入 `time` 和 `timing/` 记录物理时间与推理/渲染耗时。

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

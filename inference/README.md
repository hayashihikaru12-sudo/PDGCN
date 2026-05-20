# inference 推理模块

`inference` 模块用于在训练完成后执行多层曲面热场预测。它复用训练好的单层 PDGCN checkpoint，在推理阶段把同一曲面拓扑虚拟堆叠为多层，并叠加厚度方向 1D FDM 层间传热，最终输出 HDF5 和 ParaView 可视化用 VTK。

## 推理流程

1. 读取训练 JSON 配置，解析其中的 `datasets`、`outputs`、`hyperparameters` 和 `inference`。
2. 选择输入 HDF5：
   - 若 `inference.h5_path` 非空，使用该文件；
   - 否则使用 `datasets[inference.dataset_index].h5_dir` 下自然排序的第一个 HDF5 文件。
3. 读取 HDF5 中的路径步长和扫描速度，结合 `scale` 自动派生 `dt_star`、`inverse_pe` 和 `pi_q`。
4. 加载训练 checkpoint：
   - 优先使用 checkpoint metadata 中的 `model_config`；
   - 若 metadata 不包含模型配置，则使用当前 JSON 配置派生的模型配置。
5. 按时间帧构造单层曲面图，节点特征包含坐标、纤维方向、当前温度和热源。
6. 在每个时间步执行多层滚动：
   - 使用同一个单层 PDGCN 对每层预测面内温度增量；
   - 默认仅顶层保留热源，下层热源置零；
   - 使用 1D FDM 计算厚度方向层间传热；
   - 对迎风/侧边节点施加 Dirichlet 边界；
   - 对底层全节点施加恒温边界。
7. 写出多层温度序列 HDF5，并在启用时写出逐时间步、逐层 VTK 文件。

## 使用方法

在仓库根目录运行：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py --config configs\pdgcn_train.example.json
```

可选覆盖参数：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py `
  --config configs\pdgcn_train.example.json `
  --checkpoint runs\pdgcn\checkpoint.pt `
  --h5 case_1_HDF\xxx.h5 `
  --output runs\pdgcn\multilayer_prediction.h5
```

兼容入口仍可使用：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\infer_entry.py --config configs\pdgcn_train.example.json
```

## 配置字段

训练配置 JSON 顶层需要包含 `inference` 段：

```json
{
  "inference": {
    "num_layers": 4,
    "layer_spacing": 0.00015,
    "output_path": "../runs/pdgcn/multilayer_prediction.h5",
    "dataset_index": 0,
    "h5_path": null,
    "steps": null,
    "warmup_steps": null,
    "bottom_temperature_star": 0.0,
    "top_heat_source_only": true,
    "allow_unstable_fdm": false,
    "write_vtk": true,
    "vtk_output_dir": null
  }
}
```

字段说明：

- `num_layers`：多层堆叠层数，必须 `>= 2`。
- `layer_spacing`：真实层间距，单位与 `scale.L0` 一致，当前约定为 `m`。
- `output_path`：输出 HDF5 路径。
- `dataset_index`：使用 `datasets[]` 中的哪个数据集。
- `h5_path`：可选输入 HDF5 文件；为 `null` 时自动选取数据目录中的第一个 HDF5。
- `steps`：推理步数；为 `null` 时使用输入 HDF5 的全部帧。
- `warmup_steps`：推理初温 warmup 步数；为 `null` 时沿用训练配置。
- `bottom_temperature_star`：底层恒温边界的无量纲温度，默认 `0.0`。
- `top_heat_source_only`：是否仅顶层保留热源，默认 `true`。
- `allow_unstable_fdm`：是否允许显式 FDM 系数 `C_n > 0.5`。
- `write_vtk`：是否输出 ParaView VTK 文件，默认 `true`。
- `vtk_output_dir`：VTK 输出目录；为 `null` 时使用 `<output_path stem>_vtk/`。

## FDM 更新公式

层间显式 FDM 系数为：

```text
C_n = k_ratio * dt_star * inverse_pe / layer_spacing_star^2
layer_spacing_star = layer_spacing / L0
```

多层温度更新为：

```text
T_next = T_curr + delta_T_net + beta * T_curr * dt_star + delta_T_fdm
```

其中：

- `delta_T_net` 来自训练好的单层 PDGCN；
- `delta_T_fdm` 来自厚度方向 1D FDM；
- `beta` 对应模型配置中的 `thermal_loss_beta`；
- 默认 `layer=0` 为顶层，`layer=num_layers-1` 为底层。

若 `C_n > 0.5` 且 `allow_unstable_fdm=false`，推理会直接报错，避免显式差分不稳定。

## 输出文件

HDF5 输出包含：

- `temperature`：真实温度，形状 `[time, layer, node, 1]`。
- `temperature_star`：无量纲温度，形状 `[time, layer, node, 1]`。
- `metadata`：JSON 字符串数据集，同时写入根属性副本，记录 checkpoint、源 HDF5、层数、层间距、FDM 系数和尺度参数。

VTK 输出默认目录为：

```text
<output_path stem>_vtk/
```

文件名格式：

```text
temperature_step_000000_layer_000.vtk
temperature_step_000000_layer_001.vtk
```

每个 VTK 文件包含：

- 曲面节点三维坐标；
- 计算图 `edge_index` 写成的线单元；
- 点标量 `temperature`、`temperature_star`、`layer_index`、`time_step`。

在 ParaView 中打开 `.vtk` 后，可选择 `temperature` 或 `temperature_star` 进行着色。

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

# 单层预测推理模式代码说明

> 运行入口：`inference/single_layer_infer_entry.py`
>
> 核心实现：`inference/single_layer.py`

---

## 一、概述

`single_layer_infer_entry.py` 负责单层曲面 PD-GCN 的离线推理与 VTU 诊断输出。它从拆分式推理 JSON 配置读取参数，加载训练好的单层 PD-GCN 模型，对单个或目录内批量 HDF5 输入文件按时间帧执行推理，并将预测温度场写入输出 HDF5。

当前入口不会原地修改输入 HDF5。单文件模式会复制输入 HDF5 到 `output_path` 后追加预测组；批量模式会把每个输入文件复制到 `output_dir/pre_<原文件名>`，再追加预测组。预测结果默认写入：

```text
prediction/pdgcn_single_layer
```

当前 HDF5 落盘保存自回归 rollout 的真实温度数据集，并在同一预测组中记录物理时间轴和推理/渲染耗时。配置项 `mode` 和命令行 `--mode` 保留为兼容字段，但不再额外写入 teacher-forcing、误差或 metrics 字段。

命令行仍可通过 `--mode` 覆盖配置：

```bash
python -m inference.single_layer_infer_entry \
  --config configs/pdgcn_single_layer_infer.example.json \
  --mode autoregressive
```

---

## 二、单步推理的物理流程

HDF5 落盘使用 `rollout_single_layer_static` 中的自回归单步前向计算流程。

单步更新式为：

```text
T*_next = apply_dirichlet( apply_dirichlet(T*_in + ΔT*_source) + PDGCN(graph) )
```

分解为 4 个子步骤：

| 步骤 | 代码操作 | 物理含义 |
|------|----------|----------|
| ① 显式热源 | `T*_in + graph_explicit_source_delta(graph, model.config)` | 读取 HDF5 `dynamic/Q` 表面热流，按 `q''` → `W/m²` → 无量纲温升 ΔT*，叠加到输入温度 |
| ② Dirichlet 边界 | `apply_dirichlet_boundary(..., boundary_nodes, dirichlet_temperature_star)` | 强制边界节点温度为固定值（默认 0.0 = 环境温度 T_amb） |
| ③ PD-GCN 前向 | `source + model(graph)` | 无源 PD-GCN 计算曲面内对流、扩散和纤维各向异性输运增量 |
| ④ Dirichlet 边界 | `apply_dirichlet_boundary(...)` | 再次强制边界约束，得到最终 `T*_next` |

其中：
- `graph` 的节点特征默认为 `[x*, y*, z*, fx, fy, fz, T*]`；若 checkpoint 配置启用热源特征，则追加 `q*` 和/或当前步 `ΔT_Q*`
- `dirichlet_temperature_star` 默认为 `0.0`，取自 `model.config`
- warmup 阶段（可选）在 rollout 开始前用 `pseudo_time_relax_initial_temperature` 对初始温度做伪时间弛豫

---

## 三、自回归推理语义

```
初始温度 T*_0 → [步骤①→④] → T*_1 → [步骤①→④] → T*_2 → ... → T*_(steps-1)
```

- **输入温度来源**：上一步模型预测的温度
  - 第 0 步使用 `feature_builder.initial_temperature()`（来自初始条件）
  - 若 `warmup_steps > 0`，在 rollout 前对初始温度做伪时间弛豫
- 模型输出直接作为下一帧的输入——**误差随步数累积**
- 可用于评估模型的**长期 rollout 稳定性**
- 若要与 FEM 对比，可直接读取同一输出 HDF5 中原始保留的 `fem/temperature`，并与 `prediction/pdgcn_single_layer/temperature` 在外部分析脚本中对齐

---

## 四、HDF5 输出数据集

单层推理在输出 HDF5 副本中新增预测组：

```text
prediction/pdgcn_single_layer
```

| 数据集 | 形状 | 含义 |
|--------|------|------|
| `temperature` | `[steps, N, 1]` | 逐帧写入的 PD-GCN 预测真实温度 |
| `time` | `[steps]` | 逐帧物理时间，单位秒 |
| `timing/*` | 逐步数组或标量 | 推理、渲染和总耗时统计 |

该组织方式参考 FEM 数据的 `fem/temperature`：第一维是时间帧，第二维是节点，第三维保留单通道温度值。输出文件仍保留源 HDF5 中的 `dynamic`、`fem`、attrs 等原始内容；推理入口不再写入 `temperature_star`、`frame_index`、FEM 对齐副本、误差场、`teacher_forcing` 子组或 `metrics` 子组。

---

## 五、VTU 渲染输出的监控场

VTU 文件按 `vtu_interval` 间隔采样输出，写入同一 vtu 输出目录，通过文件名前缀区分来源：

- `INF_temperature_step_<QV>_<step>.vtu`：PDGCN 推理预测温度场
- `FEM_temperature_step_<QV>_<step>.vtu`：原始 FEM 有限元温度场（当配置 `write_fem_vtu=true` 且 HDF5 含 `fem/temperature` 时同步生成；缺失则自动跳过、不报错）

其中 `<QV>` 为工况标记，从输出 HDF5 副本根级 attrs 读取并以 case 文件名风格（小数点写作 `p`）拼入文件名，便于一眼区分不同热源强度与扫描速度的可视化结果：

- `Q` 取自 `heat_source_qmax`（原始单位 W/mm²，例 `0.6666667` → `Q0p666667`）；缺失时省略 `Q` token。
- `V` 取自 `velocity_speed`（原始单位 mm/s，例 `25` → `V25`）；缺失时省略 `V` token。
- 两个 token 都缺失时回退为不带标记的 `INF_temperature_step_<step>.vtu`。

示例：`INF_temperature_step_Q0p666667_V25_000020.vtu`、`FEM_temperature_step_Q0p666667_V25_000020.vtu`。

INF 与 FEM 两类文件使用完全一致的曲面网格（节点坐标与三角连接），温度标量字段都命名为 `temperature`，因此可在 ParaView 中对 `INF_*` 与 `FEM_*` 套用同一 color map 直接并排对比。采样步集合相同（同 `vtu_interval`），保证逐步对齐。

每个 VTU 文件包含以下 point data（节点标量场）：

| 场名 | 含义 | INF | FEM |
|------|------|:---:|:---:|
| `temperature` | 有量纲温度（INF 为预测值，FEM 为原始有限元值） | ✅ | ✅ |
| `time_step` | 时间步索引 | ✅ | ✅ |
| `fem_valid_mask` | FEM 有效节点掩码（仅当 HDF5 含 `fem/valid_mask`） | — | 可选 |

VTU 使用 Gmsh 三角网格面 + `UNSTRUCTURED_GRID` 格式，可在 ParaView 中直接打开查看。渲染旧版预测 HDF5 时，如果文件中仍存在 `temperature_star`，INF VTU 会继续兼容读取并追加该字段；新推理文件默认只写出 `temperature` 与 `time_step`。

---

## 六、命令行返回的汇总指标

`main()` 函数执行完毕后打印的汇总字段：

| 字段 | 含义 | 来源 |
|------|------|------|
| `steps` | 推理总帧数 | 配置或 HDF5 帧数 |
| `mode` | 实际使用的推理模式 | 配置 |
| `inference_seconds` | 纯推理耗时（秒） | `time.perf_counter` 计时 |
| `render_seconds` | VTU 渲染耗时（秒） | `time.perf_counter` 计时 |
| `total_seconds` | 总耗时 = inference + render | 计算值 |
| `rendered_steps` | 实际渲染了哪些帧的索引列表 | 按 `vtu_interval` 采样 |
| `average_inference_seconds` | 平均单步推理耗时 | 推理过程统计 |
| `max_inference_seconds` | 最大单步推理耗时 | 推理过程统计 |
| `min_inference_seconds` | 最小单步推理耗时 | 推理过程统计 |

---

## 七、与 FEM 数据的对齐方式

输出 HDF5 是源文件副本，因此若输入文件含有 `fem/temperature`，输出文件会同时保留原始 FEM 温度和新增预测温度：

```text
fem/temperature
prediction/pdgcn_single_layer/temperature
```

两者均采用 `[time, node, 1]` 组织形式，后续分析脚本可直接按帧索引和节点索引对齐后计算误差、RMSE、MAE 或其他自定义指标。推理入口本身不再把这些对比指标写回 HDF5。

此外，VTU 渲染阶段会按 `write_fem_vtu` 配置同步生成 `FEM_temperature_step_XXXXXX.vtu`，与推理 `INF_temperature_step_XXXXXX.vtu` 同目录、同网格，可在 ParaView 中直接可视化对比（详见第五节）。

---

## 八、配置示例

```json
{
  "training_config": "pdgcn_train.example.json",
  "single_layer_inference": {
    "output_path": "../runs/pdgcn/single_layer_prediction.h5",
    "dataset_index": 0,
    "h5_path": "../case1_validTheta_50X50/case1_Q0p666667_V25_dt0p05_F91_fem.h5",
    "h5_dir": "../case1_validTheta_50X50",
    "output_dir": "../runs/pdgcn/single_layer_batch",
    "output_prefix": "pre_",
    "prediction_group_path": "prediction/pdgcn_single_layer",
    "batch_mode": false,
    "steps": null,
    "warmup_steps": 0,
    "mode": "both",
    "write_vtu": true,
    "vtu_interval": 20,
    "vtu_output_dir": null,
    "fem_temperature_dataset": "fem/temperature",
    "fem_valid_mask_dataset": "fem/valid_mask"
  }
}
```

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `training_config` | string | 指向训练配置 JSON 的相对路径 |
| `output_path` | string | 输出 HDF5 路径 |
| `dataset_index` | int | 使用训练配置中第几个 dataset 的尺度参数 |
| `h5_path` | string/null | 输入 HDF5 路径，null 时自动从训练数据目录选取第一个 |
| `h5_dir` | string/null | 批量模式输入目录，null 时使用训练配置中的数据目录 |
| `output_dir` | string/null | 批量模式输出目录，增强 HDF5 与 VTU 均写入该目录 |
| `output_prefix` | string | 批量输出文件名前缀，默认 `pre_` |
| `prediction_group_path` | string | 预测结果组路径，默认 `prediction/pdgcn_single_layer` |
| `batch_mode` | bool | 是否启用目录级批量推理 |
| `steps` | int/null | 推理帧数，null 时使用 HDF5 全部帧 |
| `warmup_steps` | int/null | 自回归初始温度伪时间弛豫步数，null 时使用训练配置值 |
| `mode` | string | 兼容字段，可为 `"autoregressive"` / `"teacher_forcing"` / `"both"`；HDF5 仅保存自回归预测温度 |
| `write_vtu` | bool | 是否输出 VTU 可视化文件 |
| `vtu_interval` | int | VTU 输出间隔（每 N 步输出一个） |
| `vtu_output_dir` | string/null | 单文件 VTU 输出目录，null 时自动在 output 旁创建 `*_vtu` 目录；批量模式固定写入 `output_dir/pre_<原stem>_vtu/`，`INF_*` 与 `FEM_*` VTU 同目录保存 |
| `fem_temperature_dataset` | string | HDF5 中 FEM 温度数据集路径 |
| `fem_valid_mask_dataset` | string/null | HDF5 中 FEM 有效掩码数据集路径，null 时使用全 1 掩码 |
| `write_fem_vtu` | bool | 是否同步生成原始 FEM 温度场的 `FEM_temperature_step_XXXXXX.vtu`，默认 `true`；HDF5 不含 `fem/temperature` 时自动跳过 |

批量推理命令示例：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\single_layer_infer_entry.py `
  --config configs\pdgcn_single_layer_infer.example.json `
  --batch `
  --h5-dir ..\case_3_HDF `
  --output-dir ..\runs\pdgcn\single_layer_batch `
  --output-prefix pre_
```

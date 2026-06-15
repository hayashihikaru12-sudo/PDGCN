# 单层预测推理模式代码说明

> 运行入口：`inference/single_layer_infer_entry.py`
>
> 核心实现：`inference/single_layer.py`

---

## 一、概述

`single_layer_infer_entry.py` 负责单层曲面 PD-GCN 的离线推理与 VTU 诊断输出。它从拆分式推理 JSON 配置读取参数，加载训练好的单层 PD-GCN 模型，对单个 HDF5 输入文件按时间帧执行推理，并将预测温度场写入 HDF5。

支持三种推理模式，由配置项 `mode` 控制：

| 配置值 | 含义 |
|--------|------|
| `"autoregressive"` | 仅自回归 rollout |
| `"teacher_forcing"` | 仅教师强制单步评估 |
| `"both"` | 同时运行两种模式 |

命令行可通过 `--mode` 覆盖配置：

```bash
python -m inference.single_layer_infer_entry \
  --config configs/pdgcn_single_layer_infer.example.json \
  --mode autoregressive
```

---

## 二、单步推理的物理流程

两种模式共享同一个单步前向计算流程。核心循环在 `rollout_single_layer_static`（自回归）和 `_write_teacher_forcing_group`（教师强制）中。

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
- `graph` 的节点特征为 `[x*, y*, z*, fx, fy, fz, T*]`，热源 **不进入**节点特征
- `dirichlet_temperature_star` 默认为 `0.0`，取自 `model.config`
- warmup 阶段（可选）在 rollout 开始前用 `pseudo_time_relax_initial_temperature` 对初始温度做伪时间弛豫

---

## 三、两种推理模式

### 3.1 Autoregressive（自回归）模式

```
初始温度 T*_0 → [步骤①→④] → T*_1 → [步骤①→④] → T*_2 → ... → T*_(steps-1)
```

- **输入温度来源**：上一步模型预测的温度
  - 第 0 步使用 `feature_builder.initial_temperature()`（来自初始条件）
  - 若 `warmup_steps > 0`，在 rollout 前对初始温度做伪时间弛豫
- 模型输出直接作为下一帧的输入——**误差随步数累积**
- 可用于评估模型的**长期 rollout 稳定性**

### 3.2 Teacher Forcing（教师强制）模式

```
FEM T*_0 → [步骤①→④] → T*_pred,1  vs  FEM T*_1（标签）
FEM T*_1 → [步骤①→④] → T*_pred,2  vs  FEM T*_2（标签）
...
FEM T*_(steps-2) → [步骤①→④] → T*_pred,(steps-1)  vs  FEM T*_(steps-1)（标签）
```

- **输入温度来源**：每步使用 FEM 真实温度 `fem/temperature` 作为输入（不从预测结果递推）
- 只做一步前向预测，与 FEM 下一帧对比——误差不累积
- 共执行 `steps - 1` 个 transition（要求 `steps ≥ 2`）
- 用于**独立评估单步模型精度**（即单步 PDE 残差的预测误差）
- **必须** HDF5 中包含 `fem/temperature` 数据集，否则报错

---

## 四、HDF5 输出中的监控数据集

### 4.1 Autoregressive 模式（根组）

| 数据集 | 形状 | 含义 |
|--------|------|------|
| `temperature_star` | `[steps, N, 1]` | PD-GCN 预测的无量纲温度 T* |
| `temperature` | `[steps, N, 1]` | 转换回物理单位的有量纲温度（°C 或 K） |
| `fem_temperature` | `[steps, N, 1]` | 对应帧的 FEM 真实温度（仅当 HDF5 含 FEM 数据时写入） |
| `fem_valid_mask` | `[steps, N, 1]` | FEM 有效节点掩码（仅当 HDF5 含 FEM 数据时写入） |
| `temperature_error` | `[steps, N, 1]` | 预测误差 = `temperature - fem_temperature`（仅当 HDF5 含 FEM 数据时写入） |

### 4.2 Teacher Forcing 模式（`teacher_forcing` 组）

| 数据集 | 形状 | 含义 |
|--------|------|------|
| `temperature_star` | `[steps-1, N, 1]` | 单步预测的无量纲温度 T* |
| `temperature` | `[steps-1, N, 1]` | 单步预测的有量纲温度 |
| `fem_temperature` | `[steps-1, N, 1]` | 目标帧的 FEM 真实温度（标签 / ground truth） |
| `fem_valid_mask` | `[steps-1, N, 1]` | 目标帧的有效节点掩码 |
| `temperature_error` | `[steps-1, N, 1]` | 单步预测误差 = `temperature_pred - temperature_fem` |
| `frame_index` | `[steps-1]` | 目标帧索引 `[1, 2, ..., steps-1]` |

### 4.3 Both 模式

同时写入上述两套数据集（根组 + `teacher_forcing` 组）。VTU 渲染时，autoregressive 帧会额外附带 teacher forcing 的对比场。

### 4.4 全局元数据

所有模式均在 HDF5 根组写入 `metadata` 数据集（JSON 字符串），包含：

- 推理配置参数：`mode`、`num_layers`、`vtu_interval`、`warmup_steps` 等
- 模型配置：`model_config`（网络结构超参数）
- 尺度参数：`scale_params`（L0, T0, T_amb, ΔT0, v0 等）
- 时间信息：`hdf5_timing`（dt, scan_velocity 等）
- Checkpoint 信息：`checkpoint_path`、`checkpoint_epoch`
- 计时统计：`inference_seconds`、`average_inference_seconds`、`max_inference_seconds`、`min_inference_seconds`

---

## 五、VTU 渲染输出的监控场

VTU 文件按 `vtu_interval` 间隔采样输出，文件名为 `temperature_step_XXXXXX.vtu`。

每个 VTU 文件包含以下 point data（节点标量场）：

| 场名 | 含义 | 必含 |
|------|------|------|
| `temperature` | 有量纲预测温度 | ✅ |
| `temperature_star` | 无量纲预测温度 T* | ✅ |
| `time_step` | 时间步索引 | ✅ |
| `fem_temperature` | 对应帧 FEM 真实温度 | HDF5 含 FEM 数据时 |
| `temperature_error` | 预测误差（有符号） | HDF5 含 `temperature_error` 时 |
| `abs_temperature_error` | 预测绝对误差 | HDF5 含 `temperature_error` 时 |
| `fem_valid_mask` | 有效节点掩码 | HDF5 含 `fem_valid_mask` 时 |
| `teacher_temperature` | Teacher forcing 预测温度 | Both 模式且 `step > 0` 时 |
| `teacher_temperature_star` | Teacher forcing 无量纲预测温度 | 同上 |
| `teacher_temperature_error` | Teacher forcing 预测误差 | 同上 |

VTU 使用 Gmsh 三角网格面 + `UNSTRUCTURED_GRID` 格式，可在 ParaView 中直接打开查看。

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
| `average_inference_seconds` | 平均单步推理耗时 | 写入 HDF5 metadata |
| `max_inference_seconds` | 最大单步推理耗时 | 写入 HDF5 metadata |
| `min_inference_seconds` | 最小单步推理耗时 | 写入 HDF5 metadata |

---

## 七、两种模式的适用场景对比

| 维度 | Autoregressive | Teacher Forcing |
|------|---------------|-----------------|
| 输入温度来源 | 模型自身上一步预测 | FEM 真实温度（ground truth） |
| 误差是否累积 | 是（随 rollout 步数增长） | 否（每步独立评估，不递推） |
| 评估目标 | 模型长期 rollout 稳定性与累积误差 | 单步 PDE 残差的预测精度 |
| 需要 FEM 数据 | 否（有则写入对比数据集） | **是（必须，否则报错）** |
| 输出帧数 | `steps` 帧 | `steps - 1` 个 transition |
| 典型用途 | 验证推理时模型能否稳定推进 | 解耦评估：排除误差累积后的单步精度 |

---

## 八、配置示例

```json
{
  "training_config": "pdgcn_train.example.json",
  "single_layer_inference": {
    "output_path": "../runs/pdgcn/single_layer_prediction.h5",
    "dataset_index": 0,
    "h5_path": "../case1_validTheta_50X50/case1_Q0p666667_V25_dt0p05_F91_fem.h5",
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
| `steps` | int/null | 推理帧数，null 时使用 HDF5 全部帧 |
| `warmup_steps` | int/null | 自回归初始温度伪时间弛豫步数，null 时使用训练配置值 |
| `mode` | string | `"autoregressive"` / `"teacher_forcing"` / `"both"` |
| `write_vtu` | bool | 是否输出 VTU 可视化文件 |
| `vtu_interval` | int | VTU 输出间隔（每 N 步输出一个） |
| `vtu_output_dir` | string/null | VTU 输出目录，null 时自动在 output 旁创建 `*_vtu` 目录 |
| `fem_temperature_dataset` | string | HDF5 中 FEM 温度数据集路径 |
| `fem_valid_mask_dataset` | string/null | HDF5 中 FEM 有效掩码数据集路径，null 时使用全 1 掩码 |

# PDGCN

PDGCN 是面向曲面铺放热场预测的图神经网络训练框架。项目目标是构建曲面内 PDGCN 神经网络与曲面差分技术，用于预测多层堆叠铺放曲面的温度场。

本仓库中的 PDGCN 架构改造自 PIGNN 思路，训练阶段聚焦单层曲面内热动力学，推理阶段可与厚度方向 1D FDM 结合，扩展到多层堆叠结构的三维近似热场预测。

## 项目特点

- 面向 AFP 铺放过程中的曲面瞬态热场预测。
- 使用 PyTorch Geometric 图数据表示曲面局部窗口。
- 节点特征包含坐标、纤维方向、当前温度和热源强度。
- 边特征编码局部几何、扫描方向和纤维各向异性关系。
- 使用物理约束损失训练，包括 PDE 残差和出流边界约束。
- 支持固定拓扑缓存训练，减少大规模图训练时的数据搬运和重复构图开销。
- 训练输出 checkpoint 和 history，便于后续 rollout 与验证。

## 仓库结构

```text
PDGCN/
├── configs/                 # 训练配置示例
├── data/                    # HDF5 数据读取、无量纲化、特征构建、静态缓存
├── models/                  # PDGCN 模型、编码器、处理器、解码器
├── pde/                     # PDE residual、边界条件和物理损失
├── training/                # 训练入口、TBPTT、推理、checkpoint、监控
├── 训练计划.md              # 快速训练与验证闭环计划
├── AGENTS.md                # 项目背景和协作说明
└── README.md
```

以下目录为参考资料目录，按项目约定可读取但不修改：

```text
DesignPlan/                  # PDGCN+FDM 技术方案、无量纲化、损失函数等设计资料
PIGNN/                       # 原 PIGNN 仓库源码参考
```

`DesignPlan/`、`PIGNN/`、`runs/` 和训练二进制产物已在 `.gitignore` 中排除，不会纳入当前仓库跟踪。

## 环境

本项目使用 conda 中已配置好的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```

建议所有训练和测试命令都使用该 Python 解释器。

核心依赖包括：

- PyTorch
- PyTorch Geometric
- h5py
- numpy
- pytest

## 数据约定

训练输入 HDF5 需要包含以下数据集：

```text
dynamic/xyz
dynamic/fiber
dynamic/Q
edge_index
boundary_nodes/upwind
boundary_nodes/downwind
boundary_nodes/side
```

当前示例配置默认使用：

```text
DesignPlan/1.h5
```

节点特征布局：

```text
[x*, y*, z*, fx, fy, fz, T*, Q*]
```

边特征布局：

```text
[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]
```

其中带 `*` 的量为无量纲变量。

## 快速开始

### 运行测试

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe -m pytest
```

### 使用示例配置训练

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\train_entry.py --config configs\pdgcn_train.example.json
```

默认输出：

```text
runs/pdgcn/checkpoint.pt
runs/pdgcn/history.json
runs/pdgcn/cache/case_1/
```

`runs/` 为训练输出目录，默认不纳入 Git 跟踪。

## 训练流程

当前主训练入口为：

```text
training/train_entry.py
```

它会完成以下步骤：

1. 读取 JSON 训练配置。
2. 从尺度参数派生 PDE 常数 `inverse_pe`、`pi_q`，并从 HDF5 文件级步长与速度派生 `dt_star`。
3. 读取 HDF5 数据并构建固定拓扑缓存。
4. 初始化 PDGCN 模型。
5. 使用 TBPTT 窗口执行自回归温度预测。
6. 计算 PDE residual 与出流边界物理损失。
7. 保存 checkpoint 和训练 history。

示例配置位于：

```text
configs/pdgcn_train.example.json
```

该配置默认使用单个数据集 `case_1`，模型隐藏维度为 `64`，消息传递层数为 `3`，训练轮数为 `1000`，TBPTT 窗口为 `5`。

## 模型与损失

PDGCN 输入 PyTorch Geometric `Data` 对象，并输出每个节点的无量纲温度增量：

```text
delta_T*
```

训练时将预测增量累加到当前温度：

```text
T_next* = T_current* + delta_T*
```

随后通过物理损失约束：

- `loss_pde`：无量纲瞬态对流-扩散-热源方程残差。
- `loss_outflow`：下游出流边界温度梯度约束。
- `loss_total`：总物理损失。

当前训练数据本身不包含温度真值序列，因此第一版训练是物理驱动训练；准确性建议使用外部仿真温度场做验证。

## 验证建议

详见 [训练计划.md](./训练计划.md)。

推荐的验证策略：

- 使用外部仿真真值，不参与第一版训练损失。
- 若仿真节点顺序与训练 HDF5 一致，直接逐节点比较。
- 若节点顺序不同但拓扑相同，通过坐标和层号建立映射。
- 单层真值推荐形状为 `[time, node, 1]`。
- 多层真值推荐形状为 `[time, layer, node, 1]`。

建议指标：

- RMSE
- MAE
- relative RMSE
- 峰值温度误差
- 峰值位置误差
- 逐时间步误差曲线

## 推荐训练计划

不建议一开始直接长跑到最终 epoch。推荐分三阶段：

1. 冒烟训练：`20-50` epoch，确认 loss 不发散、温度场合理。
2. 稳定训练：`200-500` epoch，保存 best checkpoint 并进行仿真验证。
3. 最终长跑：`1000` epoch，启用 early stopping 和可视化验证。

当前示例配置中 `loss_threshold=0.02`，达到阈值时会提前停止训练。

## 注意事项

- `DesignPlan/` 和 `PIGNN/` 是参考资料目录，不应修改。
- `runs/` 中包含训练缓存、checkpoint 和 history，不进入 Git。
- 若使用 GitHub SSH 远程推送，需要先在本机配置 `hayashihikaru12-sudo` 账号可用的 SSH key。
- 若使用 Personal Access Token 推送，推送后应及时撤销已暴露或临时使用的 token。

## 当前状态

本仓库已经具备：

- PDGCN 模型定义。
- HDF5 数据读取和固定拓扑缓存。
- 物理损失计算。
- 训练入口。
- checkpoint 保存。
- 基础训练 history 记录。
- 快速训练与验证计划文档。

后续重点是补充训练可视化、best checkpoint 管理和外部仿真真值验证流水线。

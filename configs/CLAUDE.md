# configs 模块说明

`configs` 目录保存 PDGCN 训练与实验运行的 JSON 配置文件。

## 主要文件

- `pdgcn_train.example.json`：示例训练配置，包含 `monitoring`、`supervision`、`outputs`、`datasets` 和 `hyperparameters`。
- `pdgcn_infer.example.json`：示例推理配置，包含 `training_config` 和 `inference`。

## 配置内容

- `outputs` 指定 checkpoint 与训练历史文件输出位置。
- `supervision` 控制 FEM 温度监督训练，默认关闭；启用时读取 `fem/temperature` 和可选 `fem/valid_mask`。
- `datasets` 指定训练数据 HDF5 目录、共享静态缓存目录，以及无量纲化尺度参数。
- `hyperparameters.model` 控制 PDGCN 网络结构。
- `hyperparameters.physics_loss` 控制无源输运残差、出流边界和平滑正则等物理损失项。
- `hyperparameters.training` 控制学习率、epoch、TBPTT、warmup、梯度裁剪和设备。

`datasets[].scale.Q0` 表示表面热流标尺 `W/m^2`。训练入口会派生 `source_coefficient`，显式热源模块用它把 `q_surface*` 转换为顶层温升；热源不再进入 PD-GCN 节点特征。

`training/train_entry.py` 读取本目录配置，并通过 `training/run_config.py` 转换为训练所需的 dataclass 配置对象。

# configs 模块说明

`configs` 目录保存 PDGCN 训练与实验运行的 JSON 配置文件。

## 主要文件

- `pdgcn_train.example.json`：示例训练配置，包含 `outputs`、`datasets` 和 `hyperparameters`。

## 配置内容

- `outputs` 指定 checkpoint 与训练历史文件输出位置。
- `datasets` 指定训练数据 HDF5 目录、共享静态缓存目录，以及无量纲化尺度参数。
- `hyperparameters.model` 控制 PDGCN 网络结构。
- `hyperparameters.physics_loss` 控制物理损失项。
- `hyperparameters.training` 控制学习率、epoch、TBPTT、warmup、梯度裁剪和设备。

`training/train_entry.py` 读取本目录配置，并通过 `training/run_config.py` 转换为训练所需的 dataclass 配置对象。

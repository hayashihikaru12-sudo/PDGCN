# configs 模块说明

`configs` 目录用于保存 PDGCN 训练与实验运行的 JSON 配置文件，是训练入口读取超参数、数据路径和输出路径的主要来源。

## 主要文件

- `pdgcn_train.example.json`：示例训练配置，采用分类式配置结构，包含 `outputs`、`datasets` 和 `hyperparameters` 三部分。

## 配置内容

- `outputs` 指定 checkpoint 与训练历史文件的输出位置，当前默认写入 `runs/pdgcn`。
- `datasets` 指定训练数据 HDF5 路径、缓存目录、是否覆盖缓存，以及无量纲化所需的尺度参数。
- `hyperparameters.model` 控制 PDGCN 网络结构，例如隐藏维度、消息传递层数、迎风门控系数、dropout 和 layer norm。
- `hyperparameters.physics_loss` 控制物理损失项，例如横纵导热比、出流边界损失权重、层间热耗散项和残差时间离散格式。
- `hyperparameters.training` 控制优化过程，例如学习率、训练轮数、TBPTT 窗口、warmup 步数、梯度裁剪和设备选择。

## 与其他模块的关系

`training/train_entry.py` 会读取本目录中的配置文件，并通过 `training/run_config.py` 转换为数据加载、模型构建和训练流程需要的 dataclass 配置对象。配置中的 HDF5 输入通常指向只读参考数据 `DesignPlan/1.h5`，训练产物写入 `runs`。

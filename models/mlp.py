import torch.nn as nn


def build_mlp(in_size: int, hidden_size: int, out_size: int, *, layer_norm: bool = True, dropout: float = 0.0):
    """构建三层前馈 MLP 模块。

    参数:
        in_size: 输入特征维度。
        hidden_size: 隐层特征维度。
        out_size: 输出特征维度。
        layer_norm: 是否在输出层后追加 ``nn.LayerNorm``。
        dropout: dropout 概率；为 ``0`` 时不插入 dropout 层。

    返回:
        ``nn.Sequential`` 模块，输入形状为 ``[..., in_size]``，
        输出形状为 ``[..., out_size]``。
    """

    # 构建初始两层结构：输入投影 -> 隐层激活。
    layers = [
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
    ]
    # 可选 dropout：仅在 dropout > 0 时启用。
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # 添加第二个隐层和激活函数以提升表达能力。
    layers.extend(
        [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ]
    )
    # 第二个隐层后的可选 dropout。
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # 投影到目标输出维度。
    layers.append(nn.Linear(hidden_size, out_size))
    # 可选 LayerNorm：稳定训练并规范化输出分布。
    if layer_norm:
        layers.append(nn.LayerNorm(out_size))

    # 按顺序组装为可调用的前馈网络。
    return nn.Sequential(*layers)

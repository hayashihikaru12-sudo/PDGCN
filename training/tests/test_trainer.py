import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models import PDGCNConfig
from training import TrainConfig, train


class TrainableDeltaModel(nn.Module):
    def __init__(self):
        """初始化带单个可训练温度增量参数的测试模型。

        参数:
            self: ``TrainableDeltaModel`` 实例。

        返回:
            None。实例包含 ``config`` 和可训练参数 ``delta``。
        """

        super().__init__()
        self.config = PDGCNConfig(
            lambda_outflow=0.0,
            inverse_pe=0.0,
            source_coefficient=0.0,
            pi_q=0.0,
            non_heating_projection=False,
        )
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph):
        """为每个节点输出相同的可训练温度增量。

        参数:
            graph: PyG ``Data`` 图对象，使用 ``graph.x.shape[0]`` 获取节点数。

        返回:
            形状 ``[N, 1]`` 的张量，所有值均为参数 ``delta``。
        """

        return self.delta.expand(graph.x.shape[0], 1)


class RecordingDeltaModel(TrainableDeltaModel):
    def __init__(self):
        """初始化会记录每次前向输入温度的测试模型。"""

        super().__init__()
        self.input_temperatures = []

    def forward(self, graph):
        """记录输入温度并返回常数温度增量。"""

        self.input_temperatures.append(graph.x[:, 6:7].detach().clone())
        return super().forward(graph)


def make_graph():
    """构造用于训练测试的小图。

    参数:
        None。

    返回:
        PyG ``Data`` 图对象，包含 4 个节点、3 条边、全局速度和边界索引。
    """

    graph = Data(
        x=torch.zeros(4, 7),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        global_attr=torch.tensor([1.0]),
    )
    graph.num_nodes = 4
    graph.upwind_nodes = torch.tensor([0], dtype=torch.long)
    graph.side_nodes = torch.tensor([3], dtype=torch.long)
    graph.downwind_nodes = torch.tensor([2], dtype=torch.long)
    return graph


class TrainerTests(unittest.TestCase):
    def test_train_updates_parameter_and_returns_history(self):
        """验证训练循环会更新参数并返回历史记录。

        参数:
            self: ``TrainerTests`` 测试用例实例。

        返回:
            None。断言训练历史结构和参数变化。
        """

        model = TrainableDeltaModel()
        before = model.delta.detach().clone()
        history = train(model, [make_graph(), make_graph()], TrainConfig(lr=0.1, epochs=1, tbptt_window=1))

        self.assertEqual(len(history), 1)
        self.assertIn("loss", history[0])
        self.assertNotEqual(float(before), float(model.delta.detach()))

    def test_train_uses_model_pseudo_time_warmup_for_initial_temperature(self):
        """验证训练初温来自当前模型的伪时间自回归 warmup。"""

        model = RecordingDeltaModel()
        train(
            model,
            [make_graph()],
            TrainConfig(lr=0.1, epochs=1, tbptt_window=1, warmup_steps=2),
        )

        self.assertGreaterEqual(len(model.input_temperatures), 3)
        expected = torch.tensor([[0.0], [2.0], [2.0], [0.0]])
        self.assertTrue(torch.allclose(model.input_temperatures[-1], expected))

    def test_train_stops_when_loss_below_threshold(self):
        """验证 epoch 平均 loss 低于阈值时提前停止训练。"""

        model = TrainableDeltaModel()
        history = train(
            model,
            [make_graph(), make_graph()],
            TrainConfig(lr=0.1, epochs=5, tbptt_window=1, warmup_steps=0, loss_threshold=1e20),
        )

        self.assertEqual(len(history), 1)
        self.assertTrue(history[-1]["stopped_early"])
        self.assertEqual(history[-1]["stop_reason"], "loss_threshold")


if __name__ == "__main__":
    unittest.main()

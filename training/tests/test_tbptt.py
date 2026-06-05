import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models import PDGCNConfig
from training.tbptt import iter_tbptt_windows, rollout_window


class ConstantDeltaModel(nn.Module):
    def __init__(self, delta=1.0, *, non_heating_projection=False):
        """初始化输出常数温度增量的测试模型。

        参数:
            self: ``ConstantDeltaModel`` 实例。
            delta: 每个节点输出的无量纲温度增量初值。

        返回:
            None。实例包含 ``config`` 和参数 ``delta``。
        """

        super().__init__()
        self.config = PDGCNConfig(non_heating_projection=non_heating_projection)
        self.delta = nn.Parameter(torch.tensor(float(delta)))

    def forward(self, graph):
        """为每个节点返回同一个温度增量。

        参数:
            graph: PyG ``Data`` 图对象，使用 ``graph.x.shape[0]`` 获取节点数。

        返回:
            形状 ``[N, 1]`` 的张量，每个元素均为 ``delta``。
        """

        return self.delta.expand(graph.x.shape[0], 1)


def make_graph(num_nodes=3, temperature=0.0):
    """构造用于 TBPTT 测试的小图。

    参数:
        num_nodes: 节点数量 ``N``。
        temperature: 节点初始无量纲温度值。

    返回:
        PyG ``Data`` 图对象，包含节点特征、两条测试边和空边界索引。
    """

    graph = Data(
        x=torch.zeros(num_nodes, 7),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        global_attr=torch.tensor([1.0]),
    )
    graph.num_nodes = num_nodes
    graph.x[:, 6:7] = float(temperature)
    graph.upwind_nodes = torch.empty(0, dtype=torch.long)
    graph.side_nodes = torch.empty(0, dtype=torch.long)
    graph.downwind_nodes = torch.empty(0, dtype=torch.long)
    return graph


class TBPTTTests(unittest.TestCase):
    def test_iter_tbptt_windows_preserves_order(self):
        """验证 TBPTT 窗口切分保持原序列顺序。

        参数:
            self: ``TBPTTTests`` 测试用例实例。

        返回:
            None。断言窗口列表内容。
        """

        windows = list(iter_tbptt_windows([0, 1, 2, 3, 4], 2))
        self.assertEqual(windows, [[0, 1], [2, 3], [4]])

    def test_rollout_window_accumulates_temperature(self):
        """验证窗口 rollout 会逐步累积温度增量。

        参数:
            self: ``TBPTTTests`` 测试用例实例。

        返回:
            None。断言预测序列形状和窗口末温度。
        """

        model = ConstantDeltaModel(delta=1.0)
        window = [make_graph(), make_graph()]
        predictions, final_temperature = rollout_window(model, window, torch.zeros(3, 1))

        self.assertEqual(tuple(predictions.shape), (2, 3, 1))
        self.assertTrue(torch.allclose(predictions[0], torch.ones(3, 1)))
        self.assertTrue(torch.allclose(final_temperature, torch.full((3, 1), 2.0)))

    def test_rollout_window_projects_positive_mean_delta(self):
        model = ConstantDeltaModel(delta=1.0, non_heating_projection=True)
        window = [make_graph(), make_graph()]
        predictions, final_temperature = rollout_window(model, window, torch.zeros(3, 1))

        self.assertTrue(torch.allclose(predictions, torch.zeros(2, 3, 1), atol=1e-6))
        self.assertTrue(torch.allclose(final_temperature, torch.zeros(3, 1), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

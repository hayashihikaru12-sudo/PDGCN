import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from data import ScaleParams
from inference import rollout_multilayer_fdm
from models import PDGCNConfig
from training import rollout


class ConstantDeltaModel(nn.Module):
    def __init__(self):
        """初始化输出常数温度增量的推理测试模型。

        参数:
            self: ``ConstantDeltaModel`` 实例。

        返回:
            None。实例包含 ``config`` 和参数 ``delta``。
        """

        super().__init__()
        self.config = PDGCNConfig()
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph):
        """为图中每个节点输出相同的温度增量。

        参数:
            graph: PyG ``Data`` 图对象，使用 ``graph.x.shape[0]`` 获取节点数。

        返回:
            形状 ``[N, 1]`` 的张量，所有值均为 ``delta``。
        """

        return self.delta.expand(graph.x.shape[0], 1)


def make_graph():
    """构造用于推理测试的小图。

    参数:
        None。

    返回:
        PyG ``Data`` 图对象，包含 2 个节点、1 条边、初温和空边界索引。
    """

    graph = Data(
        x=torch.zeros(2, 7),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        global_attr=torch.tensor([1.0]),
    )
    graph.num_nodes = 2
    graph.x[:, 6:7] = 2.0
    graph.upwind_nodes = torch.empty(0, dtype=torch.long)
    graph.side_nodes = torch.empty(0, dtype=torch.long)
    graph.downwind_nodes = torch.empty(0, dtype=torch.long)
    return graph


class InferenceTests(unittest.TestCase):
    def test_rollout_returns_real_temperature_by_default(self):
        """验证 rollout 默认返回真实温度。

        参数:
            self: ``InferenceTests`` 测试用例实例。

        返回:
            None。断言输出形状和真实温度数值。
        """

        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        output = rollout(ConstantDeltaModel(), make_graph(), 2, scale_params)

        self.assertEqual(tuple(output.shape), (2, 2, 1))
        self.assertTrue(torch.allclose(output[0], torch.full((2, 1), 330.0)))
        self.assertTrue(torch.allclose(output[1], torch.full((2, 1), 340.0)))

    def test_rollout_can_return_dimensionless_temperature(self):
        """验证 rollout 可同时返回无量纲温度。

        参数:
            self: ``InferenceTests`` 测试用例实例。

        返回:
            None。断言字典中的真实温度和无量纲温度数值。
        """

        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        result = rollout(ConstantDeltaModel(), make_graph(), 1, scale_params, return_dimensionless=True)

        self.assertTrue(torch.allclose(result["temperature_star"][0], torch.full((2, 1), 3.0)))
        self.assertTrue(torch.allclose(result["temperature"][0], torch.full((2, 1), 330.0)))

    def test_rollout_can_start_from_model_pseudo_time_warmup(self):
        """验证显式 warmup 后 rollout 从松弛后的冷态温度继续推理。"""

        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        result = rollout(
            ConstantDeltaModel(),
            make_graph(),
            1,
            scale_params,
            return_dimensionless=True,
            warmup_steps=1,
        )

        self.assertTrue(torch.allclose(result["temperature_star"][0], torch.full((2, 1), 2.0)))
        self.assertTrue(torch.allclose(result["temperature"][0], torch.full((2, 1), 320.0)))

    def test_multilayer_rollout_couples_pdgcn_and_fdm(self):
        """验证多层推理会叠加 PD-GCN 增量、FDM 层间传导并钳制底层。"""

        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        result = rollout_multilayer_fdm(
            ConstantDeltaModel(),
            make_graph(),
            1,
            scale_params,
            num_layers=3,
            layer_spacing=1.0,
            return_dimensionless=True,
        )

        expected = torch.tensor(
            [
                [
                    [[2.9067245], [2.9067245]],
                    [[1.0412147], [1.0412147]],
                    [[0.00], [0.00]],
                ]
            ]
        )
        self.assertEqual(tuple(result.shape), (1, 3, 2, 1))
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_multilayer_rollout_uses_implicit_fdm_for_large_coefficients(self):
        """Verify multilayer rollout remains finite with a large implicit FDM coefficient."""

        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        model = ConstantDeltaModel()
        model.config = PDGCNConfig(inverse_pe=1.0, k_ratio=1.0, dt_star=1.0)

        result = rollout_multilayer_fdm(
            model,
            make_graph(),
            1,
            scale_params,
            num_layers=2,
            layer_spacing=1.0,
            return_dimensionless=True,
        )

        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.allclose(result[:, -1], torch.zeros_like(result[:, -1])))


if __name__ == "__main__":
    unittest.main()

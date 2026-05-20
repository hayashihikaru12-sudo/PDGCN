import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from data import ScaleParams
from inference.multilayer import rollout_multilayer_fdm
from models import PDGCNConfig


class ConstantDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PDGCNConfig()
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph):
        return self.delta.expand(graph.x.shape[0], 1)


def make_graph():
    graph = Data(
        x=torch.zeros(2, 8),
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


class MultilayerRolloutTests(unittest.TestCase):
    def test_rollout_returns_time_layer_node_temperature(self):
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

        expected = torch.tensor([[[[2.90], [2.90]], [[1.10], [1.10]], [[0.00], [0.00]]]])
        self.assertEqual(tuple(result.shape), (1, 3, 2, 1))
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_bottom_layer_is_constant(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)

        result = rollout_multilayer_fdm(
            ConstantDeltaModel(),
            make_graph(),
            2,
            scale_params,
            num_layers=2,
            layer_spacing=1.0,
            return_dimensionless=True,
            bottom_temperature_star=-0.25,
        )

        self.assertTrue(torch.allclose(result[:, -1], torch.full_like(result[:, -1], -0.25)))


if __name__ == "__main__":
    unittest.main()

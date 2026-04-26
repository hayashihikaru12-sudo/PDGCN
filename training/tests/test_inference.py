import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from data import ScaleParams
from models import PDGCNConfig
from training import rollout


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


class InferenceTests(unittest.TestCase):
    def test_rollout_returns_real_temperature_by_default(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        output = rollout(ConstantDeltaModel(), make_graph(), 2, scale_params)

        self.assertEqual(tuple(output.shape), (2, 2, 1))
        self.assertTrue(torch.allclose(output[0], torch.full((2, 1), 330.0)))
        self.assertTrue(torch.allclose(output[1], torch.full((2, 1), 340.0)))

    def test_rollout_can_return_dimensionless_temperature(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        result = rollout(ConstantDeltaModel(), make_graph(), 1, scale_params, return_dimensionless=True)

        self.assertTrue(torch.allclose(result["temperature_star"][0], torch.full((2, 1), 3.0)))
        self.assertTrue(torch.allclose(result["temperature"][0], torch.full((2, 1), 330.0)))


if __name__ == "__main__":
    unittest.main()

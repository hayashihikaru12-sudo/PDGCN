import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models import PDGCNConfig
from training.tbptt import iter_tbptt_windows, rollout_window


class ConstantDeltaModel(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.config = PDGCNConfig()
        self.delta = nn.Parameter(torch.tensor(float(delta)))

    def forward(self, graph):
        return self.delta.expand(graph.x.shape[0], 1)


def make_graph(num_nodes=3, temperature=0.0):
    graph = Data(
        x=torch.zeros(num_nodes, 8),
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
        windows = list(iter_tbptt_windows([0, 1, 2, 3, 4], 2))
        self.assertEqual(windows, [[0, 1], [2, 3], [4]])

    def test_rollout_window_accumulates_temperature(self):
        model = ConstantDeltaModel(delta=1.0)
        window = [make_graph(), make_graph()]
        predictions, final_temperature = rollout_window(model, window, torch.zeros(3, 1))

        self.assertEqual(tuple(predictions.shape), (2, 3, 1))
        self.assertTrue(torch.allclose(predictions[0], torch.ones(3, 1)))
        self.assertTrue(torch.allclose(final_temperature, torch.full((3, 1), 2.0)))


if __name__ == "__main__":
    unittest.main()

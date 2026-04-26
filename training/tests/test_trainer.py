import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models import PDGCNConfig
from training import TrainConfig, train


class TrainableDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PDGCNConfig(lambda_outflow=0.0, inverse_pe=0.0, pi_q=0.0)
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph):
        return self.delta.expand(graph.x.shape[0], 1)


def make_graph():
    graph = Data(
        x=torch.zeros(4, 8),
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
        model = TrainableDeltaModel()
        before = model.delta.detach().clone()
        history = train(model, [make_graph(), make_graph()], TrainConfig(lr=0.1, epochs=1, tbptt_window=1))

        self.assertEqual(len(history), 1)
        self.assertIn("loss", history[0])
        self.assertNotEqual(float(before), float(model.delta.detach()))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Data

from data import ScaleParams
from inference.io import _should_write_cloud_step
from inference.multilayer import _build_multilayer_graph, _smooth_delta_by_graph, rollout_multilayer_fdm
from models import PDGCNConfig


class ConstantDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PDGCNConfig()
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.call_count = 0

    def forward(self, graph):
        self.call_count += 1
        return self.delta.expand(graph.x.shape[0], 1)


def make_graph():
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


def make_geometric_graph():
    graph = Data(
        x=torch.zeros(2, 7),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        global_attr=torch.tensor([1.0]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        normal=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        velocity_direction=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    )
    graph.num_nodes = 2
    graph.x[:, 0:3] = graph.pos
    graph.x[:, 3:6] = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    graph.x[:, 6:7] = 2.0
    graph.upwind_nodes = torch.empty(0, dtype=torch.long)
    graph.side_nodes = torch.empty(0, dtype=torch.long)
    graph.downwind_nodes = torch.empty(0, dtype=torch.long)
    return graph


class MultilayerRolloutTests(unittest.TestCase):
    def test_delta_smoothing_updates_only_internal_nodes(self):
        delta = torch.tensor(
            [
                [[0.0], [20.0], [20.0], [5.0]],
                [[10.0], [30.0], [90.0], [7.0]],
            ]
        )
        edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
        boundary_nodes = {
            "upwind": torch.tensor([0]),
            "side": torch.empty(0, dtype=torch.long),
            "downwind": torch.tensor([2]),
        }

        smoothed = _smooth_delta_by_graph(delta, edge_index, boundary_nodes, alpha=0.5, steps=1)

        expected = torch.tensor(
            [
                [[0.0], [15.0], [20.0], [5.0]],
                [[10.0], [40.0], [90.0], [7.0]],
            ]
        )
        self.assertTrue(torch.allclose(smoothed, expected, atol=1e-6))

    def test_delta_smoothing_can_be_disabled(self):
        delta = torch.tensor([[[0.0], [10.0], [20.0]]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        boundary_nodes = {
            "upwind": torch.empty(0, dtype=torch.long),
            "side": torch.empty(0, dtype=torch.long),
            "downwind": torch.empty(0, dtype=torch.long),
        }

        alpha_disabled = _smooth_delta_by_graph(
            delta,
            edge_index,
            boundary_nodes,
            alpha=0.0,
            steps=1,
        )
        steps_disabled = _smooth_delta_by_graph(
            delta,
            edge_index,
            boundary_nodes,
            alpha=0.5,
            steps=0,
        )
        self.assertTrue(torch.allclose(alpha_disabled, delta))
        self.assertTrue(torch.allclose(steps_disabled, delta))

    def test_delta_smoothing_keeps_constant_fields(self):
        delta = torch.full((2, 3, 1), 4.0)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        boundary_nodes = {
            "upwind": torch.empty(0, dtype=torch.long),
            "side": torch.empty(0, dtype=torch.long),
            "downwind": torch.empty(0, dtype=torch.long),
        }

        smoothed = _smooth_delta_by_graph(delta, edge_index, boundary_nodes, alpha=0.2, steps=3)

        self.assertTrue(torch.allclose(smoothed, delta, atol=1e-6))

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

        expected = torch.tensor(
            [[[[2.9067245], [2.9067245]], [[1.0412147], [1.0412147]], [[0.00], [0.00]]]]
        )
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

    def test_multilayer_rollout_does_not_apply_thermal_loss_compensation(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        model = ConstantDeltaModel()
        model.config = PDGCNConfig(thermal_loss_beta=10.0)

        result = rollout_multilayer_fdm(
            model,
            make_graph(),
            1,
            scale_params,
            num_layers=3,
            layer_spacing=1.0,
            return_dimensionless=True,
        )

        expected = torch.tensor(
            [[[[2.9067245], [2.9067245]], [[1.0412147], [1.0412147]], [[0.00], [0.00]]]]
        )
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_rollout_can_disable_pdgcn_inplane_update(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        model = ConstantDeltaModel()

        result = rollout_multilayer_fdm(
            model,
            make_graph(),
            1,
            scale_params,
            num_layers=3,
            layer_spacing=1.0,
            return_dimensionless=True,
            use_pdgcn_inplane=False,
        )

        expected = torch.tensor(
            [[[[1.9088937], [1.9088937]], [[0.0867679], [0.0867679]], [[0.00], [0.00]]]]
        )
        self.assertEqual(model.call_count, 0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_rollout_can_apply_pdgcn_inplane_only_to_top_layer(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        model = ConstantDeltaModel()

        result = rollout_multilayer_fdm(
            model,
            make_graph(),
            1,
            scale_params,
            num_layers=3,
            layer_spacing=1.0,
            return_dimensionless=True,
            pdgcn_inplane_top_layer_only=True,
        )

        expected = torch.tensor(
            [[[[2.8633406], [2.8633406]], [[0.1301518], [0.1301518]], [[0.00], [0.00]]]]
        )
        self.assertEqual(model.call_count, 1)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_explicit_source_heats_only_top_layer_when_fdm_is_off(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        graph = make_graph()
        graph.q_surface_star = torch.tensor([[1.0], [0.0]])
        model = ConstantDeltaModel()
        model.delta.data.fill_(0.0)
        model.config = PDGCNConfig(
            inverse_pe=0.0,
            k_ratio=0.0,
            dt_star=0.5,
            source_coefficient=2.0,
            heat_source_absorptivity=1.0,
        )

        result = rollout_multilayer_fdm(
            model,
            graph,
            1,
            scale_params,
            num_layers=3,
            layer_spacing=1.0,
            return_dimensionless=True,
        )

        expected = torch.tensor([[[[3.0], [2.0]], [[0.0], [0.0]], [[0.0], [0.0]]]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_multilayer_graph_zeroes_source_features_below_top_layer(self):
        graph = make_graph()
        graph.x = torch.zeros(2, 9)
        graph.x[:, 6:7] = 2.0
        graph.q_surface_star = torch.tensor([[1.0], [0.5]])
        graph.q_feature_index = 7
        graph.delta_t_source_feature_index = 8
        graph.include_q_in_features = True
        graph.include_delta_t_source_in_features = True
        temperature = torch.full((3, 2, 1), 2.0)
        source_delta = torch.zeros(3, 2, 1)
        source_delta[0] = torch.tensor([[0.25], [0.125]])

        multilayer = _build_multilayer_graph(
            graph,
            temperature,
            source_delta,
            layer_spacing_star=0.0,
            layer_fiber_angles_deg=[0.0, 0.0, 0.0],
            normal_offset_sign=-1,
        )

        expected_q = torch.tensor([[1.0], [0.5], [0.0], [0.0], [0.0], [0.0]])
        expected_delta = torch.tensor([[0.25], [0.125], [0.0], [0.0], [0.0], [0.0]])
        self.assertTrue(torch.allclose(multilayer.x[:, 7:8], expected_q, atol=1e-6))
        self.assertTrue(torch.allclose(multilayer.x[:, 8:9], expected_delta, atol=1e-6))

    def test_cloud_interval_writes_every_nth_step_from_zero(self):
        written = [step for step in range(12) if _should_write_cloud_step(step, 5)]

        self.assertEqual(written, [0, 5, 10])

    def test_multilayer_graph_offsets_nodes_along_normal(self):
        graph = make_geometric_graph()
        temperature = torch.full((3, 2, 1), 2.0)

        multilayer = _build_multilayer_graph(
            graph,
            temperature,
            layer_spacing_star=0.15,
            layer_fiber_angles_deg=[0.0, 45.0, 90.0],
            normal_offset_sign=-1,
        )

        expected_z = torch.tensor([0.0, 0.0, -0.15, -0.15, -0.30, -0.30])
        self.assertTrue(torch.allclose(multilayer.pos[:, 2], expected_z, atol=1e-6))

    def test_multilayer_graph_rotates_fibers_without_projection(self):
        graph = make_geometric_graph()
        temperature = torch.full((2, 2, 1), 2.0)

        multilayer = _build_multilayer_graph(
            graph,
            temperature,
            layer_spacing_star=0.15,
            layer_fiber_angles_deg=[0.0, 90.0],
            normal_offset_sign=-1,
        )

        self.assertTrue(torch.allclose(multilayer.x[0:2, 3:6], torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])))
        self.assertTrue(torch.allclose(multilayer.x[2:4, 3:6], torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6))
        self.assertAlmostEqual(float(multilayer.edge_attr[0, 6]), 1.0, places=6)
        self.assertAlmostEqual(float(multilayer.edge_attr[1, 6]), 0.0, places=6)

    def test_multilayer_rollout_calls_model_once_per_step(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        model = ConstantDeltaModel()

        rollout_multilayer_fdm(
            model,
            make_geometric_graph(),
            3,
            scale_params,
            num_layers=4,
            layer_spacing=1.0,
            return_dimensionless=True,
            layer_fiber_angles_deg=[0.0, 45.0, -45.0, 90.0],
        )

        self.assertEqual(model.call_count, 3)

    def test_multilayer_rollout_batches_model_by_layer(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        model = ConstantDeltaModel()

        rollout_multilayer_fdm(
            model,
            make_geometric_graph(),
            3,
            scale_params,
            num_layers=5,
            layer_spacing=1.0,
            return_dimensionless=True,
            layer_fiber_angles_deg=[0.0, 15.0, 30.0, 45.0, 60.0],
            layer_batch_size=2,
        )

        self.assertEqual(model.call_count, 9)

    def test_layer_batched_rollout_matches_full_layer_rollout(self):
        scale_params = ScaleParams(L0=1.0, v0=1.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)
        angles = [0.0, 15.0, 30.0, 45.0, 60.0]

        full = rollout_multilayer_fdm(
            ConstantDeltaModel(),
            make_geometric_graph(),
            2,
            scale_params,
            num_layers=5,
            layer_spacing=1.0,
            return_dimensionless=True,
            layer_fiber_angles_deg=angles,
        )
        batched = rollout_multilayer_fdm(
            ConstantDeltaModel(),
            make_geometric_graph(),
            2,
            scale_params,
            num_layers=5,
            layer_spacing=1.0,
            return_dimensionless=True,
            layer_fiber_angles_deg=angles,
            layer_batch_size=2,
        )

        self.assertTrue(torch.allclose(full, batched, atol=1e-6))

    def test_multilayer_graph_uses_absolute_layer_indices_for_batches(self):
        graph = make_geometric_graph()
        temperature = torch.full((2, 2, 1), 2.0)

        multilayer = _build_multilayer_graph(
            graph,
            temperature,
            layer_spacing_star=0.15,
            layer_fiber_angles_deg=[0.0, 45.0, 90.0, 180.0],
            normal_offset_sign=-1,
            layer_indices=torch.tensor([2, 3], dtype=torch.long),
        )

        expected_z = torch.tensor([-0.30, -0.30, -0.45, -0.45])
        self.assertTrue(torch.allclose(multilayer.pos[:, 2], expected_z, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                multilayer.x[0:2, 3:6],
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
                atol=1e-6,
            )
        )
        self.assertEqual(multilayer.x.shape[1], 7)


if __name__ == "__main__":
    unittest.main()

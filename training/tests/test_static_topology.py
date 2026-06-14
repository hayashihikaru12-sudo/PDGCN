import shutil
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from data import HDF5FrameReader, HDF5Loader, ScaleParams, build_graph, build_static_cache
from models import PDGCN, PDGCNConfig
from training import (
    GpuFeatureBuilder,
    StaticGraphState,
    rollout_static_topology,
    train_static_topology,
    train_static_topology_sequences,
)
from training.config import TrainConfig
from training.run_config import SupervisionRunConfig


class TrainableDeltaModel(nn.Module):
    def __init__(self):
        """初始化固定拓扑训练测试用模型。

        参数:
            self: ``TrainableDeltaModel`` 实例。

        返回:
            None。实例包含一个可训练的温度增量参数。
        """

        super().__init__()
        self.config = PDGCNConfig(lambda_outflow=0.0, inverse_pe=0.0, source_coefficient=0.0, pi_q=0.0)
        self.delta = nn.Parameter(torch.tensor(0.1))

    def forward(self, graph):
        """返回每个节点共享的温度增量。

        参数:
            graph: 当前帧图对象，使用 ``graph.x.shape[0]`` 获取节点数量。

        返回:
            形状 ``[N, 1]`` 的温度增量张量。
        """

        return self.delta.expand(graph.x.shape[0], 1)


class RecordingDeltaModel(TrainableDeltaModel):
    def __init__(self):
        super().__init__()
        self.forward_temperatures = []
        self.warmup_deltas = []

    def forward(self, graph):
        if not torch.is_grad_enabled():
            self.warmup_deltas.append(float(self.delta.detach().cpu()))
        else:
            self.forward_temperatures.append(graph.x[:, 6:7].detach().cpu().clone())
        return super().forward(graph)


def make_h5(path: Path):
    """创建固定拓扑测试 HDF5 文件。

    参数:
        path: 输出 HDF5 文件路径。

    返回:
        None。函数会写入两帧四节点的小图数据。
    """

    xyz = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0], [1.1, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    fiber = np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), (2, 4, 1))
    normal = np.tile(np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32), (2, 4, 1))
    q = np.array([[[0.0], [1.0], [0.5], [0.0]], [[0.0], [0.8], [0.4], [0.0]]], dtype=np.float32)
    edge_index = np.array([[0, 1, 2, 0], [1, 3, 3, 2]], dtype=np.int64)

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["velocity_speed"] = 2.0
        h5_file.attrs["velocity_direction_local"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dynamic = h5_file.create_group("dynamic")
        dynamic.create_dataset("xyz", data=xyz)
        dynamic.create_dataset("fiber", data=fiber)
        dynamic.create_dataset("normal", data=normal)
        dynamic.create_dataset("Q", data=q)
        h5_file.create_dataset("edge_index", data=edge_index)
        boundary = h5_file.create_group("boundary_nodes")
        boundary.create_dataset("upwind", data=np.array([0], dtype=np.int64))
        boundary.create_dataset("downwind", data=np.array([3], dtype=np.int64))
        boundary.create_dataset("side", data=np.array([2], dtype=np.int64))


def add_fem_temperature(path: Path, *, with_mask: bool = True):
    fem_temperature = np.array(
        [
            [[[300.0], [302.0], [300.0], [304.0]]],
            [[[300.0], [303.5], [300.0], [305.5]]],
        ],
        dtype=np.float32,
    ).reshape(2, 4, 1)
    with h5py.File(path, "a") as h5_file:
        fem = h5_file.require_group("fem")
        fem.create_dataset("temperature", data=fem_temperature)
        fem.create_dataset("temperature_unit", data="degC")
        fem.create_dataset("time", data=np.array([0.0, 0.25], dtype=np.float64))
        if with_mask:
            fem.create_dataset("valid_mask", data=np.ones((2, 4, 1), dtype=np.uint8))


class StaticTopologyTests(unittest.TestCase):
    def setUp(self):
        """准备固定拓扑测试缓存目录。

        参数:
            self: ``StaticTopologyTests`` 测试用例实例。

        返回:
            None。创建临时 HDF5 文件和缓存目录路径。
        """

        self.root = Path("training/tests/_tmp_static_topology")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.h5_path = self.root / "input.h5"
        self.cache_dir = self.root / "cache"
        make_h5(self.h5_path)
        self.scale = ScaleParams(
            L0=0.002,
            v0=0.002,
            T_amb=300.0,
            delta_T0=10.0,
            Q0=2.0e6,
            heat_source_effective_thickness=0.001,
        )

    def tearDown(self):
        """清理固定拓扑测试临时目录。

        参数:
            self: ``StaticTopologyTests`` 测试用例实例。

        返回:
            None。删除测试产生的临时文件。
        """

        if self.root.exists():
            shutil.rmtree(self.root)

    def test_gpu_feature_builder_matches_build_graph_on_cpu(self):
        """验证新特征工厂与旧构图路径在同一帧上输出一致。

        参数:
            self: ``StaticTopologyTests`` 测试用例实例。

        返回:
            None。断言节点、边和全局特征一致。
        """

        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        reader = HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)
        static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
        builder = GpuFeatureBuilder(static_state, self.scale)

        node_base, global_condition = reader.read_frame(0)
        graph_fast = builder.build(node_base, global_condition, torch.zeros(4, 1))

        raw = HDF5Loader(self.h5_path, scale_params=self.scale).load_graph_data(0)
        graph_ref = build_graph(raw, self.scale, scan_velocity=0.002, initial_temperature=torch.full((4, 1), 300.0))

        self.assertTrue(torch.allclose(graph_fast.x, graph_ref.x, atol=1e-6))
        self.assertTrue(torch.allclose(graph_fast.edge_attr, graph_ref.edge_attr, atol=1e-6))
        self.assertTrue(torch.allclose(graph_fast.global_attr, graph_ref.global_attr, atol=1e-6))
        self.assertTrue(torch.allclose(graph_fast.q_surface_star, graph_ref.q_surface_star, atol=1e-6))
        reader.close()

    def test_edge_cos_theta_uses_receiver_tangent_velocity(self):
        tilted_h5_path = self.root / "tilted_normal.h5"
        make_h5(tilted_h5_path)
        with h5py.File(tilted_h5_path, "a") as h5_file:
            normal = h5_file["dynamic/normal"][()]
            normal[:, 1, :] = np.array([1.0, 0.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
            h5_file["dynamic/normal"][...] = normal

        raw = HDF5Loader(tilted_h5_path, scale_params=self.scale).load_graph_data(0)
        graph = build_graph(raw, self.scale, scan_velocity=0.002, initial_temperature=torch.full((4, 1), 300.0))

        self.assertAlmostEqual(float(graph.edge_attr[0, 4]), float(1.0 / np.sqrt(2.0)), places=6)

    def test_hdf5_reader_converts_native_units_to_si(self):
        reader = HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)
        try:
            node_base, global_condition = reader.read_frame(0)
        finally:
            reader.close()

        self.assertAlmostEqual(float(node_base[1, 0]), 0.001)
        self.assertAlmostEqual(float(global_condition[0]), 0.002)
        self.assertAlmostEqual(float(node_base[1, 12]), 1.0e6)
        self.assertTrue(torch.allclose(node_base[1, 6:9], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(node_base[1, 9:12], torch.tensor([1.0, 0.0, 0.0])))

    def test_hdf5_reader_requires_effective_thickness_for_heat_flux_conversion(self):
        with self.assertRaisesRegex(ValueError, "heat_source_effective_thickness"):
            HDF5FrameReader(self.h5_path, expected_num_nodes=4, pin_memory=False)

    def test_hdf5_reader_reads_fem_temperature_and_mask(self):
        add_fem_temperature(self.h5_path)
        reader = HDF5FrameReader(
            self.h5_path,
            expected_num_nodes=4,
            scale_params=self.scale,
            require_fem_temperature=True,
            pin_memory=False,
        )
        try:
            self.assertTrue(reader.has_fem_temperature)
            temperature = reader.read_fem_temperature(1)
            mask = reader.read_fem_valid_mask(1)
        finally:
            reader.close()

        self.assertEqual(tuple(temperature.shape), (4, 1))
        self.assertEqual(tuple(mask.shape), (4, 1))
        self.assertAlmostEqual(float(temperature[1, 0]), 303.5)
        self.assertTrue(torch.allclose(mask, torch.ones(4, 1)))

    def test_hdf5_reader_uses_all_one_fem_mask_when_missing(self):
        add_fem_temperature(self.h5_path, with_mask=False)
        reader = HDF5FrameReader(
            self.h5_path,
            expected_num_nodes=4,
            scale_params=self.scale,
            require_fem_temperature=True,
            pin_memory=False,
        )
        try:
            mask = reader.read_fem_valid_mask(0)
        finally:
            reader.close()

        self.assertTrue(torch.allclose(mask, torch.ones(4, 1)))

    def test_hdf5_reader_requires_fem_temperature_when_requested(self):
        with self.assertRaisesRegex(KeyError, "fem/temperature"):
            HDF5FrameReader(
                self.h5_path,
                expected_num_nodes=4,
                scale_params=self.scale,
                require_fem_temperature=True,
                pin_memory=False,
            )

    def test_static_train_and_rollout_smoke(self):
        """验证固定拓扑训练和流式推理入口可运行。

        参数:
            self: ``StaticTopologyTests`` 测试用例实例。

        返回:
            None。断言训练历史和推理输出形状。
        """

        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        reader = HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)
        static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
        builder = GpuFeatureBuilder(static_state, self.scale)
        model = TrainableDeltaModel()

        history = train_static_topology(
            model,
            reader,
            static_state,
            builder,
            TrainConfig(lr=0.01, epochs=1, tbptt_window=1, device="cpu"),
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(len(history[0]["window_losses"]), 2)
        for key in (
            "loss_total",
            "loss_pde",
            "loss_outflow",
            "loss_beta",
            "loss_smooth",
            "temperature_mean",
            "temperature_max",
            "temperature_min",
            "temperature_var",
        ):
            self.assertIn(key, history[0])

        output = rollout_static_topology(
            model,
            reader,
            static_state,
            builder,
            steps=2,
            scale_params=self.scale,
            return_all=True,
        )
        self.assertEqual(tuple(output.shape), (2, 4, 1))
        self.assertTrue(torch.isfinite(output).all())
        reader.close()

    def test_static_train_real_pdgcn_supports_multi_step_tbptt(self):
        """验证真实 PD-GCN 在固定拓扑多步 TBPTT 下可反向传播。"""

        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        reader = HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)
        try:
            static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
            builder = GpuFeatureBuilder(static_state, self.scale)
            model = PDGCN(
                PDGCNConfig(
                    hidden_size=8,
                    message_passing_num=1,
                    lambda_outflow=0.0,
                    inverse_pe=0.0,
                    source_coefficient=0.0,
                    pi_q=0.0,
                )
            )

            history = train_static_topology(
                model,
                reader,
                static_state,
                builder,
                TrainConfig(lr=1e-4, epochs=1, tbptt_window=2, warmup_steps=2, device="cpu"),
            )
        finally:
            reader.close()

        self.assertEqual(len(history), 1)
        self.assertEqual(len(history[0]["window_losses"]), 1)
        self.assertTrue(torch.isfinite(torch.tensor(history[0]["window_losses"])).all())

    def test_static_train_supervised_uses_fem_teacher_forcing(self):
        add_fem_temperature(self.h5_path)
        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        reader = HDF5FrameReader(
            self.h5_path,
            expected_num_nodes=4,
            scale_params=self.scale,
            require_fem_temperature=True,
            pin_memory=False,
        )
        model = RecordingDeltaModel()
        try:
            static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
            builder = GpuFeatureBuilder(static_state, self.scale)
            history = train_static_topology(
                model,
                reader,
                static_state,
                builder,
                TrainConfig(lr=0.01, epochs=1, tbptt_window=1, warmup_steps=3, device="cpu"),
                supervision_config=SupervisionRunConfig(enabled=True, lambda_temperature=1.0),
            )
        finally:
            reader.close()

        self.assertEqual(len(history), 1)
        self.assertEqual(len(history[0]["window_losses"]), 1)
        self.assertEqual(model.warmup_deltas, [])
        self.assertEqual(tuple(model.forward_temperatures[0].shape), (4, 1))
        expected_input = torch.tensor([[0.0], [0.2], [0.0], [0.4]])
        self.assertTrue(torch.allclose(model.forward_temperatures[0], expected_input, atol=1e-6))
        self.assertIn("loss_temperature", history[0])
        self.assertIn("loss_supervised", history[0])
        self.assertIn("loss_physics", history[0])
        self.assertAlmostEqual(history[0]["loss_temperature"], 0.00125, places=6)
        self.assertAlmostEqual(history[0]["loss_supervised"], 0.00125, places=6)
        self.assertAlmostEqual(history[0]["fem_temperature_rmse"], float(np.sqrt(0.125)), places=6)
        self.assertEqual(model.forward_temperatures[0].shape[1], 1)

    def test_static_train_calls_epoch_callback(self):
        """验证固定拓扑训练每个 epoch 结束后会触发 loss 回调。"""

        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        reader = HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)
        records = []
        try:
            static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
            builder = GpuFeatureBuilder(static_state, self.scale)
            train_static_topology(
                TrainableDeltaModel(),
                reader,
                static_state,
                builder,
                TrainConfig(lr=0.01, epochs=2, tbptt_window=1, warmup_steps=0, device="cpu"),
                epoch_callback=records.append,
            )
        finally:
            reader.close()

        self.assertEqual([record["epoch"] for record in records], [0, 1])
        self.assertTrue(all(torch.isfinite(torch.tensor(record["loss"])) for record in records))

    def test_static_rollout_can_start_from_model_pseudo_time_warmup(self):
        """验证固定拓扑推理可显式启用模型伪时间 warmup。"""

        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        reader = HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)
        try:
            static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
            builder = GpuFeatureBuilder(static_state, self.scale)
            output = rollout_static_topology(
                TrainableDeltaModel(),
                reader,
                static_state,
                builder,
                steps=1,
                scale_params=self.scale,
                return_all=True,
                return_dimensionless=True,
                warmup_steps=2,
            )
        finally:
            reader.close()

        expected = torch.tensor([[[0.0], [0.3], [0.0], [0.3]]])
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_static_cache_contains_only_static_files(self):
        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)

        self.assertTrue((self.cache_dir / "static.pt").exists())
        self.assertTrue((self.cache_dir / "meta.json").exists())
        self.assertFalse((self.cache_dir / "dynamic_node_base.npy").exists())
        self.assertFalse((self.cache_dir / "global.npy").exists())

    def test_hdf5_frame_reader_rejects_missing_dynamic_dataset(self):
        bad_h5_path = self.root / "missing_q.h5"
        make_h5(bad_h5_path)
        with h5py.File(bad_h5_path, "a") as h5_file:
            del h5_file["dynamic/Q"]

        with self.assertRaisesRegex(KeyError, "dynamic/Q"):
            HDF5FrameReader(bad_h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)

    def test_hdf5_frame_reader_rejects_missing_normal_dataset(self):
        bad_h5_path = self.root / "missing_normal.h5"
        make_h5(bad_h5_path)
        with h5py.File(bad_h5_path, "a") as h5_file:
            del h5_file["dynamic/normal"]

        with self.assertRaisesRegex(KeyError, "dynamic/normal"):
            HDF5FrameReader(bad_h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False)

    def test_hdf5_frame_reader_rejects_node_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "static cache expects"):
            HDF5FrameReader(self.h5_path, expected_num_nodes=5, scale_params=self.scale, pin_memory=False)

    def test_multi_file_training_resets_temperature_per_file(self):
        second_h5_path = self.root / "input2.h5"
        make_h5(second_h5_path)
        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
        builder = GpuFeatureBuilder(static_state, self.scale)
        model = RecordingDeltaModel()
        readers = [
            HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False),
            HDF5FrameReader(second_h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False),
        ]
        try:
            history = train_static_topology_sequences(
                model,
                readers,
                static_state,
                builder,
                TrainConfig(lr=0.01, epochs=1, tbptt_window=2, warmup_steps=0, device="cpu"),
            )
        finally:
            for reader in readers:
                reader.close()

        self.assertEqual(history[0]["file_window_counts"], [1, 1])
        self.assertGreaterEqual(len(model.forward_temperatures), 4)
        self.assertTrue(torch.allclose(model.forward_temperatures[0], torch.zeros(4, 1)))
        self.assertTrue(torch.allclose(model.forward_temperatures[2], torch.zeros(4, 1)))

    def test_multi_file_warmup_uses_latest_model_parameters(self):
        second_h5_path = self.root / "input2.h5"
        make_h5(second_h5_path)
        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
        builder = GpuFeatureBuilder(static_state, self.scale)
        model = RecordingDeltaModel()
        readers = [
            HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False),
            HDF5FrameReader(second_h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False),
        ]
        try:
            train_static_topology_sequences(
                model,
                readers,
                static_state,
                builder,
                TrainConfig(lr=0.01, epochs=1, tbptt_window=2, warmup_steps=1, device="cpu"),
            )
        finally:
            for reader in readers:
                reader.close()

        self.assertGreaterEqual(len(model.warmup_deltas), 2)
        self.assertNotEqual(model.warmup_deltas[0], model.warmup_deltas[1])

    def test_static_train_calls_slice_callback_after_each_hdf5_file(self):
        second_h5_path = self.root / "input2.h5"
        make_h5(second_h5_path)
        build_static_cache(self.h5_path, self.cache_dir, self.scale, overwrite=True)
        static_state = StaticGraphState.from_cache(self.cache_dir, device="cpu")
        builder = GpuFeatureBuilder(static_state, self.scale)
        readers = [
            HDF5FrameReader(self.h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False),
            HDF5FrameReader(second_h5_path, expected_num_nodes=4, scale_params=self.scale, pin_memory=False),
        ]
        callbacks = []
        try:
            train_static_topology_sequences(
                TrainableDeltaModel(),
                readers,
                static_state,
                builder,
                TrainConfig(lr=0.01, epochs=1, tbptt_window=2, warmup_steps=0, device="cpu"),
                slice_callback=callbacks.append,
            )
        finally:
            for reader in readers:
                reader.close()

        self.assertEqual([record["slice_index"] for record in callbacks], [0, 1])
        self.assertEqual([record["epoch"] for record in callbacks], [0, 0])


if __name__ == "__main__":
    unittest.main()

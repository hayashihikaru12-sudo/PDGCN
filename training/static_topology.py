from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch_geometric.data import Data

from data.dimensionless import ScaleParams, temperature_from_dimensionless
from data.velocity import tangent_velocity_direction
from data.static_cache import STATIC_FILE, HDF5FrameReader
from pde import apply_dirichlet_boundary, project_non_heating_delta, total_loss

from .config import TrainConfig
from .graph_utils import clone_graph_with_temperature, graph_explicit_source_delta
from .warmup import pseudo_time_relax_initial_temperature


@dataclass
class StaticGraphState:
    """常驻设备的固定拓扑图状态。"""

    edge_index: torch.Tensor
    source: torch.Tensor
    receiver: torch.Tensor
    boundary_nodes: dict
    node_type: torch.Tensor
    num_nodes: int
    num_edges: int
    device: torch.device

    @classmethod
    def from_cache(cls, cache_dir, device=None):
        """从静态缓存加载固定拓扑并移动到目标设备。

        参数:
            cache_dir: ``build_static_cache`` 生成的缓存目录。
            device: 目标设备；若为 ``None``，优先使用 CUDA，否则使用 CPU。

        返回:
            ``StaticGraphState`` 实例，索引和边界节点已在目标设备上。
        """

        target_device = torch.device(device) if device is not None else _default_device()
        payload = torch.load(Path(cache_dir) / STATIC_FILE, map_location="cpu")
        edge_index = payload["edge_index"].to(device=target_device, dtype=torch.long, non_blocking=True)
        boundary_nodes = {
            name: value.to(device=target_device, dtype=torch.long, non_blocking=True)
            for name, value in payload["boundary_nodes"].items()
        }
        node_type = payload["node_type"].to(device=target_device, dtype=torch.long, non_blocking=True)
        return cls(
            edge_index=edge_index,
            source=edge_index[0],
            receiver=edge_index[1],
            boundary_nodes=boundary_nodes,
            node_type=node_type,
            num_nodes=int(payload["num_nodes"]),
            num_edges=int(payload["num_edges"]),
            device=target_device,
        )


class GpuFeatureBuilder:
    """在目标设备上由基础动态特征生成 PD-GCN 节点和边特征。"""

    def __init__(self, static_state: StaticGraphState, scale_params: ScaleParams, *, dtype=torch.float32):
        """初始化 GPU 特征构建器和复用缓冲区。

        参数:
            static_state: ``StaticGraphState``，提供固定拓扑和边界节点。
            scale_params: ``ScaleParams``，提供无量纲化标尺。
            dtype: 特征张量数据类型，默认 ``torch.float32``。

        返回:
            None。实例会预分配节点、边、基础特征和全局条件缓冲区。
        """

        self.static_state = static_state
        self.scale_params = scale_params
        self.device = static_state.device
        self.dtype = dtype
        self.node_base = torch.empty(static_state.num_nodes, 13, device=self.device, dtype=dtype)
        self.global_raw = torch.empty(1, device=self.device, dtype=dtype)
        self.x = torch.empty(static_state.num_nodes, 7, device=self.device, dtype=dtype)
        self.edge_attr = torch.empty(static_state.num_edges, 7, device=self.device, dtype=dtype)
        self.q_surface_star = torch.empty(static_state.num_nodes, 1, device=self.device, dtype=dtype)
        self.graph = Data(
            x=self.x,
            edge_index=static_state.edge_index,
            edge_attr=self.edge_attr,
            node_type=static_state.node_type,
            global_attr=self.global_raw,
            pos=self.x[:, 0:3],
            q_surface_star=self.q_surface_star,
        )
        self.graph.num_nodes = static_state.num_nodes
        self.graph.upwind_nodes = static_state.boundary_nodes["upwind"]
        self.graph.downwind_nodes = static_state.boundary_nodes["downwind"]
        self.graph.side_nodes = static_state.boundary_nodes["side"]

    def build(self, node_base_cpu, global_cpu, temperature_star):
        """用当前帧基础特征更新复用图对象。

        参数:
            node_base_cpu: CPU 张量，形状 ``[N, 13]``，列为
                ``[x, y, z, fx, fy, fz, nx, ny, nz, vx, vy, vz, Q]``。
            global_cpu: CPU 张量，形状 ``[G]``，当前只使用第 1 个值作为真实扫描速度。
            temperature_star: 当前无量纲温度，形状 ``[N, 1]``。

        返回:
            复用的 PyG ``Data`` 图对象；其 ``x``、``edge_attr`` 和 ``global_attr``
            已指向当前帧特征缓冲区。
        """

        self.node_base = self.node_base.detach()
        self.global_raw = self.global_raw.detach()
        self.x = self.x.detach()
        self.edge_attr = self.edge_attr.detach()
        self.q_surface_star = self.q_surface_star.detach()

        self.node_base.copy_(node_base_cpu, non_blocking=self.device.type == "cuda")
        self.global_raw.copy_(global_cpu[:1], non_blocking=self.device.type == "cuda")

        coords_star = self.node_base[:, 0:3] / float(self.scale_params.L0)
        fibers_unit = _normalize_vectors(self.node_base[:, 3:6], eps=float(self.scale_params.eps))
        normals_unit = _normalize_vectors(self.node_base[:, 6:9], eps=float(self.scale_params.eps))
        velocity_direction = self.node_base[:, 9:12]
        q_surface_star = self.node_base[:, 12:13] / float(self.scale_params.Q0)
        temperature = temperature_star.to(device=self.device, dtype=self.dtype, non_blocking=True).reshape(-1, 1)

        self.x[:, 0:3] = coords_star
        self.x[:, 3:6] = fibers_unit
        self.x[:, 6:7] = temperature
        self.q_surface_star[:, 0:1] = q_surface_star

        source = self.static_state.source
        receiver = self.static_state.receiver
        delta = coords_star[receiver] - coords_star[source]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(float(self.scale_params.eps))
        direction = delta / distance
        tangent_velocity = tangent_velocity_direction(
            velocity_direction,
            normals_unit,
            eps=float(self.scale_params.eps),
        )
        fiber_mid = _normalize_vectors(fibers_unit[source] + fibers_unit[receiver], eps=float(self.scale_params.eps))
        cos_phi = torch.sum(fiber_mid * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)

        self.edge_attr[:, 0:3] = delta
        self.edge_attr[:, 3:4] = distance
        self.edge_attr[:, 4:5] = torch.sum(tangent_velocity[receiver] * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        self.edge_attr[:, 5:6] = cos_phi
        self.edge_attr[:, 6:7] = cos_phi.square()

        self.global_raw[:] = self.global_raw / float(self.scale_params.v0)
        self.graph.x = self.x
        self.graph.edge_attr = self.edge_attr
        self.graph.global_attr = self.global_raw
        self.graph.pos = self.x[:, 0:3]
        self.graph.q_surface_star = self.q_surface_star
        return self.graph

    def initial_temperature(self):
        """创建当前静态图的冷态无量纲初温。

        参数:
            无。

        返回:
            形状 ``[N, 1]`` 的零张量，位于特征构建器所在设备。训练入口会在
            ``warmup_steps > 0`` 时用当前 PD-GCN 权重将该冷态松弛为初温。
        """

        return torch.zeros(self.static_state.num_nodes, 1, device=self.device, dtype=self.dtype)


def _snapshot_graph_for_tbptt(graph):
    """为单个 TBPTT 时间步创建保留梯度的图快照。

    ``GpuFeatureBuilder`` 会在下一帧原地复用特征缓冲区；窗口末才
    ``backward`` 时，真实 PD-GCN 会需要这些旧帧特征。这里仅克隆会被
    覆盖且可能被 autograd 保存的动态特征，静态拓扑索引继续共享。
    """

    snapshot = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr.clone(),
        node_type=graph.node_type,
        global_attr=graph.global_attr.clone(),
        pos=graph.pos.clone(),
    )
    if hasattr(graph, "q_surface_star"):
        snapshot.q_surface_star = graph.q_surface_star.clone()
    if hasattr(graph, "q_surface"):
        snapshot.q_surface = graph.q_surface.clone()
    snapshot.num_nodes = graph.num_nodes
    snapshot.upwind_nodes = graph.upwind_nodes
    snapshot.downwind_nodes = graph.downwind_nodes
    snapshot.side_nodes = graph.side_nodes
    return snapshot


def train_static_topology(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_callback: Optional[Callable[[dict], None]] = None,
    monitor_callback: Optional[Callable[[dict, dict], None]] = None,
    slice_callback: Optional[Callable[[dict], None]] = None,
    monitor_frame_index: Optional[int] = None,
    start_epoch: int = 0,
):
    """使用固定拓扑流式数据管线训练 PD-GCN。

    参数:
        model: 待训练模型，输入当前帧图对象并输出 ``delta_T*``。
        frame_reader: ``HDF5FrameReader``，按帧提供 CPU 基础特征。
        static_state: ``StaticGraphState``，提供常驻设备的静态拓扑。
        feature_builder: ``GpuFeatureBuilder``，在设备上构建当前帧特征。
        config: ``TrainConfig``，提供训练轮数、窗口长度、warmup、学习率和设备。
        optimizer: 可选优化器；若为 ``None``，创建 Adam。
        epoch_callback: 可选回调；每个 epoch 结束后接收该 epoch 的历史记录。

    返回:
        训练历史列表；每个元素包含 ``epoch``、平均 ``loss`` 和窗口损失。
    """

    return train_static_topology_sequences(
        model,
        [frame_reader],
        static_state,
        feature_builder,
        config,
        optimizer=optimizer,
        epoch_callback=epoch_callback,
        monitor_callback=monitor_callback,
        slice_callback=slice_callback,
        monitor_frame_index=monitor_frame_index,
        start_epoch=start_epoch,
    )


def train_static_topology_sequences(
    model,
    frame_readers: Sequence[HDF5FrameReader],
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_callback: Optional[Callable[[dict], None]] = None,
    monitor_callback: Optional[Callable[[dict, dict], None]] = None,
    slice_callback: Optional[Callable[[dict], None]] = None,
    monitor_frame_index: Optional[int] = None,
    start_epoch: int = 0,
):
    """按独立 HDF5 序列训练固定拓扑 PD-GCN。"""

    if not frame_readers:
        raise ValueError("frame_readers must contain at least one sequence.")

    device = static_state.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

    history = []
    start_epoch = int(start_epoch)
    for epoch in range(start_epoch, start_epoch + int(config.epochs)):
        model.train()
        window_records = []
        file_window_counts = []
        last_snapshot = None
        for file_index, frame_reader in enumerate(frame_readers):
            sequence_records, sequence_snapshot = _train_one_static_sequence_epoch(
                model,
                frame_reader,
                static_state,
                feature_builder,
                config,
                optimizer,
                monitor_frame_index=monitor_frame_index,
            )
            window_records.extend(sequence_records)
            file_window_counts.append(len(sequence_records))
            if sequence_snapshot is not None:
                last_snapshot = sequence_snapshot
            if slice_callback is not None:
                slice_callback({"epoch": epoch, "slice_index": file_index, "frame_reader": frame_reader})

        window_losses = [record["loss_total"] for record in window_records]
        epoch_record = {
            "epoch": epoch,
            "loss": sum(window_losses) / max(len(window_losses), 1),
            "loss_total": _mean_records(window_records, "loss_total"),
            "loss_pde": _mean_records(window_records, "loss_pde"),
            "loss_outflow": _mean_records(window_records, "loss_outflow"),
            "loss_beta": _mean_records(window_records, "loss_beta"),
            "loss_smooth": _mean_records(window_records, "loss_smooth"),
            "loss_zero_source_anchor": _mean_records(window_records, "loss_zero_source_anchor"),
            "temperature_mean": _mean_records(window_records, "temperature_mean"),
            "temperature_max": _max_records(window_records, "temperature_max"),
            "temperature_min": _min_records(window_records, "temperature_min"),
            "temperature_var": _mean_records(window_records, "temperature_var"),
            "window_losses": window_losses,
            "file_window_counts": file_window_counts,
        }
        should_stop = config.loss_threshold is not None and epoch_record["loss"] < float(config.loss_threshold)
        if should_stop:
            epoch_record["stopped_early"] = True
            epoch_record["stop_reason"] = "loss_threshold"
        history.append(epoch_record)
        if monitor_callback is not None:
            monitor_callback(epoch_record, {"snapshot": last_snapshot} if last_snapshot is not None else {})
        if epoch_callback is not None:
            epoch_callback(epoch_record)
        if should_stop:
            break
    return history


def _train_one_static_sequence_epoch(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    *,
    monitor_frame_index: Optional[int] = None,
):
    if int(config.warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = _snapshot_graph_for_tbptt(
            feature_builder.build(node_base_cpu, global_cpu, feature_builder.initial_temperature())
        )
        current_temperature = pseudo_time_relax_initial_temperature(
            model,
            warmup_graph,
            int(config.warmup_steps),
        )
    else:
        current_temperature = feature_builder.initial_temperature().detach()

    window_records = []
    selected_snapshot = None
    for start in range(0, frame_reader.num_frames, int(config.tbptt_window)):
        end = min(start + int(config.tbptt_window), frame_reader.num_frames)
        optimizer.zero_grad()
        loss_terms = []
        component_records = []
        window_temperature = current_temperature
        window_snapshot = None

        for frame_idx in range(start, end):
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = _snapshot_graph_for_tbptt(
                feature_builder.build(node_base_cpu, global_cpu, window_temperature)
            )
            source_temperature = apply_dirichlet_boundary(
                window_temperature + graph_explicit_source_delta(graph, model.config),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(graph, source_temperature)
            delta_temperature = model(graph)
            if getattr(model.config, "non_heating_projection", True):
                delta_temperature = project_non_heating_delta(
                    delta_temperature,
                    static_state.boundary_nodes,
                )
            next_temperature = apply_dirichlet_boundary(
                source_temperature + delta_temperature,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            zero_source_anchor_delta = _compute_zero_source_anchor_delta(model, graph)
            components = _compute_loss_components(
                model,
                next_temperature,
                source_temperature,
                graph,
                static_state,
                zero_source_anchor_delta=zero_source_anchor_delta,
            )
            loss_terms.append(components["loss_total"])
            component_records.append(_detach_loss_record(components))
            if _should_capture_frame(frame_idx, monitor_frame_index, frame_reader.num_frames):
                window_snapshot = _build_monitor_snapshot(
                    graph,
                    components["residual"],
                    next_temperature,
                    feature_builder.scale_params,
                    frame_idx=frame_idx,
                )
            window_temperature = next_temperature

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        optimizer.step()

        window_record = _aggregate_component_records(component_records)
        window_record.update(_temperature_stats(window_temperature, feature_builder.scale_params))
        window_record["loss_total"] = float(loss.detach().cpu())
        window_records.append(window_record)
        if window_snapshot is not None:
            selected_snapshot = window_snapshot
        current_temperature = window_temperature.detach()
    return window_records, selected_snapshot


@torch.no_grad()
def evaluate_static_topology_sequence(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    *,
    epoch: int = 0,
    slice_index: int = 0,
    monitor_frame_index: Optional[int] = None,
):
    """无梯度评估一个 HDF5 切片并返回平均损失分量和监控快照。"""

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    current_temperature = feature_builder.initial_temperature()
    if int(config.warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = _snapshot_graph_for_tbptt(
            feature_builder.build(node_base_cpu, global_cpu, current_temperature)
        )
        current_temperature = pseudo_time_relax_initial_temperature(
            model,
            warmup_graph,
            int(config.warmup_steps),
        )

    component_records = []
    snapshot = None
    try:
        for frame_idx in range(frame_reader.num_frames):
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + graph_explicit_source_delta(graph, model.config),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(graph, source_temperature)
            delta_temperature = model(graph)
            if getattr(model.config, "non_heating_projection", True):
                delta_temperature = project_non_heating_delta(
                    delta_temperature,
                    static_state.boundary_nodes,
                )
            next_temperature = apply_dirichlet_boundary(
                source_temperature + delta_temperature,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            zero_source_anchor_delta = _compute_zero_source_anchor_delta(model, graph)
            components = _compute_loss_components(
                model,
                next_temperature,
                source_temperature,
                graph,
                static_state,
                zero_source_anchor_delta=zero_source_anchor_delta,
            )
            component_records.append(_detach_loss_record(components))
            if _should_capture_frame(frame_idx, monitor_frame_index, frame_reader.num_frames):
                snapshot = _build_monitor_snapshot(
                    graph,
                    components["residual"],
                    next_temperature,
                    feature_builder.scale_params,
                    frame_idx=frame_idx,
                )
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    record = {
        "epoch": int(epoch),
        "slice_index": int(slice_index),
        **_aggregate_component_records(component_records),
        **_temperature_stats(current_temperature, feature_builder.scale_params),
    }
    record["loss"] = record["loss_total"]
    return record, {"snapshot": snapshot} if snapshot is not None else {}


def _compute_loss_components(
    model,
    next_temperature,
    current_temperature,
    graph,
    static_state: StaticGraphState,
    *,
    zero_source_anchor_delta=None,
):
    return total_loss(
        T_next=next_temperature,
        T_current=current_temperature,
        v_scan_star=graph.global_attr,
        dt_star=model.config.dt_star,
        edge_index=static_state.edge_index,
        edge_attr=graph.edge_attr,
        boundary_nodes=static_state.boundary_nodes,
        inverse_pe=model.config.inverse_pe,
        k_ratio=model.config.k_ratio,
        lambda_outflow=model.config.lambda_outflow,
        gradient_regularization=model.config.gradient_regularization,
        dirichlet_temperature_star=model.config.dirichlet_temperature_star,
        residual_time_scheme=model.config.residual_time_scheme,
        zero_source_anchor_delta=zero_source_anchor_delta,
        zero_source_anchor_weight=getattr(model.config, "zero_source_anchor_weight", 0.0),
        return_components=True,
    )


def _compute_zero_source_anchor_delta(model, graph):
    """在冷态零源图上前向 PD-GCN，返回输出温度增量。

    若锚定权重为 ``0`` 则跳过额外前向，返回 ``None``。
    """

    weight = float(getattr(model.config, "zero_source_anchor_weight", 0.0))
    if weight <= 0.0:
        return None
    reference_temperature = float(
        getattr(model.config, "zero_source_anchor_reference_temperature_star", 0.0)
    )
    anchor_temperature = torch.full_like(graph.x[:, 6:7], reference_temperature)
    anchor_graph = clone_graph_with_temperature(graph, anchor_temperature)
    if hasattr(anchor_graph, "q_surface_star"):
        anchor_graph.q_surface_star = torch.zeros_like(anchor_graph.q_surface_star)
    if hasattr(anchor_graph, "q_surface"):
        anchor_graph.q_surface = torch.zeros_like(anchor_graph.q_surface)
    if hasattr(anchor_graph, "q_star"):
        anchor_graph.q_star = torch.zeros_like(anchor_graph.q_star)
    return model(anchor_graph)


def _detach_loss_record(components):
    record = {
        "loss_total": float(components["loss_total"].detach().cpu()),
        "loss_pde": float(components["loss_pde"].detach().cpu()),
        "loss_outflow": float(components["loss_outflow"].detach().cpu()),
        "loss_beta": float(components["loss_beta"].detach().cpu()),
        "loss_smooth": float(components["loss_smooth"].detach().cpu()),
    }
    anchor = components.get("loss_zero_source_anchor")
    if anchor is not None:
        record["loss_zero_source_anchor"] = float(anchor.detach().cpu())
    else:
        record["loss_zero_source_anchor"] = 0.0
    return record


def _aggregate_component_records(records):
    return {
        "loss_total": _mean_records(records, "loss_total"),
        "loss_pde": _mean_records(records, "loss_pde"),
        "loss_outflow": _mean_records(records, "loss_outflow"),
        "loss_beta": _mean_records(records, "loss_beta"),
        "loss_smooth": _mean_records(records, "loss_smooth"),
        "loss_zero_source_anchor": _mean_records(records, "loss_zero_source_anchor"),
    }


def _temperature_stats(temperature_star, scale_params: ScaleParams):
    temperature = temperature_from_dimensionless(temperature_star.detach(), scale_params)
    return {
        "temperature_mean": float(temperature.mean().cpu()),
        "temperature_max": float(temperature.max().cpu()),
        "temperature_min": float(temperature.min().cpu()),
        "temperature_var": float(temperature.var(unbiased=False).cpu()),
    }


def _build_monitor_snapshot(graph, residual, temperature_star, scale_params: ScaleParams, *, frame_idx: int):
    return {
        "frame_index": int(frame_idx),
        "coords": graph.pos.detach().cpu().numpy(),
        "edge_index": graph.edge_index.detach().cpu().numpy(),
        "residual": residual.detach().reshape(-1).cpu().numpy(),
        "temperature": temperature_from_dimensionless(temperature_star.detach(), scale_params).reshape(-1).cpu().numpy(),
    }


def _should_capture_frame(frame_idx: int, monitor_frame_index: Optional[int], num_frames: int) -> bool:
    target = int(num_frames) // 2 if monitor_frame_index is None else min(int(monitor_frame_index), int(num_frames) - 1)
    return int(frame_idx) == target


def _mean_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return sum(values) / max(len(values), 1)


def _max_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return max(values) if values else 0.0


def _min_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return min(values) if values else 0.0


@torch.no_grad()
def rollout_static_topology(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    steps: int,
    scale_params: ScaleParams,
    *,
    writer: Optional[Callable[[int, torch.Tensor], None]] = None,
    return_all: bool = False,
    return_dimensionless: bool = False,
    warmup_steps: int = 0,
):
    """使用固定拓扑数据管线进行流式推理。

    参数:
        model: 已训练模型。
        frame_reader: ``HDF5FrameReader``，提供动态基础特征。
        static_state: ``StaticGraphState``，提供固定拓扑。
        feature_builder: ``GpuFeatureBuilder``，构建当前帧图特征。
        steps: 推理步数。
        scale_params: ``ScaleParams``，用于把无量纲温度还原为真实温度。
        writer: 可选回调 ``writer(step, temperature)``；每步输出会先移动到 CPU。
        return_all: 是否把所有步输出堆叠返回；大图默认应保持 ``False``。
        return_dimensionless: ``writer`` 和返回值是否使用无量纲温度。
        warmup_steps: 可选 PD-GCN 伪时间松弛步数；默认 ``0`` 表示冷态初温。

    返回:
        若 ``return_all=False``，返回 ``None``；否则返回形状 ``[steps, N, 1]``
        的温度序列张量。
    """

    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if int(steps) > frame_reader.num_frames:
        raise ValueError(f"steps={steps} exceeds available frames {frame_reader.num_frames}.")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    current_temperature = feature_builder.initial_temperature()
    if int(warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = _snapshot_graph_for_tbptt(
            feature_builder.build(node_base_cpu, global_cpu, current_temperature)
        )
        current_temperature = pseudo_time_relax_initial_temperature(
            model,
            warmup_graph,
            int(warmup_steps),
        )
    outputs = []
    try:
        for frame_idx in range(int(steps)):
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + graph_explicit_source_delta(graph, model.config),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(graph, source_temperature)
            delta_temperature = model(graph)
            if getattr(model.config, "non_heating_projection", True):
                delta_temperature = project_non_heating_delta(
                    delta_temperature,
                    static_state.boundary_nodes,
                )
            next_temperature = apply_dirichlet_boundary(
                source_temperature + delta_temperature,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            output = next_temperature if return_dimensionless else temperature_from_dimensionless(next_temperature, scale_params)
            if writer is not None:
                writer(frame_idx, output.detach().cpu())
            if return_all:
                outputs.append(output.detach().cpu())
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    if return_all:
        return torch.stack(outputs, dim=0)
    return None


def _normalize_vectors(vectors, *, eps: float):
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

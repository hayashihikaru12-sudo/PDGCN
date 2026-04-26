from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch_geometric.data import Data

from data.dimensionless import ScaleParams, temperature_from_dimensionless
from data.static_cache import STATIC_FILE, FrameMemmapReader
from pde import apply_dirichlet_boundary, total_loss

from .config import TrainConfig
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
        self.node_base = torch.empty(static_state.num_nodes, 7, device=self.device, dtype=dtype)
        self.global_raw = torch.empty(1, device=self.device, dtype=dtype)
        self.x = torch.empty(static_state.num_nodes, 8, device=self.device, dtype=dtype)
        self.edge_attr = torch.empty(static_state.num_edges, 7, device=self.device, dtype=dtype)
        self.graph = Data(
            x=self.x,
            edge_index=static_state.edge_index,
            edge_attr=self.edge_attr,
            node_type=static_state.node_type,
            global_attr=self.global_raw,
            pos=self.x[:, 0:3],
        )
        self.graph.num_nodes = static_state.num_nodes
        self.graph.upwind_nodes = static_state.boundary_nodes["upwind"]
        self.graph.downwind_nodes = static_state.boundary_nodes["downwind"]
        self.graph.side_nodes = static_state.boundary_nodes["side"]

    def build(self, node_base_cpu, global_cpu, temperature_star):
        """用当前帧基础特征更新复用图对象。

        参数:
            node_base_cpu: CPU 张量，形状 ``[N, 7]``，列为
                ``[x, y, z, fx, fy, fz, Q]``，仍为真实单位基础特征。
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

        self.node_base.copy_(node_base_cpu, non_blocking=self.device.type == "cuda")
        self.global_raw.copy_(global_cpu[:1], non_blocking=self.device.type == "cuda")

        coords_star = self.node_base[:, 0:3] / float(self.scale_params.L0)
        fibers_unit = _normalize_vectors(self.node_base[:, 3:6], eps=float(self.scale_params.eps))
        q_star = self.node_base[:, 6:7] / float(self.scale_params.Q0)
        temperature = temperature_star.to(device=self.device, dtype=self.dtype, non_blocking=True).reshape(-1, 1)

        self.x[:, 0:3] = coords_star
        self.x[:, 3:6] = fibers_unit
        self.x[:, 6:7] = temperature
        self.x[:, 7:8] = q_star

        source = self.static_state.source
        receiver = self.static_state.receiver
        delta = coords_star[receiver] - coords_star[source]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(float(self.scale_params.eps))
        direction = delta / distance
        fiber_mid = _normalize_vectors(fibers_unit[source] + fibers_unit[receiver], eps=float(self.scale_params.eps))
        cos_phi = torch.sum(fiber_mid * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)

        self.edge_attr[:, 0:3] = delta
        self.edge_attr[:, 3:4] = distance
        self.edge_attr[:, 4:5] = direction[:, 0:1]
        self.edge_attr[:, 5:6] = cos_phi
        self.edge_attr[:, 6:7] = cos_phi.square()

        self.global_raw[:] = self.global_raw / float(self.scale_params.v0)
        self.graph.x = self.x
        self.graph.edge_attr = self.edge_attr
        self.graph.global_attr = self.global_raw
        self.graph.pos = self.x[:, 0:3]
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
    snapshot.num_nodes = graph.num_nodes
    snapshot.upwind_nodes = graph.upwind_nodes
    snapshot.downwind_nodes = graph.downwind_nodes
    snapshot.side_nodes = graph.side_nodes
    return snapshot


def train_static_topology(
    model,
    frame_reader: FrameMemmapReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_callback: Optional[Callable[[dict], None]] = None,
):
    """使用固定拓扑流式数据管线训练 PD-GCN。

    参数:
        model: 待训练模型，输入当前帧图对象并输出 ``delta_T*``。
        frame_reader: ``FrameMemmapReader``，按帧提供 CPU 基础特征。
        static_state: ``StaticGraphState``，提供常驻设备的静态拓扑。
        feature_builder: ``GpuFeatureBuilder``，在设备上构建当前帧特征。
        config: ``TrainConfig``，提供训练轮数、窗口长度、warmup、学习率和设备。
        optimizer: 可选优化器；若为 ``None``，创建 Adam。
        epoch_callback: 可选回调；每个 epoch 结束后接收该 epoch 的历史记录。

    返回:
        训练历史列表；每个元素包含 ``epoch``、平均 ``loss`` 和窗口损失。
    """

    device = static_state.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

    history = []
    for epoch in range(int(config.epochs)):
        model.train()
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
        window_losses = []

        for start in range(0, frame_reader.num_frames, int(config.tbptt_window)):
            end = min(start + int(config.tbptt_window), frame_reader.num_frames)
            optimizer.zero_grad()
            loss_terms = []
            window_temperature = current_temperature

            for frame_idx in range(start, end):
                node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
                graph = _snapshot_graph_for_tbptt(
                    feature_builder.build(node_base_cpu, global_cpu, window_temperature)
                )
                delta_temperature = model(graph)
                next_temperature = apply_dirichlet_boundary(
                    window_temperature + delta_temperature,
                    static_state.boundary_nodes,
                    value=getattr(model.config, "dirichlet_temperature_star", 0.0),
                )
                loss_terms.append(
                    total_loss(
                        T_next=next_temperature,
                        T_current=window_temperature,
                        v_scan_star=graph.global_attr,
                        Q_star=graph.x[:, 7:8],
                        dt_star=model.config.dt_star,
                        edge_index=static_state.edge_index,
                        edge_attr=graph.edge_attr,
                        boundary_nodes=static_state.boundary_nodes,
                        inverse_pe=model.config.inverse_pe,
                        pi_q=model.config.pi_q,
                        k_ratio=model.config.k_ratio,
                        lambda_outflow=model.config.lambda_outflow,
                        dirichlet_temperature_star=model.config.dirichlet_temperature_star,
                    )
                )
                window_temperature = next_temperature

            loss = torch.stack(loss_terms).mean()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            optimizer.step()

            window_losses.append(float(loss.detach().cpu()))
            current_temperature = window_temperature.detach()

        epoch_record = {
            "epoch": epoch,
            "loss": sum(window_losses) / max(len(window_losses), 1),
            "window_losses": window_losses,
        }
        should_stop = config.loss_threshold is not None and epoch_record["loss"] < float(config.loss_threshold)
        if should_stop:
            epoch_record["stopped_early"] = True
            epoch_record["stop_reason"] = "loss_threshold"
        history.append(epoch_record)
        if epoch_callback is not None:
            epoch_callback(epoch_record)
        if should_stop:
            break
    return history


@torch.no_grad()
def rollout_static_topology(
    model,
    frame_reader: FrameMemmapReader,
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
        frame_reader: ``FrameMemmapReader``，提供动态基础特征。
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
            next_temperature = apply_dirichlet_boundary(
                current_temperature + model(graph),
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

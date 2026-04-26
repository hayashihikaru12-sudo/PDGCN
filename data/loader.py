from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import torch


@dataclass(frozen=True)
class GraphRawData:
    xyz: torch.Tensor
    fiber: torch.Tensor
    q: torch.Tensor
    edge_index: torch.Tensor
    boundary_nodes: Dict[str, torch.Tensor]
    frame_idx: int
    num_frames: int


class HDF5Loader:
    """从生成好的 HDF5 数据集中读取单帧 PD-GCN 图数据。"""

    REQUIRED_DATASETS = (
        "dynamic/xyz",
        "dynamic/fiber",
        "dynamic/Q",
        "edge_index",
        "boundary_nodes/upwind",
        "boundary_nodes/downwind",
        "boundary_nodes/side",
    )

    def __init__(self, file_path):
        """初始化 HDF5 数据加载器。

        参数:
            file_path: HDF5 文件路径，类型可为字符串或 ``pathlib.Path``；
                文件应包含 ``dynamic/xyz``、``dynamic/fiber``、``dynamic/Q``、
                ``edge_index`` 和边界节点索引。

        返回:
            None。实例会保存规范化后的 ``Path`` 到 ``self.file_path``。
        """

        self.file_path = Path(file_path)

    def load_graph_data(self, frame_idx: int = 0, device: Optional[torch.device] = None) -> GraphRawData:
        """读取单个时间帧的原始图数据。

        参数:
            frame_idx: 要读取的帧编号，整数范围为 ``[0, num_frames - 1]``。
            device: 可选目标设备；若提供，返回的张量会移动到该设备。

        返回:
            ``GraphRawData`` 数据对象，包含：
            ``xyz`` 形状 ``[N, 3]``、``fiber`` 形状 ``[N, 3]``、
            ``q`` 形状 ``[N, 1]``、``edge_index`` 形状 ``[2, E]``、
            ``boundary_nodes`` 字典以及帧索引信息。
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as h5_file:
            self._validate_required_keys(h5_file)

            xyz_all = h5_file["dynamic/xyz"]
            fiber_all = h5_file["dynamic/fiber"]
            q_all = h5_file["dynamic/Q"]
            num_frames = int(xyz_all.shape[0])

            if not 0 <= frame_idx < num_frames:
                raise IndexError(f"frame_idx must be in [0, {num_frames - 1}], got {frame_idx}.")

            xyz = _as_tensor(xyz_all[frame_idx], dtype=torch.float32, device=device)
            fiber = _as_tensor(fiber_all[frame_idx], dtype=torch.float32, device=device)
            q = _as_tensor(q_all[frame_idx], dtype=torch.float32, device=device)
            edge_index = _as_tensor(h5_file["edge_index"][()], dtype=torch.long, device=device)

            boundary_nodes = {
                name: _as_tensor(h5_file[f"boundary_nodes/{name}"][()], dtype=torch.long, device=device)
                for name in ("upwind", "downwind", "side")
            }

        self._validate_shapes(xyz, fiber, q, edge_index, boundary_nodes)

        return GraphRawData(
            xyz=xyz,
            fiber=fiber,
            q=q,
            edge_index=edge_index,
            boundary_nodes=boundary_nodes,
            frame_idx=frame_idx,
            num_frames=num_frames,
        )

    def _validate_required_keys(self, h5_file):
        """检查 HDF5 文件是否包含 PD-GCN 构图所需字段。

        参数:
            h5_file: 已打开的 ``h5py.File`` 对象。

        返回:
            None。若缺少必需数据集则抛出 ``KeyError``。
        """

        missing = [key for key in self.REQUIRED_DATASETS if key not in h5_file]
        if missing:
            raise KeyError(f"Missing required HDF5 datasets: {missing}")

    @staticmethod
    def _validate_shapes(xyz, fiber, q, edge_index, boundary_nodes):
        """校验单帧图数据的张量形状和索引范围。

        参数:
            xyz: 节点坐标张量，形状必须为 ``[N, 3]``。
            fiber: 节点纤维方向张量，形状必须与 ``xyz`` 相同。
            q: 节点热源张量，形状必须为 ``[N, 1]``。
            edge_index: 图边索引张量，形状必须为 ``[2, E]``。
            boundary_nodes: 边界节点字典，值为一维 long 索引张量。

        返回:
            None。若形状不匹配或索引越界则抛出 ``ValueError``。
        """

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"dynamic/xyz frame must have shape [N, 3], got {tuple(xyz.shape)}.")
        if fiber.shape != xyz.shape:
            raise ValueError(f"dynamic/fiber frame must match xyz shape {tuple(xyz.shape)}, got {tuple(fiber.shape)}.")
        if q.ndim != 2 or q.shape[0] != xyz.shape[0] or q.shape[1] != 1:
            raise ValueError(f"dynamic/Q frame must have shape [N, 1], got {tuple(q.shape)}.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")

        num_nodes = xyz.shape[0]
        if edge_index.numel() > 0:
            edge_min = int(edge_index.min().item())
            edge_max = int(edge_index.max().item())
            if edge_min < 0 or edge_max >= num_nodes:
                raise ValueError(f"edge_index values must be within [0, {num_nodes - 1}], got [{edge_min}, {edge_max}].")

        for name, indices in boundary_nodes.items():
            if indices.ndim != 1:
                raise ValueError(f"boundary_nodes/{name} must be one-dimensional, got {tuple(indices.shape)}.")
            if indices.numel() == 0:
                continue
            idx_min = int(indices.min().item())
            idx_max = int(indices.max().item())
            if idx_min < 0 or idx_max >= num_nodes:
                raise ValueError(
                    f"boundary_nodes/{name} values must be within [0, {num_nodes - 1}], got [{idx_min}, {idx_max}]."
                )


def _as_tensor(data, *, dtype, device=None):
    """将 HDF5/NumPy 数据转换为 PyTorch 张量。

    参数:
        data: 可由 ``torch.as_tensor`` 接收的数据，通常来自 HDF5 数据集。
        dtype: 目标 ``torch.dtype``，例如 ``torch.float32`` 或 ``torch.long``。
        device: 可选目标设备；若提供，则返回张量会移动到该设备。

    返回:
        ``torch.Tensor``，数据类型为 ``dtype``，设备为 ``device`` 或默认设备。
    """

    tensor = torch.as_tensor(data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor

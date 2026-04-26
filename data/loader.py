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
    """Load one PD-GCN graph frame from the generated HDF5 dataset."""

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
        self.file_path = Path(file_path)

    def load_graph_data(self, frame_idx: int = 0, device: Optional[torch.device] = None) -> GraphRawData:
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
        missing = [key for key in self.REQUIRED_DATASETS if key not in h5_file]
        if missing:
            raise KeyError(f"Missing required HDF5 datasets: {missing}")

    @staticmethod
    def _validate_shapes(xyz, fiber, q, edge_index, boundary_nodes):
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
    tensor = torch.as_tensor(data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor

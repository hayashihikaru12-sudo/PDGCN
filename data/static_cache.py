import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from .dimensionless import ScaleParams


STATIC_FILE = "static.pt"
DYNAMIC_NODE_FILE = "dynamic_node_base.npy"
GLOBAL_FILE = "global.npy"
META_FILE = "meta.json"


def build_static_cache(
    h5_path,
    cache_dir,
    scale_params: ScaleParams,
    *,
    scan_velocity: Optional[float] = None,
    overwrite: bool = False,
):
    """从首个 HDF5 切片生成训练集共享的静态拓扑缓存。"""

    h5_path = Path(h5_path)
    cache_dir = Path(cache_dir)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    _prepare_cache_dir(cache_dir, overwrite=overwrite)

    with h5py.File(h5_path, "r") as h5_file:
        _validate_static_h5(h5_file)
        num_frames, num_nodes = _validate_dynamic_shapes(
            h5_path,
            h5_file["dynamic/xyz"].shape,
            h5_file["dynamic/fiber"].shape,
            h5_file["dynamic/Q"].shape,
        )

        edge_index_np = np.asarray(h5_file["edge_index"][()], dtype=np.int64)
        if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {edge_index_np.shape}.")
        if edge_index_np.size and (edge_index_np.min() < 0 or edge_index_np.max() >= num_nodes):
            raise ValueError("edge_index contains node indices outside the valid node range.")

        boundary_nodes = {
            name: torch.as_tensor(h5_file[f"boundary_nodes/{name}"][()], dtype=torch.long)
            for name in ("upwind", "downwind", "side")
        }
        node_type = _build_node_type(num_nodes, boundary_nodes)

    static_payload = {
        "edge_index": torch.as_tensor(edge_index_np, dtype=torch.long),
        "boundary_nodes": boundary_nodes,
        "node_type": node_type,
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index_np.shape[1]),
    }
    torch.save(static_payload, cache_dir / STATIC_FILE)

    meta = {
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index_np.shape[1]),
        "node_feature_size": 8,
        "edge_feature_size": 7,
        "global_size": 1,
        "dynamic_node_base_size": 7,
        "dynamic_node_base_layout": ["x", "y", "z", "fx", "fy", "fz", "Q"],
        "global_layout": ["scan_velocity"],
        "source_h5": str(h5_path),
        "source_num_frames": int(num_frames),
        "scale_params": asdict(scale_params),
    }
    (cache_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_dir


class HDF5FrameReader:
    """按帧从单个 HDF5 切片读取动态基础特征。"""

    def __init__(
        self,
        h5_path,
        *,
        expected_num_nodes: Optional[int] = None,
        scan_velocity: Optional[float] = None,
        pin_memory: bool = True,
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.h5_file = h5py.File(self.h5_path, "r")
        try:
            missing = [
                key
                for key in ("dynamic/xyz", "dynamic/fiber", "dynamic/Q")
                if key not in self.h5_file
            ]
            if missing:
                raise KeyError(
                    f"HDF5 file {self.h5_path} is missing required dynamic datasets: {missing}"
                )

            self.xyz = self.h5_file["dynamic/xyz"]
            self.fiber = self.h5_file["dynamic/fiber"]
            self.q = self.h5_file["dynamic/Q"]
            self.num_frames, self.num_nodes = _validate_dynamic_shapes(
                self.h5_path,
                self.xyz.shape,
                self.fiber.shape,
                self.q.shape,
            )
            if expected_num_nodes is not None and self.num_nodes != int(expected_num_nodes):
                raise ValueError(
                    f"HDF5 file {self.h5_path} has {self.num_nodes} nodes, "
                    f"but static cache expects {expected_num_nodes}."
                )
            self.velocity = _resolve_scan_velocity(self.h5_file, scan_velocity)
        except Exception:
            self.h5_file.close()
            raise

        self.node_feature_size = 7
        self.global_size = 1
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._node_buffer = _allocate_cpu_buffer(
            (self.num_nodes, self.node_feature_size),
            pin_memory=self.pin_memory,
        )
        self._global_buffer = _allocate_cpu_buffer((self.global_size,), pin_memory=self.pin_memory)
        self._node_array = self._node_buffer.numpy()
        self._global_array = self._global_buffer.numpy()

    def read_frame(self, frame_idx: int):
        if not 0 <= int(frame_idx) < self.num_frames:
            raise IndexError(f"frame_idx must be in [0, {self.num_frames - 1}], got {frame_idx}.")
        idx = int(frame_idx)
        self._node_array[:, 0:3] = self.xyz[idx, :, :]
        self._node_array[:, 3:6] = self.fiber[idx, :, :]
        self._node_array[:, 6:7] = self.q[idx, :, :]
        self._global_array[0] = np.float32(self.velocity)
        return self._node_buffer, self._global_buffer

    def close(self):
        self.xyz = None
        self.fiber = None
        self.q = None
        self._node_array = None
        self._global_array = None
        if getattr(self, "h5_file", None) is not None:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


FrameMemmapReader = HDF5FrameReader


def _prepare_cache_dir(cache_dir: Path, *, overwrite: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (STATIC_FILE, DYNAMIC_NODE_FILE, GLOBAL_FILE, META_FILE):
            path = cache_dir / name
            if path.exists():
                path.unlink()


def _validate_static_h5(h5_file):
    required = (
        "dynamic/xyz",
        "dynamic/fiber",
        "dynamic/Q",
        "edge_index",
        "boundary_nodes/upwind",
        "boundary_nodes/downwind",
        "boundary_nodes/side",
    )
    missing = [key for key in required if key not in h5_file]
    if missing:
        raise KeyError(f"Missing required HDF5 datasets: {missing}")


def _validate_dynamic_shapes(h5_path, xyz_shape, fiber_shape, q_shape):
    if len(xyz_shape) != 3 or xyz_shape[2] != 3:
        raise ValueError(f"HDF5 file {h5_path} dynamic/xyz must have shape [T, N, 3], got {xyz_shape}.")
    if fiber_shape != xyz_shape:
        raise ValueError(
            f"HDF5 file {h5_path} dynamic/fiber shape {fiber_shape} must match dynamic/xyz shape {xyz_shape}."
        )
    if len(q_shape) != 3 or q_shape[:2] != xyz_shape[:2] or q_shape[2] != 1:
        raise ValueError(
            f"HDF5 file {h5_path} dynamic/Q shape {q_shape} must be [T, N, 1] matching dynamic/xyz."
        )
    if int(xyz_shape[0]) <= 0:
        raise ValueError(f"HDF5 file {h5_path} must contain at least one frame.")
    return int(xyz_shape[0]), int(xyz_shape[1])


def _resolve_scan_velocity(h5_file, scan_velocity):
    if scan_velocity is not None:
        return float(scan_velocity)
    if "velocity_speed" in h5_file.attrs:
        return float(h5_file.attrs["velocity_speed"])
    return 1.0


def _build_node_type(num_nodes: int, boundary_nodes):
    node_type = torch.zeros(num_nodes, dtype=torch.long)
    selected = [boundary_nodes[name].reshape(-1) for name in ("upwind", "downwind", "side")]
    if selected:
        boundary_index = torch.unique(torch.cat(selected, dim=0))
        node_type[boundary_index] = 1
    return node_type


def _allocate_cpu_buffer(shape, *, pin_memory: bool):
    try:
        return torch.empty(shape, dtype=torch.float32, pin_memory=pin_memory)
    except RuntimeError:
        return torch.empty(shape, dtype=torch.float32)

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
    """从 HDF5 数据生成固定拓扑训练缓存。

    参数:
        h5_path: 输入 HDF5 文件路径，需包含 ``dynamic/xyz``、``dynamic/fiber``、
            ``dynamic/Q``、``edge_index`` 和边界节点。
        cache_dir: 输出缓存目录。
        scale_params: ``ScaleParams`` 实例，会写入元信息用于一致性检查。
        scan_velocity: 可选真实扫描速度；若为 ``None``，优先读取 HDF5 根属性
            ``velocity_speed``，再退化为 ``1.0``。
        overwrite: 缓存文件已存在时是否覆盖。

    返回:
        ``Path``，表示缓存目录路径。
    """

    h5_path = Path(h5_path)
    cache_dir = Path(cache_dir)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    _prepare_cache_dir(cache_dir, overwrite=overwrite)

    with h5py.File(h5_path, "r") as h5_file:
        _validate_h5(h5_file)

        xyz = h5_file["dynamic/xyz"]
        fiber = h5_file["dynamic/fiber"]
        q = h5_file["dynamic/Q"]
        if xyz.shape != fiber.shape:
            raise ValueError(f"dynamic/fiber shape {fiber.shape} must match dynamic/xyz shape {xyz.shape}.")
        if q.shape[:2] != xyz.shape[:2] or q.shape[2:] != (1,):
            raise ValueError(f"dynamic/Q shape {q.shape} must be [T, N, 1] matching dynamic/xyz.")

        num_frames, num_nodes, _ = xyz.shape
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
        velocity = _resolve_scan_velocity(h5_file, scan_velocity)

        dynamic = np.lib.format.open_memmap(
            cache_dir / DYNAMIC_NODE_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(num_frames, num_nodes, 7),
        )
        dynamic[:, :, 0:3] = xyz[:, :, :]
        dynamic[:, :, 3:6] = fiber[:, :, :]
        dynamic[:, :, 6:7] = q[:, :, :]
        dynamic.flush()

        global_condition = np.lib.format.open_memmap(
            cache_dir / GLOBAL_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(num_frames, 1),
        )
        global_condition[:, 0] = np.float32(velocity)
        global_condition.flush()

    static_payload = {
        "edge_index": torch.as_tensor(edge_index_np, dtype=torch.long),
        "boundary_nodes": boundary_nodes,
        "node_type": node_type,
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index_np.shape[1]),
    }
    torch.save(static_payload, cache_dir / STATIC_FILE)

    meta = {
        "num_frames": int(num_frames),
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index_np.shape[1]),
        "dynamic_node_base_shape": [int(num_frames), int(num_nodes), 7],
        "global_shape": [int(num_frames), 1],
        "dynamic_node_base_layout": ["x", "y", "z", "fx", "fy", "fz", "Q"],
        "global_layout": ["scan_velocity"],
        "scale_params": asdict(scale_params),
    }
    (cache_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_dir


class FrameMemmapReader:
    """按帧读取固定拓扑缓存中的动态基础特征。"""

    def __init__(self, cache_dir, *, pin_memory: bool = True):
        """初始化 memmap 帧读取器。

        参数:
            cache_dir: ``build_static_cache`` 生成的缓存目录。
            pin_memory: 在 CUDA 可用时是否把 CPU 暂存张量放入 pinned memory，
                以支持非阻塞 CPU-GPU 传输。

        返回:
            None。实例会打开动态节点特征和全局条件的 memmap。
        """

        self.cache_dir = Path(cache_dir)
        self.meta = json.loads((self.cache_dir / META_FILE).read_text(encoding="utf-8"))
        self.dynamic_node_base = np.load(self.cache_dir / DYNAMIC_NODE_FILE, mmap_mode="r")
        self.global_condition = np.load(self.cache_dir / GLOBAL_FILE, mmap_mode="r")
        self.num_frames = int(self.meta["num_frames"])
        self.num_nodes = int(self.meta["num_nodes"])
        self.node_feature_size = int(self.dynamic_node_base.shape[2])
        self.global_size = int(self.global_condition.shape[1])
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._node_buffer = _allocate_cpu_buffer(
            (self.num_nodes, self.node_feature_size),
            pin_memory=self.pin_memory,
        )
        self._global_buffer = _allocate_cpu_buffer((self.global_size,), pin_memory=self.pin_memory)
        self._node_array = self._node_buffer.numpy()
        self._global_array = self._global_buffer.numpy()

    def read_frame(self, frame_idx: int):
        """读取一个时间帧到可复用 CPU 张量缓冲区。

        参数:
            frame_idx: 时间帧索引，范围为 ``[0, num_frames - 1]``。

        返回:
            ``(node_base, global_condition)`` 二元组；``node_base`` 形状
            ``[N, 7]``，列为 ``[x, y, z, fx, fy, fz, Q]``；
            ``global_condition`` 形状 ``[G]``。
        """

        if not 0 <= int(frame_idx) < self.num_frames:
            raise IndexError(f"frame_idx must be in [0, {self.num_frames - 1}], got {frame_idx}.")
        np.copyto(self._node_array, self.dynamic_node_base[int(frame_idx)])
        np.copyto(self._global_array, self.global_condition[int(frame_idx)])
        return self._node_buffer, self._global_buffer

    def close(self):
        """释放 memmap 引用，便于 Windows 下删除或覆盖缓存文件。

        参数:
            无。

        返回:
            None。函数会清空内部 memmap 和 NumPy 视图引用。
        """

        self.dynamic_node_base = None
        self.global_condition = None
        self._node_array = None
        self._global_array = None

    def __enter__(self):
        """进入上下文管理器。

        参数:
            无。

        返回:
            当前 ``FrameMemmapReader`` 实例。
        """

        return self

    def __exit__(self, exc_type, exc, traceback):
        """退出上下文管理器并释放 memmap 引用。

        参数:
            exc_type: 异常类型；正常退出时为 ``None``。
            exc: 异常对象；正常退出时为 ``None``。
            traceback: 异常堆栈；正常退出时为 ``None``。

        返回:
            ``False``，表示不吞掉上下文中的异常。
        """

        self.close()
        return False


def _prepare_cache_dir(cache_dir: Path, *, overwrite: bool):
    if cache_dir.exists():
        existing = [cache_dir / name for name in (STATIC_FILE, DYNAMIC_NODE_FILE, GLOBAL_FILE, META_FILE)]
        if any(path.exists() for path in existing) and not overwrite:
            raise FileExistsError(f"Cache already exists at {cache_dir}; pass overwrite=True to rebuild it.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (STATIC_FILE, DYNAMIC_NODE_FILE, GLOBAL_FILE, META_FILE):
            path = cache_dir / name
            if path.exists():
                path.unlink()


def _validate_h5(h5_file):
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

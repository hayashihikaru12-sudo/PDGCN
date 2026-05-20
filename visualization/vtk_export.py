from pathlib import Path

import numpy as np


def sanitize_edges(edge_index, num_points: int):
    """Return unique undirected valid edges as an int64 array with shape [M, 2]."""

    if edge_index is None:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError(f"edge_index must have shape [2, E] or [E, 2], got {edges.shape}.")
    if edges.shape[0] == 2:
        edges = edges.T
    elif edges.shape[1] != 2:
        raise ValueError(f"edge_index must have shape [2, E] or [E, 2], got {edges.shape}.")

    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 1] >= 0)
        & (edges[:, 0] < int(num_points))
        & (edges[:, 1] < int(num_points))
        & (edges[:, 0] != edges[:, 1])
    )
    edges = edges[valid]
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def write_polydata_vtk(path, coords, *, point_data, edge_index=None, title: str = "PDGCN VTK export"):
    """Write ParaView-readable legacy ASCII POLYDATA."""

    path = Path(path)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
    num_points = int(coords.shape[0])
    if num_points <= 0:
        raise ValueError("coords must contain at least one point.")

    clean_data = {}
    for name, values in dict(point_data).items():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape[0] != num_points:
            raise ValueError(
                f"point_data[{name!r}] length must match number of points {num_points}, got {array.shape[0]}."
            )
        clean_data[_clean_field_name(name)] = array
    if not clean_data:
        raise ValueError("point_data must contain at least one scalar field.")

    edges = sanitize_edges(edge_index, num_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{_ascii_title(title)}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {num_points} float\n")
        for x, y, z in coords:
            handle.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

        if len(edges) > 0:
            handle.write(f"LINES {len(edges)} {len(edges) * 3}\n")
            for source, target in edges:
                handle.write(f"2 {int(source)} {int(target)}\n")
        else:
            handle.write(f"VERTICES {num_points} {num_points * 2}\n")
            for index in range(num_points):
                handle.write(f"1 {index}\n")

        handle.write(f"POINT_DATA {num_points}\n")
        for name, values in clean_data.items():
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9g}\n")

    return path


def _clean_field_name(name):
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in str(name).strip())
    return cleaned or "scalar"


def _ascii_title(title):
    return str(title).encode("ascii", errors="ignore").decode("ascii")[:255] or "VTK export"

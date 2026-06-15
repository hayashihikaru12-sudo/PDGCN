from pathlib import Path
from xml.sax.saxutils import escape

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


def write_unstructured_cloud_vtk(
    path,
    coords,
    *,
    point_data,
    layer_count: int,
    nodes_per_layer: int,
    edge_index=None,
    title: str = "PDGCN cloud VTK export",
    triangle_edge_factor: float = 2.0,
):
    """Write a filled multilayer cloud as legacy ASCII UNSTRUCTURED_GRID wedge cells."""

    path = Path(path)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
    layer_count = int(layer_count)
    nodes_per_layer = int(nodes_per_layer)
    if layer_count < 2:
        raise ValueError(f"layer_count must be at least 2 for wedge cells, got {layer_count}.")
    if nodes_per_layer < 3:
        raise ValueError(f"nodes_per_layer must be at least 3 for triangulation, got {nodes_per_layer}.")
    expected_points = layer_count * nodes_per_layer
    if coords.shape[0] != expected_points:
        raise ValueError(
            f"coords length must equal layer_count * nodes_per_layer ({expected_points}), got {coords.shape[0]}."
        )

    clean_data = _clean_point_data(point_data, coords.shape[0])
    triangles = triangulate_base_layer(
        coords[:nodes_per_layer],
        edge_index=edge_index,
        nodes_per_layer=nodes_per_layer,
        triangle_edge_factor=triangle_edge_factor,
    )
    if len(triangles) == 0:
        raise ValueError("triangulation produced no valid triangles for unstructured cloud VTK export.")
    cells = _build_wedge_cells(triangles, layer_count, nodes_per_layer)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{_ascii_title(title)}\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {coords.shape[0]} float\n")
        for x, y, z in coords:
            handle.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

        handle.write(f"CELLS {len(cells)} {len(cells) * 7}\n")
        for cell in cells:
            handle.write("6 " + " ".join(str(int(index)) for index in cell) + "\n")

        handle.write(f"CELL_TYPES {len(cells)}\n")
        for _ in cells:
            handle.write("13\n")

        handle.write(f"POINT_DATA {coords.shape[0]}\n")
        for name, values in clean_data.items():
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9g}\n")

    return path


def write_topology_wedge_vtk(
    path,
    coords,
    *,
    point_data,
    layer_count: int,
    nodes_per_layer: int,
    edge_index,
    title: str = "PDGCN topology wedge VTK export",
):
    """Write multilayer topology as VTK WEDGE cells recovered from a triangular edge mesh."""

    path = Path(path)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
    layer_count = int(layer_count)
    nodes_per_layer = int(nodes_per_layer)
    if layer_count < 2:
        raise ValueError(f"layer_count must be at least 2 for wedge cells, got {layer_count}.")
    if nodes_per_layer < 3:
        raise ValueError(f"nodes_per_layer must be at least 3 for triangle topology, got {nodes_per_layer}.")
    expected_points = layer_count * nodes_per_layer
    if coords.shape[0] != expected_points:
        raise ValueError(
            f"coords length must equal layer_count * nodes_per_layer ({expected_points}), got {coords.shape[0]}."
        )

    clean_data = _clean_point_data(point_data, coords.shape[0])
    triangles = triangles_from_edge_index(edge_index, nodes_per_layer, coords=coords[:nodes_per_layer])
    if len(triangles) == 0:
        raise ValueError("edge_index did not contain any recoverable triangular cells for topology wedge VTK export.")
    cells = _build_wedge_cells(triangles, layer_count, nodes_per_layer)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{_ascii_title(title)}\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {coords.shape[0]} float\n")
        for x, y, z in coords:
            handle.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

        handle.write(f"CELLS {len(cells)} {len(cells) * 7}\n")
        for cell in cells:
            handle.write("6 " + " ".join(str(int(index)) for index in cell) + "\n")

        handle.write(f"CELL_TYPES {len(cells)}\n")
        for _ in cells:
            handle.write("13\n")

        handle.write(f"POINT_DATA {coords.shape[0]}\n")
        for name, values in clean_data.items():
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9g}\n")

    return path


def write_surface_vtu(
    path,
    coords,
    *,
    point_data,
    edge_index=None,
    title: str = "PDGCN surface VTU export",
):
    """Write a single surface as XML VTU triangles, falling back to vertex cells."""

    path = Path(path)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
    num_points = int(coords.shape[0])
    if num_points <= 0:
        raise ValueError("coords must contain at least one point.")

    clean_data = _clean_point_data(point_data, num_points)
    cells, cell_types = _surface_cells(coords, edge_index)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write(
            '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        )
        handle.write(f"  <!-- {_xml_comment(title)} -->\n")
        handle.write("  <UnstructuredGrid>\n")
        handle.write(f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{len(cells)}">\n')
        handle.write("      <Points>\n")
        handle.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        handle.write(_format_float_data(coords.reshape(-1), indent="          "))
        handle.write("        </DataArray>\n")
        handle.write("      </Points>\n")
        handle.write("      <Cells>\n")
        handle.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        handle.write(_format_int_data(cells.reshape(-1), indent="          "))
        handle.write("        </DataArray>\n")
        handle.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        handle.write(_format_int_data(np.cumsum([len(cell) for cell in cells]), indent="          "))
        handle.write("        </DataArray>\n")
        handle.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        handle.write(_format_int_data(cell_types, indent="          "))
        handle.write("        </DataArray>\n")
        handle.write("      </Cells>\n")
        active_scalar = next(iter(clean_data))
        handle.write(f'      <PointData Scalars="{escape(active_scalar)}">\n')
        for name, values in clean_data.items():
            handle.write(f'        <DataArray type="Float32" Name="{escape(name)}" format="ascii">\n')
            handle.write(_format_float_data(values, indent="          "))
            handle.write("        </DataArray>\n")
        handle.write("      </PointData>\n")
        handle.write("    </Piece>\n")
        handle.write("  </UnstructuredGrid>\n")
        handle.write("</VTKFile>\n")

    return path


def triangles_from_edge_index(edge_index, num_points: int, *, coords=None, eps: float = 1e-14):
    """Recover triangular cells from an undirected edge graph."""

    edges = sanitize_edges(edge_index, int(num_points))
    if len(edges) == 0:
        return np.empty((0, 3), dtype=np.int64)

    adjacency = [set() for _ in range(int(num_points))]
    for source, target in edges:
        source = int(source)
        target = int(target)
        adjacency[source].add(target)
        adjacency[target].add(source)

    triangles = []
    for source, target in edges:
        source = int(source)
        target = int(target)
        common = adjacency[source].intersection(adjacency[target])
        for third in common:
            third = int(third)
            if target < third and source < target:
                triangles.append((source, target, third))

    if not triangles:
        return np.empty((0, 3), dtype=np.int64)

    triangles = np.asarray(triangles, dtype=np.int64)
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
        triangles = triangles[_triangle_area3d(coords, triangles) > float(eps)]
    return triangles


def _surface_cells(coords, edge_index):
    num_points = int(np.asarray(coords).shape[0])
    triangles = triangles_from_edge_index(edge_index, num_points, coords=coords)
    if len(triangles) == 0:
        triangles = triangulate_base_layer(coords, edge_index=edge_index, nodes_per_layer=num_points)
    if len(triangles) > 0:
        return np.asarray(triangles, dtype=np.int32), np.full((len(triangles),), 5, dtype=np.uint8)
    vertices = np.arange(num_points, dtype=np.int32).reshape(num_points, 1)
    return vertices, np.full((num_points,), 1, dtype=np.uint8)


def triangulate_base_layer(coords, *, edge_index=None, nodes_per_layer=None, triangle_edge_factor: float = 2.0):
    """Triangulate one surface layer by stable 2D projection and filter long triangles."""

    from scipy.spatial import Delaunay

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
    if coords.shape[0] < 3:
        return np.empty((0, 3), dtype=np.int64)

    projected = _project_to_dominant_axes(coords)
    if _projected_rank(projected) < 2:
        return np.empty((0, 3), dtype=np.int64)

    triangles = np.asarray(Delaunay(projected).simplices, dtype=np.int64)
    triangles = _filter_degenerate_triangles(projected, triangles)
    scale = _triangulation_length_scale(coords, projected, edge_index, nodes_per_layer)
    if scale is not None:
        triangles = _filter_triangles_by_length_scale(coords, projected, triangles, scale, triangle_edge_factor)
    return triangles


def _clean_point_data(point_data, num_points: int):
    clean_data = {}
    for name, values in dict(point_data).items():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape[0] != int(num_points):
            raise ValueError(
                f"point_data[{name!r}] length must match number of points {num_points}, got {array.shape[0]}."
            )
        clean_data[_clean_field_name(name)] = array
    if not clean_data:
        raise ValueError("point_data must contain at least one scalar field.")
    return clean_data


def _build_wedge_cells(triangles, layer_count: int, nodes_per_layer: int):
    cells = []
    for layer_index in range(int(layer_count) - 1):
        lower_offset = layer_index * int(nodes_per_layer)
        upper_offset = (layer_index + 1) * int(nodes_per_layer)
        for a, b, c in np.asarray(triangles, dtype=np.int64):
            cells.append([lower_offset + a, lower_offset + b, lower_offset + c, upper_offset + a, upper_offset + b, upper_offset + c])
    return np.asarray(cells, dtype=np.int64)


def _filter_degenerate_triangles(projected, triangles, eps: float = 1e-14):
    a = projected[triangles[:, 0]]
    b = projected[triangles[:, 1]]
    c = projected[triangles[:, 2]]
    ba = b - a
    ca = c - a
    area2 = np.abs(ba[:, 0] * ca[:, 1] - ba[:, 1] * ca[:, 0])
    return triangles[area2 > eps]


def _triangulation_length_scale(coords, projected, edge_index, nodes_per_layer):
    graph_scale = _graph_edge_length_scale(coords, edge_index, nodes_per_layer)
    if graph_scale is not None:
        return graph_scale
    return _nearest_neighbor_length_scale(projected)


def _graph_edge_length_scale(coords, edge_index, nodes_per_layer):
    if edge_index is None or nodes_per_layer is None:
        return None
    edges = sanitize_edges(edge_index, int(nodes_per_layer))
    if len(edges) == 0:
        return None
    lengths = _row_norm(coords[edges[:, 0]] - coords[edges[:, 1]])
    positive = lengths[lengths > 0.0]
    if len(positive) == 0:
        return None
    return float(np.median(positive))


def _nearest_neighbor_length_scale(projected):
    from scipy.spatial import cKDTree

    projected = np.asarray(projected, dtype=np.float64)
    if projected.shape[0] < 2:
        return None
    distances, _ = cKDTree(projected).query(projected, k=2)
    positive = distances[:, 1][distances[:, 1] > 0.0]
    if len(positive) == 0:
        return None
    return float(np.median(positive))


def _filter_triangles_by_length_scale(coords, projected, triangles, scale: float, triangle_edge_factor: float):
    if len(triangles) == 0:
        return triangles
    max_edge = float(scale) * float(triangle_edge_factor)
    max_radius = float(scale) * float(triangle_edge_factor)
    edge_lengths = _triangle_edge_lengths(coords, triangles)
    circumradii = _triangle_circumradii(projected, triangles)
    keep = (np.max(edge_lengths, axis=1) <= max_edge) & (circumradii <= max_radius)
    filtered = np.asarray(triangles, dtype=np.int64)[keep]
    return filtered if len(filtered) > 0 else triangles


def _triangle_edge_lengths(coords, triangles):
    points = coords[np.asarray(triangles, dtype=np.int64)]
    ab = _row_norm(points[:, 0] - points[:, 1])
    bc = _row_norm(points[:, 1] - points[:, 2])
    ca = _row_norm(points[:, 2] - points[:, 0])
    return np.stack([ab, bc, ca], axis=1)


def _triangle_circumradii(projected, triangles, eps: float = 1e-14):
    points = np.asarray(projected, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    a = _row_norm(points[:, 1] - points[:, 2])
    b = _row_norm(points[:, 2] - points[:, 0])
    c = _row_norm(points[:, 0] - points[:, 1])
    area2 = np.abs(
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    )
    return (a * b * c) / np.maximum(2.0 * area2, eps)


def _triangle_area3d(coords, triangles):
    points = np.asarray(coords, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    return 0.5 * _row_norm(cross)


def _project_to_dominant_axes(coords):
    ranges = np.ptp(np.asarray(coords, dtype=np.float64), axis=0)
    axes = np.argsort(ranges)[-2:]
    return coords[:, np.sort(axes)]


def _projected_rank(projected, eps: float = 1e-12):
    projected = np.asarray(projected, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] == 0:
        return 0
    ranges = np.ptp(projected, axis=0)
    return min(int(np.count_nonzero(ranges > float(eps))), 2)


def _row_norm(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.sum(values * values, axis=1))


def _clean_field_name(name):
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in str(name).strip())
    return cleaned or "scalar"


def _ascii_title(title):
    return str(title).encode("ascii", errors="ignore").decode("ascii")[:255] or "VTK export"


def _xml_comment(title):
    return str(title).replace("--", "-").replace("<", "").replace(">", "")[:255] or "VTU export"


def _format_float_data(values, *, indent: str):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return ""
    lines = []
    for start in range(0, array.size, 9):
        chunk = array[start : start + 9]
        lines.append(indent + " ".join(f"{float(value):.9g}" for value in chunk) + "\n")
    return "".join(lines)


def _format_int_data(values, *, indent: str):
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return ""
    lines = []
    for start in range(0, array.size, 18):
        chunk = array[start : start + 18]
        lines.append(indent + " ".join(str(int(value)) for value in chunk) + "\n")
    return "".join(lines)

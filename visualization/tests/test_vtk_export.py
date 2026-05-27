import shutil
import unittest
from pathlib import Path

import numpy as np

from visualization import (
    sanitize_edges,
    triangulate_base_layer,
    triangles_from_edge_index,
    write_polydata_vtk,
    write_topology_wedge_vtk,
    write_unstructured_cloud_vtk,
)
from visualization.vtk_export import _projected_rank


class VTKExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("visualization/tests/_tmp_vtk")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_write_polydata_vtk_with_lines_and_scalars(self):
        path = self.root / "surface.vtk"
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.2]], dtype=np.float32)
        edge_index = np.array([[0, 1, 1, 3, 2], [1, 0, 1, 2, 2]], dtype=np.int64)

        write_polydata_vtk(path, coords, edge_index=edge_index, point_data={"temperature": [300.0, 301.0, 302.0]})

        text = path.read_text(encoding="ascii")
        self.assertIn("DATASET POLYDATA", text)
        self.assertIn("POINTS 3 float", text)
        self.assertIn("LINES 1 3", text)
        self.assertIn("POINT_DATA 3", text)
        self.assertIn("SCALARS temperature float 1", text)

    def test_write_polydata_vtk_without_valid_edges_uses_vertices(self):
        path = self.root / "points.vtk"
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        write_polydata_vtk(path, coords, edge_index=None, point_data={"residual": [0.1, -0.1]})

        text = path.read_text(encoding="ascii")
        self.assertIn("VERTICES 2 4", text)
        self.assertIn("SCALARS residual float 1", text)

    def test_sanitize_edges_filters_invalid_edges(self):
        edges = sanitize_edges(
            np.array([[0, 1, 2, -1, 2], [1, 0, 2, 1, 10]], dtype=np.int64),
            num_points=3,
        )

        self.assertEqual(edges.tolist(), [[0, 1]])

    def test_write_unstructured_cloud_vtk_with_wedge_cells(self):
        path = self.root / "cloud.vtk"
        layer0 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        layer1 = layer0 + np.array([0.0, 0.0, -0.1], dtype=np.float32)
        coords = np.vstack([layer0, layer1])
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)

        write_unstructured_cloud_vtk(
            path,
            coords,
            point_data={"temperature": np.arange(8, dtype=np.float32)},
            layer_count=2,
            nodes_per_layer=4,
            edge_index=edge_index,
        )

        text = path.read_text(encoding="ascii")
        self.assertIn("DATASET UNSTRUCTURED_GRID", text)
        self.assertIn("POINTS 8 float", text)
        self.assertIn("CELLS", text)
        self.assertIn("CELL_TYPES", text)
        self.assertIn("\n13\n", text)
        self.assertIn("SCALARS temperature float 1", text)

    def test_triangles_from_edge_index_recovers_gmsh_triangle_edges(self):
        edge_index = np.array([[0, 1, 2, 0, 1], [1, 2, 0, 3, 3]], dtype=np.int64)

        triangles = triangles_from_edge_index(edge_index, 4)

        self.assertEqual({tuple(row) for row in triangles.tolist()}, {(0, 1, 2), (0, 1, 3)})

    def test_write_topology_wedge_vtk_uses_recovered_triangles(self):
        path = self.root / "topology_wedge.vtk"
        layer0 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        layer1 = layer0 + np.array([0.0, 0.0, -0.1], dtype=np.float32)
        coords = np.vstack([layer0, layer1])
        edge_index = np.array([[0, 1, 3, 0, 2], [1, 3, 0, 2, 3]], dtype=np.int64)

        write_topology_wedge_vtk(
            path,
            coords,
            point_data={"temperature": np.arange(8, dtype=np.float32)},
            layer_count=2,
            nodes_per_layer=4,
            edge_index=edge_index,
        )

        text = path.read_text(encoding="ascii")
        self.assertIn("DATASET UNSTRUCTURED_GRID", text)
        self.assertIn("POINTS 8 float", text)
        self.assertIn("CELLS 2 14", text)
        self.assertIn("\n13\n", text)

    def test_projected_rank_handles_large_point_sets_without_pairwise_matrix(self):
        x = np.linspace(0.0, 1.0, 50000, dtype=np.float64)
        projected = np.column_stack([x, x * 0.25 + 0.1])

        self.assertEqual(_projected_rank(projected), 2)

    def test_triangulate_base_layer_keeps_continuous_square_without_graph_triangle_edges(self):
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)

        triangles = triangulate_base_layer(coords, edge_index=edge_index, nodes_per_layer=len(coords))

        self.assertEqual(len(triangles), 2)

    def test_triangulate_base_layer_filters_large_convex_hull_cap_by_geometry(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        edge_index = np.array(
            [
                [0, 1, 0, 1, 3, 3],
                [1, 2, 3, 4, 4, 5],
            ],
            dtype=np.int64,
        )

        triangles = triangulate_base_layer(coords, edge_index=edge_index, nodes_per_layer=len(coords))

        self.assertGreater(len(triangles), 0)
        for triangle in triangles:
            a, b, c = [int(index) for index in triangle]
            self.assertNotEqual(set((a, b, c)), {2, 4, 5})


if __name__ == "__main__":
    unittest.main()

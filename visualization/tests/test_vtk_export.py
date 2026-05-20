import shutil
import unittest
from pathlib import Path

import numpy as np

from visualization import sanitize_edges, write_polydata_vtk


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


if __name__ == "__main__":
    unittest.main()

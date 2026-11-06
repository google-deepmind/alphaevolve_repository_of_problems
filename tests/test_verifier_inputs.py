# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the notebook verifiers without running search or visualization.

Run from the repository root with:
  python -m unittest discover -s tests

NumPy is the only third-party dependency needed for these tests.
"""

import ast
import contextlib
import io
import itertools
import json
import math
import unittest
from pathlib import Path

import numpy as np


def load_verifier(notebook_name, function_name, helpers=()):
    """Load the actual definitions, leaving expensive notebook drivers unrun."""
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / notebook_name
        / f"{notebook_name}.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    names = {function_name, *helpers}
    namespace = {"np": np, "itertools": itertools, "sqrt": math.sqrt}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or f"def {function_name}(" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        # Imports from math belong to the verifier; omit plotting/search imports.
        nodes = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.ImportFrom) and node.module == "math")
        ]
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace[function_name]
    raise AssertionError(f"No {function_name} definition found in {path}")


class VerifierInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.squares = staticmethod(
            load_verifier("squares_in_square", "evaluate_squares_in_square")
        )
        cls.sphere = staticmethod(
            load_verifier("no_5_on_a_sphere", "verify_3d_construction")
        )
        cls.sarkozy = staticmethod(
            load_verifier("furstenberg_sarkozy", "check_sarkozy_subset")
        )
        cls.triangles = staticmethod(
            load_verifier(
                "subsets_of_the_grid_with_no_isosceles_triangles",
                "verify_construction",
                ("dist_sq",),
            )
        )

    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_square_centers_must_be_finite(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for coordinate in (0, 1):
                with self.subTest(invalid=invalid, coordinate=coordinate):
                    centers = np.array([[0.5, 0.5]])
                    centers[0, coordinate] = invalid
                    self.assertEqual(self.squares(centers, np.array([0.5])), -math.inf)

    def test_square_sides_must_be_finite_and_positive(self):
        for side in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0):
            with self.subTest(side=side):
                self.assertEqual(
                    self.squares(np.array([[0.5, 0.5]]), np.array([side])), -math.inf
                )
        self.assertEqual(
            self.squares(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([1.0, -1.0])),
            -math.inf,
        )

    def test_square_numeric_object_arrays(self):
        centers = np.array([[0.5, 0.5]], dtype=object)
        sides = np.array([1.0], dtype=object)
        self.assertEqual(self.squares(centers, sides), 1.0)
        self.assertEqual(
            self.squares(np.empty((0, 2), dtype=object), np.array([], dtype=object)),
            0.0,
        )
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                self.assertEqual(
                    self.squares(np.array([[invalid, 0.5]], dtype=object), sides),
                    -math.inf,
                )
                self.assertEqual(
                    self.squares(centers, np.array([invalid], dtype=object)),
                    -math.inf,
                )

    def test_square_valid_and_invalid_geometry(self):
        centers = np.array([[0.25, 0.25], [0.75, 0.75]])
        sides = np.array([0.5, 0.5])
        self.assertEqual(self.squares(centers, sides), 1.0)
        np.testing.assert_array_equal(centers, [[0.25, 0.25], [0.75, 0.75]])
        np.testing.assert_array_equal(sides, [0.5, 0.5])
        self.assertEqual(self.squares(np.array([[0.5, 0.5]]), np.array([1.0])), 1.0)
        self.assertEqual(
            self.squares(np.array([[0.5, 0.5]]), np.array([2.0])), -math.inf
        )
        self.assertEqual(self.squares(np.array([[0.5, 0.5]] * 2), sides), -math.inf)
        self.assertEqual(self.squares(np.empty((0, 2)), np.array([])), 0.0)

    def test_sphere_rejects_nonfinite_coordinates_before_combinations(self):
        valid = [
            (0, 1, 5),
            (5, 0, 1),
            (4, 2, 0),
            (5, 3, 5),
            (1, 5, 0),
            (3, 2, 3),
            (3, 1, 1),
        ]
        self.assertTrue(self.sphere(valid))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for coordinate in range(3):
                for count in (1, len(valid)):
                    with self.subTest(
                        invalid=invalid, coordinate=coordinate, count=count
                    ):
                        points = [list(point) for point in valid[:count]]
                        points[0][coordinate] = invalid
                        self.assertFalse(self.sphere(points))

    def test_sphere_preserves_valid_and_degenerate_inputs(self):
        self.assertTrue(self.sphere([]))
        self.assertTrue(self.sphere([(0, 0, 0)]))
        self.assertTrue(self.sphere([(10**400, 0, 0)]))
        self.assertFalse(
            self.sphere([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 2, 0)])
        )

    def test_point_iterators_are_checked_after_validation(self):
        self.assertFalse(self.sphere(iter([(0, 0, 0)] * 5)))
        self.assertFalse(self.triangles(iter([(0, 0), (1, 0), (2, 0)])))
        self.assertTrue(self.sphere(iter([(0, 0, 0)])))
        self.assertTrue(self.triangles(iter([(0, 0), (1, 0), (3, 0)])))

    def test_sarkozy_rejects_nonfinite_residues(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for subset in ([invalid], [0, invalid], [invalid, 0]):
                with self.subTest(subset=subset):
                    self.assertFalse(self.sarkozy(11, subset))

    def test_sarkozy_preserves_integer_arithmetic(self):
        self.assertFalse(self.sarkozy(11, [0, 1]))
        self.assertTrue(self.sarkozy(7, [0, 2], power=3))
        self.assertTrue(self.sarkozy(11, []))
        self.assertTrue(self.sarkozy(11, [10**400]))
        # Conversion to float would erase the difference before taking modulo.
        base = 10**400
        self.assertFalse(self.sarkozy(11, [base, base + 1]))
        self.assertTrue(self.sarkozy(7, [base, base + 2], power=3))
        self.assertTrue(self.sarkozy(7, np.array([0, 2]), power=3))

    def test_triangles_reject_nonfinite_coordinates_before_permutations(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for coordinate in (0, 1):
                for count in (1, 3):
                    with self.subTest(
                        invalid=invalid, coordinate=coordinate, count=count
                    ):
                        points = [[0, 0], [1, 0], [2, 0]][:count]
                        points[-1][coordinate] = invalid
                        self.assertFalse(self.triangles(points))

    def test_triangles_preserve_integer_arithmetic_and_geometry(self):
        self.assertFalse(self.triangles([(0, 0), (1, 0), (2, 0)]))
        self.assertTrue(self.triangles([(0, 0), (1, 0), (3, 0)]))
        self.assertTrue(self.triangles([]))
        self.assertTrue(self.triangles([(10**400, 10**400)]))
        base = 10**100
        self.assertFalse(self.triangles([(base, 0), (base + 1, 0), (base + 2, 0)]))
        self.assertTrue(self.triangles([(base, 0), (base + 1, 0), (base + 3, 0)]))


if __name__ == "__main__":
    unittest.main()

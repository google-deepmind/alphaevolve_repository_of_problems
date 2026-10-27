"""get_score in experiments/difference_bases must not depend on whether 0 was part of the candidate set."""

import json
import os

import numpy as np  # noqa: F401  (used by the notebook cell)
import pytest

NOTEBOOK = os.environ.get(
    "DIFFERENCE_BASES_NOTEBOOK",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "experiments",
        "difference_bases",
        "difference_bases.ipynb",
    ),
)


@pytest.fixture(scope="module")
def cell():
    with open(NOTEBOOK, encoding="utf-8") as handle:
        source = "".join(json.load(handle)["cells"][1]["source"])
    namespace = {}
    exec(
        source, namespace
    )  # defines combine_A_B, get_score and the A, B, L data of the notebook
    return namespace


def test_recorded_notebook_set_keeps_its_score(cell):
    # L already contains 0, so its score must be unchanged.
    assert cell["get_score"](cell["L"]) == pytest.approx(2.639027469506608)
    assert cell["get_score"](cell["A"]) == pytest.approx(8 / 3)


def test_score_does_not_depend_on_an_implicit_zero_or_the_input_order(cell):
    get_score = cell["get_score"]
    # {0, 1, 2, 5} has differences 1..5, so |B|^2 / 5 = 3.2.
    assert get_score([0, 1, 2, 5]) == pytest.approx(3.2)
    assert get_score([1, 2, 5]) == pytest.approx(get_score([0, 1, 2, 5]))
    assert get_score([5, 2, 1]) == pytest.approx(get_score([0, 1, 2, 5]))

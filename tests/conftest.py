import numpy as np
import pytest

from hexaform.background.cartesian import CartesianGrid
from hexaform.packing.rsa import PackingSpec, pack


@pytest.fixture(scope="session")
def small_grid():
    return CartesianGrid.cube(1.0, 16)


@pytest.fixture(scope="session")
def packed(small_grid):
    rng = np.random.default_rng(11)
    return pack(small_grid, PackingSpec(n_particles=4, radius_mean=0.10, radius_std=0.02), rng)

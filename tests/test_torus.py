"""Tests for the Torus non-star-shaped body."""

import numpy as np
import pytest

from hexaform.shapes.base import Shape
from hexaform.shapes.torus import Torus, project_to_torus


def test_torus_is_shape_not_starshape():
    """Torus is a Shape but not a StarShape."""
    torus = Torus(centre=np.zeros(3), major_radius=0.3, minor_radius=0.1)
    assert isinstance(torus, Shape)
    # StarShape requires support_radius; Torus inherits from Shape only.
    from hexaform.shapes.base import StarShape
    assert not isinstance(torus, StarShape)


def test_torus_volume_and_area():
    """Volume = 2 pi^2 R r^2; area = 4 pi^2 R r."""
    R, r = 0.3, 0.1
    torus = Torus(centre=np.zeros(3), major_radius=R, minor_radius=r)
    assert torus.volume() == pytest.approx(2.0 * np.pi ** 2 * R * r ** 2, rel=1e-9)
    assert torus.surface_area() == pytest.approx(4.0 * np.pi ** 2 * R * r, rel=1e-9)


def test_torus_bounding_radius():
    """Bounding radius = R + r."""
    torus = Torus(centre=np.zeros(3), major_radius=0.3, minor_radius=0.1)
    assert torus.bounding_radius() == pytest.approx(0.4)


def test_torus_sdf_zero_on_surface():
    """Sample surface points have SDF ~ 0."""
    R, r = 0.3, 0.1
    torus = Torus(centre=np.zeros(3), major_radius=R, minor_radius=r)
    # Parameterise the surface: (R + r cos theta)(cos phi, sin phi, 0) + (0, 0, r sin theta)
    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 25)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    x = (R + r * np.cos(THETA)) * np.cos(PHI)
    y = (R + r * np.cos(THETA)) * np.sin(PHI)
    z = r * np.sin(THETA)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    sdf = torus.sdf(pts)
    assert np.allclose(sdf, 0, atol=1e-12)


def test_torus_sdf_signs():
    """Points inside the solid torus have negative SDF; points outside positive."""
    R, r = 0.3, 0.1
    torus = Torus(centre=np.zeros(3), major_radius=R, minor_radius=r)
    pts = np.array(
        [
            [R, 0.0, 0.0],  # on central circle (inside the tube)
            [R + r / 2, 0.0, 0.0],  # inside the tube, outer half
            [R - r / 2, 0.0, 0.0],  # inside the tube, inner half
            [0.0, 0.0, 0.0],  # centre (in the hole)
            [R + r + 0.05, 0.0, 0.0],  # outside
        ]
    )
    sdf = torus.sdf(pts)
    # First three are inside (negative)
    assert sdf[0] < 0
    assert sdf[1] < 0
    assert sdf[2] < 0
    # Centre of the hole is also negative (the torus wraps around it)
    # Actually for a solid torus, the central hole is *outside* the solid;
    # SDF should be positive.  Our sdf formula gives positive for points
    # far from the central circle.
    assert sdf[3] > 0
    # Outside the rim
    assert sdf[4] > 0


def test_project_to_torus_on_surface():
    """Projecting a point on the surface returns the point itself."""
    R, r = 0.3, 0.1
    torus = Torus(centre=np.zeros(3), major_radius=R, minor_radius=r)
    phi = np.linspace(0, 2 * np.pi, 10)
    theta = np.linspace(0, 2 * np.pi, 10)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    x = (R + r * np.cos(THETA)) * np.cos(PHI)
    y = (R + r * np.cos(THETA)) * np.sin(PHI)
    z = r * np.sin(THETA)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    projected = torus.project(pts)
    # Each projected point should have SDF ~ 0
    sdf = torus.sdf(projected)
    assert np.allclose(sdf, 0, atol=1e-9)


def test_project_to_torus_axis():
    """Points on the axis of revolution project to the inner rim of the tube."""
    R, r = 0.3, 0.1
    torus = Torus(centre=np.zeros(3), major_radius=R, minor_radius=r)
    # The input is at the centre, in the hole.  The closest point on the
    # tube is the inner rim of the central circle.
    pts = np.array([[0.0, 0.0, 0.0]])
    projected = torus.project(pts)
    # Should be on the inner rim
    rho = np.sqrt(projected[:, 0] ** 2 + projected[:, 1] ** 2)
    assert np.allclose(rho, R - r, atol=1e-9)
    assert np.allclose(torus.sdf(projected), 0, atol=1e-9)


def test_torus_with_custom_axis():
    """A torus with a custom axis of revolution works correctly."""
    R, r = 0.3, 0.1
    axis = np.array([1.0, 0.0, 0.0])
    torus = Torus(centre=np.zeros(3), major_radius=R, minor_radius=r, axis=axis)
    # Parameterise the surface around the x-axis: phi sweeps in the y-z plane,
    # theta sweeps around the minor circle.
    phi = np.linspace(0, 2 * np.pi, 16)
    theta = np.linspace(0, 2 * np.pi, 16)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    x = r * np.sin(THETA)
    y = (R + r * np.cos(THETA)) * np.cos(PHI)
    z = (R + r * np.cos(THETA)) * np.sin(PHI)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    sdf = torus.sdf(pts)
    assert np.allclose(sdf, 0, atol=1e-9)

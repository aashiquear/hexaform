"""Element quality and isoparametric volume metrics for HEX8/HEX20/HEX27."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from .elements import REFERENCE_COORDS, element_type_from_width, normalize_element_type
from .mesh import HexMesh


def _shape_derivatives(element_type: str, samples: np.ndarray) -> np.ndarray:
    """Return ``(n_samples, n_nodes, 3)`` derivatives in reference space."""
    kind = normalize_element_type(element_type)
    x = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    r, s, t = x[:, 0], x[:, 1], x[:, 2]
    refs = REFERENCE_COORDS[kind]
    out = np.empty((x.shape[0], refs.shape[0], 3), dtype=np.float64)

    if kind == "HEX8":
        for n, (ri, si, ti) in enumerate(refs):
            out[:, n, 0] = ri * (1 + si * s) * (1 + ti * t) / 8
            out[:, n, 1] = si * (1 + ri * r) * (1 + ti * t) / 8
            out[:, n, 2] = ti * (1 + ri * r) * (1 + si * s) / 8
        return out

    if kind == "HEX20":
        for n, (ri, si, ti) in enumerate(refs):
            if ri and si and ti:
                a, b, c = 1 + ri * r, 1 + si * s, 1 + ti * t
                q = ri * r + si * s + ti * t - 2
                out[:, n, 0] = ri * b * c * (q + a) / 8
                out[:, n, 1] = si * a * c * (q + b) / 8
                out[:, n, 2] = ti * a * b * (q + c) / 8
            elif ri == 0:
                b, c = 1 + si * s, 1 + ti * t
                out[:, n, 0] = -r * b * c / 2
                out[:, n, 1] = (1 - r * r) * si * c / 4
                out[:, n, 2] = (1 - r * r) * b * ti / 4
            elif si == 0:
                a, c = 1 + ri * r, 1 + ti * t
                out[:, n, 0] = (1 - s * s) * ri * c / 4
                out[:, n, 1] = -s * a * c / 2
                out[:, n, 2] = (1 - s * s) * a * ti / 4
            else:
                a, b = 1 + ri * r, 1 + si * s
                out[:, n, 0] = (1 - t * t) * ri * b / 4
                out[:, n, 1] = (1 - t * t) * a * si / 4
                out[:, n, 2] = -t * a * b / 2
        return out

    lr, dr = _lagrange_1d(r)
    ls, ds = _lagrange_1d(s)
    lt, dt = _lagrange_1d(t)
    for n, (ri, si, ti) in enumerate(refs):
        ir, js, kt = int(ri + 1), int(si + 1), int(ti + 1)
        out[:, n, 0] = dr[:, ir] * ls[:, js] * lt[:, kt]
        out[:, n, 1] = lr[:, ir] * ds[:, js] * lt[:, kt]
        out[:, n, 2] = lr[:, ir] * ls[:, js] * dt[:, kt]
    return out


def _lagrange_1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([0.5 * x * (x - 1), 1 - x * x, 0.5 * x * (x + 1)], axis=1)
    derivs = np.stack([x - 0.5, -2 * x, x + 0.5], axis=1)
    return values, derivs


def _jacobians(
    points: np.ndarray,
    cells: np.ndarray,
    derivatives: np.ndarray,
    chunk_size: int = 10_000,
):
    """Yield element chunks of Jacobian matrices ``(e, q, 3, 3)``."""
    for start in range(0, cells.shape[0], chunk_size):
        conn = cells[start : start + chunk_size]
        p = points[conn]
        yield start, np.einsum("eni,qnj->eqij", p, derivatives, optimize=True)


def _quality_samples(element_type: str) -> np.ndarray:
    if element_type == "HEX8":
        return np.concatenate([REFERENCE_COORDS["HEX8"], np.zeros((1, 3))], axis=0)
    gauss = np.sqrt(3.0 / 5.0) * np.array([-1.0, 0.0, 1.0])
    grid = np.stack(np.meshgrid(gauss, gauss, gauss, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.concatenate([REFERENCE_COORDS["HEX8"], grid], axis=0)


def scaled_jacobian(
    points: np.ndarray, cells: np.ndarray, element_type: str | None = None
) -> np.ndarray:
    """Minimum sampled isoparametric scaled Jacobian per element."""
    cells = np.asarray(cells, dtype=np.int64)
    kind = normalize_element_type(element_type or element_type_from_width(cells.shape[1]))
    derivatives = _shape_derivatives(kind, _quality_samples(kind))
    result = np.empty(cells.shape[0], dtype=np.float64)
    for start, jac in _jacobians(np.asarray(points), cells, derivatives):
        det = np.linalg.det(jac)
        denom = np.prod(np.linalg.norm(jac, axis=2), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(denom > 0, det / denom, -1.0)
        result[start : start + jac.shape[0]] = values.min(axis=1)
    return result


def edge_ratio(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Longest/shortest corner edge of each element (1.0 is ideal)."""
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    p = np.asarray(points)[np.asarray(cells)[:, :8]]
    lengths = np.stack(
        [np.linalg.norm(p[:, a, :] - p[:, b, :], axis=1) for a, b in edges], axis=1
    )
    lo = lengths.min(axis=1)
    return np.where(lo > 0, lengths.max(axis=1) / np.maximum(lo, 1e-300), np.inf)


def element_volumes(
    points: np.ndarray, cells: np.ndarray, element_type: str | None = None
) -> np.ndarray:
    """Signed isoparametric element volumes from 3x3x3 Gauss integration."""
    cells = np.asarray(cells, dtype=np.int64)
    kind = normalize_element_type(element_type or element_type_from_width(cells.shape[1]))
    locations = np.sqrt(3.0 / 5.0) * np.array([-1.0, 0.0, 1.0])
    weights_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    samples = np.stack(
        np.meshgrid(locations, locations, locations, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    weights = (
        weights_1d[:, None, None] * weights_1d[None, :, None] * weights_1d[None, None, :]
    ).reshape(-1)
    derivatives = _shape_derivatives(kind, samples)
    result = np.empty(cells.shape[0], dtype=np.float64)
    for start, jac in _jacobians(np.asarray(points), cells, derivatives):
        result[start : start + jac.shape[0]] = np.linalg.det(jac) @ weights
    return result


@dataclass
class QualityReport:
    n_elements: int
    min_scaled_jacobian: float
    mean_scaled_jacobian: float
    p01_scaled_jacobian: float
    n_inverted: int
    n_below_threshold: int
    threshold: float
    max_edge_ratio: float
    mean_edge_ratio: float
    min_volume: float
    total_volume: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        status = "PASS" if self.n_inverted == 0 and self.n_below_threshold == 0 else "FAIL"
        return (
            f"[{status}] elements={self.n_elements}  "
            f"scaled Jacobian min={self.min_scaled_jacobian:.4f} "
            f"mean={self.mean_scaled_jacobian:.4f} p01={self.p01_scaled_jacobian:.4f}  "
            f"inverted={self.n_inverted}  below {self.threshold:g}={self.n_below_threshold}  "
            f"max edge ratio={self.max_edge_ratio:.3f}"
        )


def assess(mesh: HexMesh, threshold: float = 0.2) -> QualityReport:
    """Compute the standard Hexaform quality report for a mesh."""
    sj = scaled_jacobian(mesh.points, mesh.cells, mesh.element_type)
    er = edge_ratio(mesh.points, mesh.cells)
    vol = element_volumes(mesh.points, mesh.cells, mesh.element_type)
    return QualityReport(
        n_elements=mesh.n_elements,
        min_scaled_jacobian=float(sj.min()) if sj.size else float("nan"),
        mean_scaled_jacobian=float(sj.mean()) if sj.size else float("nan"),
        p01_scaled_jacobian=float(np.percentile(sj, 1.0)) if sj.size else float("nan"),
        n_inverted=int(np.count_nonzero(sj <= 0.0)),
        n_below_threshold=int(np.count_nonzero(sj < threshold)),
        threshold=float(threshold),
        max_edge_ratio=float(er.max()) if er.size else float("nan"),
        mean_edge_ratio=float(er.mean()) if er.size else float("nan"),
        min_volume=float(vol.min()) if vol.size else float("nan"),
        total_volume=float(vol.sum()) if vol.size else 0.0,
    )

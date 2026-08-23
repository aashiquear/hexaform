"""Native Exodus II reader/writer.

Both source reviews arrive at the same conclusion about output: Exodus must be
treated as a **first-class data model**, not a conversion target.  The
ScholarGPT review specifically cites published experience where a generic
converter's ExodusII output did not produce a working file and the authors fell
back to writing netCDF directly.  Hexaform therefore writes Exodus itself.

Exodus II is netCDF-3 classic format.  Two backends are supported and probed in
order:

``netCDF4``
    Preferred; written with ``format="NETCDF3_64BIT_OFFSET"`` so files above the
    2 GB classic limit still work.
``scipy.io.netcdf_file``
    Pure-SciPy fallback, so a Hexaform install needs no compiled netCDF library.

What is written
---------------
Coordinates, HEX8/HEX20/HEX27 element blocks (with ``elem_type`` attributes), element and
node number maps, named node sets and side sets, an optional per-element
variable (``particle_id``), and a QA record.  This is exactly the metadata an
elastoplastic phase-field fracture solver needs to apply boundary conditions,
assign two materials and post-process per-particle quantities.

Conventions
-----------
* Node and element ids in the file are **1-based**; ``HexMesh`` is 0-based.
* Side set side ids are 1..6 in the Exodus hexahedron face order (see
  :mod:`hexaform.core.elements`).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.elements import ELEMENT_TYPES, element_type_from_width, normalize_element_type
from ..core.mesh import HexMesh

EXODUS_API_VERSION = 5.22
EXODUS_VERSION = 5.22
LEN_STRING = 33
LEN_LINE = 81


class _Backend:
    """Thin uniform wrapper over the two supported netCDF implementations."""

    def __init__(self, path: Path, prefer: str = "auto"):
        self.path = Path(path)
        self.kind = None
        errors: List[str] = []

        if prefer in ("auto", "netcdf4"):
            try:
                import netCDF4  # type: ignore

                self._ds = netCDF4.Dataset(
                    str(self.path), "w", format="NETCDF3_64BIT_OFFSET"
                )
                self.kind = "netcdf4"
            except Exception as exc:  # pragma: no cover - environment dependent
                errors.append(f"netCDF4: {exc}")
        if self.kind is None and prefer in ("auto", "scipy"):
            try:
                from scipy.io import netcdf_file  # type: ignore

                self._ds = netcdf_file(str(self.path), "w", version=2)
                self.kind = "scipy"
            except Exception as exc:  # pragma: no cover
                errors.append(f"scipy: {exc}")
        if self.kind is None:
            raise RuntimeError(
                "no usable netCDF backend for Exodus output; tried: " + "; ".join(errors)
            )

    # -- uniform API ---------------------------------------------------
    def dim(self, dimname: str, size: int | None) -> None:
        self._ds.createDimension(dimname, size)

    def var(self, varname: str, dtype: str, dims: Sequence[str], data=None, **attrs):
        v = self._ds.createVariable(varname, dtype, tuple(dims))
        for key, value in attrs.items():
            # netCDF4 reserves some Python attribute names (notably ``name``),
            # but Exodus genuinely uses ``name = "ID"`` on its property arrays.
            if hasattr(v, "setncattr"):
                v.setncattr(key, value)
            else:
                setattr(v, key, value)
        if data is not None:
            v[...] = data
        return v

    def set_global(self, **attrs) -> None:
        for key, value in attrs.items():
            setattr(self._ds, key, value)

    def close(self) -> None:
        self._ds.close()


def _pad_names(names: Sequence[str], width: int = LEN_STRING) -> np.ndarray:
    """Encode names into the fixed-width char arrays Exodus expects."""
    out = np.zeros((len(names), width), dtype="S1")
    for i, name in enumerate(names):
        raw = str(name).encode("ascii", "replace")[: width - 1]
        for j, ch in enumerate(raw):
            out[i, j] = bytes([ch])
    return out


def write(
    mesh: HexMesh,
    path: str | Path,
    title: str = "Hexaform mesh",
    element_variables: Sequence[str] | None = None,
    backend: str = "auto",
) -> Path:
    """Write ``mesh`` to an Exodus II ``.e`` file.

    Parameters
    ----------
    element_variables:
        Which of ``mesh.element_fields`` to emit as Exodus element variables at
        time step 1.  Defaults to all of them.
    backend:
        ``"auto"``, ``"netcdf4"`` or ``"scipy"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = _Backend(path, backend)

    block_ids = [int(b) for b in mesh.block_ids]
    ss_names = list(mesh.side_sets.keys())
    ns_names = list(mesh.node_sets.keys())
    ev_names = list(element_variables if element_variables is not None else mesh.element_fields)
    ev_names = [n for n in ev_names if n in mesh.element_fields]

    try:
        ds.set_global(
            api_version=np.float32(EXODUS_API_VERSION),
            version=np.float32(EXODUS_VERSION),
            floating_point_word_size=np.int32(8),
            file_size=np.int32(1),
            int64_status=np.int32(0),
            title=str(title)[: LEN_LINE - 1],
        )

        # -- dimensions -------------------------------------------------
        # The unlimited dimension must be declared first: scipy's classic
        # netCDF writer only allows the *first* dimension to be unlimited, and
        # a record dimension that is declared but never written leaves an
        # unreadable file.  So it is only created when results are emitted.
        if ev_names:
            ds.dim("time_step", None)
        ds.dim("len_string", LEN_STRING)
        ds.dim("len_line", LEN_LINE)
        ds.dim("four", 4)
        ds.dim("num_dim", 3)
        ds.dim("num_nodes", mesh.n_nodes)
        ds.dim("num_elem", mesh.n_elements)
        ds.dim("num_el_blk", len(block_ids))
        if ns_names:
            ds.dim("num_node_sets", len(ns_names))
        if ss_names:
            ds.dim("num_side_sets", len(ss_names))

        if ev_names:
            ds.var("time_whole", "f8", ("time_step",))[0] = 0.0

        # -- coordinates ------------------------------------------------
        ds.var("coordx", "f8", ("num_nodes",), mesh.points[:, 0])
        ds.var("coordy", "f8", ("num_nodes",), mesh.points[:, 1])
        ds.var("coordz", "f8", ("num_nodes",), mesh.points[:, 2])
        ds.var("coor_names", "S1", ("num_dim", "len_string"), _pad_names(["x", "y", "z"]))

        # -- element blocks ---------------------------------------------
        ds.var("eb_prop1", "i4", ("num_el_blk",), np.array(block_ids, dtype=np.int32),
               name="ID")
        ds.var("eb_status", "i4", ("num_el_blk",), np.ones(len(block_ids), dtype=np.int32))
        ds.var(
            "eb_names", "S1", ("num_el_blk", "len_string"),
            _pad_names([mesh.block_name(b) for b in block_ids]),
        )

        elem_order = []
        for n, bid in enumerate(block_ids, start=1):
            sel = np.flatnonzero(mesh.block_id == bid)
            elem_order.append(sel)
            ds.dim(f"num_el_in_blk{n}", sel.size)
            ds.dim(f"num_nod_per_el{n}", mesh.nodes_per_element)
            ds.var(
                f"connect{n}", "i4",
                (f"num_el_in_blk{n}", f"num_nod_per_el{n}"),
                (mesh.cells[sel] + 1).astype(np.int32),
                elem_type=mesh.element_type,
            )
        elem_order_flat = np.concatenate(elem_order) if elem_order else np.zeros(0, dtype=np.int64)

        # Exodus stores elements grouped by block; the maps record the original ids.
        ds.var("elem_num_map", "i4", ("num_elem",), (elem_order_flat + 1).astype(np.int32))
        ds.var("node_num_map", "i4", ("num_nodes",),
               np.arange(1, mesh.n_nodes + 1, dtype=np.int32))

        # position of each original element within the file ordering
        file_pos = np.empty(mesh.n_elements, dtype=np.int64)
        file_pos[elem_order_flat] = np.arange(elem_order_flat.size)

        # -- node sets ---------------------------------------------------
        if ns_names:
            ds.var("ns_prop1", "i4", ("num_node_sets",),
                   np.arange(1, len(ns_names) + 1, dtype=np.int32), name="ID")
            ds.var("ns_status", "i4", ("num_node_sets",),
                   np.ones(len(ns_names), dtype=np.int32))
            ds.var("ns_names", "S1", ("num_node_sets", "len_string"), _pad_names(ns_names))
            for n, name in enumerate(ns_names, start=1):
                nodes = mesh.node_sets[name]
                ds.dim(f"num_nod_ns{n}", nodes.size)
                ds.var(f"node_ns{n}", "i4", (f"num_nod_ns{n}",),
                       (nodes + 1).astype(np.int32))

        # -- side sets ---------------------------------------------------
        if ss_names:
            ds.var("ss_prop1", "i4", ("num_side_sets",),
                   np.arange(1, len(ss_names) + 1, dtype=np.int32), name="ID")
            ds.var("ss_status", "i4", ("num_side_sets",),
                   np.ones(len(ss_names), dtype=np.int32))
            ds.var("ss_names", "S1", ("num_side_sets", "len_string"), _pad_names(ss_names))
            for n, name in enumerate(ss_names, start=1):
                elems, sides = mesh.side_sets[name]
                ds.dim(f"num_side_ss{n}", elems.size)
                ds.var(f"elem_ss{n}", "i4", (f"num_side_ss{n}",),
                       (file_pos[elems] + 1).astype(np.int32))
                ds.var(f"side_ss{n}", "i4", (f"num_side_ss{n}",), sides.astype(np.int32))

        # -- element variables -------------------------------------------
        if ev_names:
            ds.dim("num_elem_var", len(ev_names))
            ds.var("name_elem_var", "S1", ("num_elem_var", "len_string"),
                   _pad_names(ev_names))
            ds.var("elem_var_tab", "i4", ("num_el_blk", "num_elem_var"),
                   np.ones((len(block_ids), len(ev_names)), dtype=np.int32))
            for vi, vname in enumerate(ev_names, start=1):
                values = np.asarray(mesh.element_fields[vname], dtype=np.float64)
                for bi, sel in enumerate(elem_order, start=1):
                    ds.var(
                        f"vals_elem_var{vi}eb{bi}", "f8",
                        ("time_step", f"num_el_in_blk{bi}"),
                    )[0, ...] = values[sel]

        # -- QA record ----------------------------------------------------
        ds.dim("num_qa_rec", 1)
        stamp = _dt.datetime.now()
        ds.var(
            "qa_records", "S1", ("num_qa_rec", "four", "len_string"),
            _pad_names(
                ["hexaform", _hexaform_version(), stamp.strftime("%m/%d/%Y"),
                 stamp.strftime("%H:%M:%S")]
            ).reshape(1, 4, LEN_STRING),
        )
    finally:
        ds.close()
    return path


def _hexaform_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0"


# ----------------------------------------------------------------------
# reader (round-trip validation)
# ----------------------------------------------------------------------
def read(path: str | Path) -> HexMesh:
    """Read a homogeneous Exodus II HEX8/HEX20/HEX27 file into a :class:`HexMesh`.

    Used by the validation stage: the ScholarGPT review's step 11 requires
    verifying node numbering, connectivity, blocks, sets and Jacobians *after*
    re-reading the file rather than trusting the in-memory mesh.
    """
    path = Path(path)
    try:
        import netCDF4  # type: ignore

        ds = netCDF4.Dataset(str(path), "r")
        close = ds.close
        getvar = lambda n: np.asarray(ds.variables[n][:])  # noqa: E731
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        has = lambda n: n in ds.variables  # noqa: E731
        get_attr = lambda n, a: getattr(ds.variables[n], a, None)  # noqa: E731
    except Exception:
        from scipy.io import netcdf_file  # type: ignore

        ds = netcdf_file(str(path), "r", mmap=False)
        close = ds.close
        getvar = lambda n: np.asarray(ds.variables[n][:])  # noqa: E731
        dims = dict(ds.dimensions)
        has = lambda n: n in ds.variables  # noqa: E731
        get_attr = lambda n, a: getattr(ds.variables[n], a, None)  # noqa: E731

    try:
        points = np.stack([getvar("coordx"), getvar("coordy"), getvar("coordz")], axis=1)
        block_ids = getvar("eb_prop1").astype(int).tolist()
        names = _decode_names(getvar("eb_names")) if has("eb_names") else []

        cells: List[np.ndarray] = []
        block_of: List[np.ndarray] = []
        block_names: Dict[int, str] = {}
        mesh_element_type: str | None = None
        for n, bid in enumerate(block_ids, start=1):
            conn = getvar(f"connect{n}").astype(np.int64) - 1
            etype = get_attr(f"connect{n}", "elem_type")
            if isinstance(etype, bytes):
                etype = etype.decode()
            block_type = (
                normalize_element_type(str(etype))
                if etype
                else element_type_from_width(conn.shape[1])
            )
            if conn.shape[1] != ELEMENT_TYPES[block_type]:
                raise ValueError(
                    f"block {bid} declares {block_type} but has {conn.shape[1]} nodes per element"
                )
            if mesh_element_type is not None and block_type != mesh_element_type:
                raise ValueError("mixed hexahedral element types are not supported")
            mesh_element_type = block_type
            cells.append(conn)
            block_of.append(np.full(conn.shape[0], bid, dtype=np.int32))
            if n - 1 < len(names):
                block_names[bid] = names[n - 1]

        mesh = HexMesh(
            points=points,
            cells=(
                np.concatenate(cells, axis=0)
                if cells
                else np.zeros((0, ELEMENT_TYPES[mesh_element_type or "HEX8"]), np.int64)
            ),
            block_id=np.concatenate(block_of) if block_of else np.zeros(0, np.int32),
            element_type=mesh_element_type or "HEX8",
            block_names=block_names,
        )

        n_ns = dims.get("num_node_sets", 0)
        ns_names = _decode_names(getvar("ns_names")) if (n_ns and has("ns_names")) else []
        for n in range(1, n_ns + 1):
            label = ns_names[n - 1] if n - 1 < len(ns_names) else f"nodeset_{n}"
            mesh.node_sets[label] = getvar(f"node_ns{n}").astype(np.int64) - 1

        n_ss = dims.get("num_side_sets", 0)
        ss_names = _decode_names(getvar("ss_names")) if (n_ss and has("ss_names")) else []
        for n in range(1, n_ss + 1):
            label = ss_names[n - 1] if n - 1 < len(ss_names) else f"sideset_{n}"
            mesh.side_sets[label] = (
                getvar(f"elem_ss{n}").astype(np.int64) - 1,
                getvar(f"side_ss{n}").astype(np.int32),
            )
        return mesh
    finally:
        close()


def _decode_names(arr: np.ndarray) -> List[str]:
    out = []
    for row in np.asarray(arr):
        chars = [c.decode() if isinstance(c, bytes) else str(c) for c in row.ravel()]
        out.append("".join(chars).replace("\x00", "").strip())
    return out


def round_trip_check(mesh: HexMesh, path: str | Path) -> Dict[str, object]:
    """Write, re-read and compare; returns a dictionary of differences."""
    back = read(path)
    coord_err = (
        float(np.abs(np.sort(back.points, axis=0) - np.sort(mesh.points, axis=0)).max())
        if back.n_nodes == mesh.n_nodes
        else float("inf")
    )
    counts_in = {mesh.block_name(b): int((mesh.block_id == b).sum()) for b in mesh.block_ids}
    counts_out = {back.block_name(b): int((back.block_id == b).sum()) for b in back.block_ids}
    return {
        "nodes_match": back.n_nodes == mesh.n_nodes,
        "elements_match": back.n_elements == mesh.n_elements,
        "element_type_match": back.element_type == mesh.element_type,
        "blocks_match": counts_in == counts_out,
        "side_sets_match": {
            k: (v[0].size == mesh.side_sets.get(k, (np.zeros(0),))[0].size)
            for k, v in back.side_sets.items()
        },
        "node_sets_match": {
            k: (v.size == mesh.node_sets.get(k, np.zeros(0)).size)
            for k, v in back.node_sets.items()
        },
        "max_coordinate_error": coord_err,
        "n_blocks": len(back.block_ids),
    }

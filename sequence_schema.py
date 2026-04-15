"""
Pickle format for polyscope sequence viewer (normative spec for producers).

Top-level object
----------------
``list`` of structure specification dicts (order = draw order).

Common fields (every dict)
--------------------------
structure_type : str
    ``STRUCTURE_SURFACE_MESH`` (``\"surface_mesh\"``),
    ``STRUCTURE_CURVE_NETWORK`` (``\"curve_network\"``), or
    ``STRUCTURE_DATA`` (``\"data\"``).
name : str
    Required for ``surface_mesh`` and ``curve_network``: stable identifier, unique among
    structures of that type. **Optional** for ``data`` (viewer assigns ``sequence_data_N`` if
    omitted); if provided, must be unique among all ``data`` entries.

vertices : numpy.ndarray, float, shape (T, V, 3)
    Required for ``surface_mesh`` and ``curve_network``. Time-varying 3D positions.
    **T must match every other structure.** All values must be **finite** (no ``nan``,
    no ``inf``).

surface_mesh extras
-------------------
faces : numpy.ndarray, integer, shape (F, 3)
    Triangle vertex indices for each frame (same topology for all t). Mesh geometry uses
    ``vertices`` only; those positions must be finite (no NaN or infinity).

curve_network extras
--------------------
edges : numpy.ndarray, integer, shape (E, 2)
    Segments between vertex indices 0 .. V-1.

data extras
-----------
values : numpy.ndarray, float, shape (T,) or (T, K) with K >= 1
    Per-frame numerical series (e.g. errors). **T** must match mesh/curve ``vertices``
    time axis, or match other ``data`` entries if the pickle has no geometry. All values
    must be **finite** (no ``nan``, no ``inf``). The viewer draws a minimal full-width
    sparkline (ImGui ``PlotLines``, same width as the timeline) above the transport bar;
    for ``(T, K)`` only the first column is shown.

Optional (mesh / curve)
-----------------------
color : numpy.ndarray, shape (3,)
    Constant RGB. Use floats in ``[0, 1]``, or integers in ``[0, 255]`` (values ``> 1`` are
    normalized by dividing by 255). Passed to Polyscope as ``color=`` when registering.

Validation
----------
The viewer checks: non-empty list, required keys per type, shared **T** across mesh, curve,
and data entries, finite ``vertices`` / ``values``, integer-like faces/edges with plausible
shapes, name rules per type, and optional ``color`` shape ``(3,)`` (integer or float) on
geometry structures.
"""

from __future__ import annotations

import numpy as np

STRUCTURE_SURFACE_MESH = "surface_mesh"
STRUCTURE_CURVE_NETWORK = "curve_network"
STRUCTURE_DATA = "data"

_STRUCTURE_TYPES = frozenset(
    {STRUCTURE_SURFACE_MESH, STRUCTURE_CURVE_NETWORK, STRUCTURE_DATA}
)


def _time_axis_length(spec: dict, st: str, index: int) -> int:
    if st in (STRUCTURE_SURFACE_MESH, STRUCTURE_CURVE_NETWORK):
        if "vertices" not in spec:
            raise ValueError(f"Item {index}: missing 'vertices'")
        v = spec["vertices"]
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Item {index}: 'vertices' must be numpy.ndarray")
        if v.ndim != 3 or v.shape[2] != 3:
            raise ValueError(
                f"Item {index}: 'vertices' must have shape (T, V, 3), got {v.shape}"
            )
        return int(v.shape[0])
    if st == STRUCTURE_DATA:
        if "values" not in spec:
            raise ValueError(f"Item {index}: data requires 'values'")
        val = spec["values"]
        if not isinstance(val, np.ndarray):
            raise TypeError(f"Item {index}: 'values' must be numpy.ndarray")
        if val.ndim == 1:
            return int(val.shape[0])
        if val.ndim == 2 and val.shape[1] >= 1:
            return int(val.shape[0])
        raise ValueError(
            f"Item {index}: 'values' must have shape (T,) or (T, K) with K>=1, got {val.shape}"
        )
    raise AssertionError("unreachable")


def validate_sequences(specs: list) -> int:
    """
    Validate a loaded sequence list. Returns the shared frame count T.

    Raises
    ------
    TypeError, ValueError
        If the structure is invalid (including non-finite ``vertices`` or ``values``).
    """
    if not isinstance(specs, list):
        raise TypeError(f"Expected a list of structure specs, got {type(specs).__name__}")
    if len(specs) == 0:
        raise ValueError("Sequence list is empty")

    lengths: list[int] = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise TypeError(f"Item {i} must be a dict, got {type(spec).__name__}")
        st = spec.get("structure_type")
        if st not in _STRUCTURE_TYPES:
            raise ValueError(
                f"Item {i}: structure_type must be one of {sorted(_STRUCTURE_TYPES)}, got {st!r}"
            )
        lengths.append(_time_axis_length(spec, st, i))

    t0 = lengths[0]
    for i, ln in enumerate(lengths):
        if ln != t0:
            raise ValueError(
                f"Item {i}: time length T={ln} differs from first item T={t0}"
            )

    names_mesh: set[str] = set()
    names_curve: set[str] = set()
    names_data: set[str] = set()

    for i, spec in enumerate(specs):
        st = spec.get("structure_type")
        assert st in _STRUCTURE_TYPES

        if st == STRUCTURE_DATA:
            name = spec.get("name")
            if name is not None and name != "":
                if not isinstance(name, str):
                    raise ValueError(f"Item {i}: 'name' must be a string when provided")
                if name in names_data:
                    raise ValueError(f"Duplicate data name: {name!r}")
                names_data.add(name)
        else:
            if "name" not in spec:
                raise ValueError(f"Item {i}: missing required key 'name'")
            name = spec["name"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"Item {i}: 'name' must be a non-empty string")
            if st == STRUCTURE_SURFACE_MESH:
                if name in names_mesh:
                    raise ValueError(f"Duplicate surface_mesh name: {name!r}")
                names_mesh.add(name)
            else:
                if name in names_curve:
                    raise ValueError(f"Duplicate curve_network name: {name!r}")
                names_curve.add(name)

        if st in (STRUCTURE_SURFACE_MESH, STRUCTURE_CURVE_NETWORK):
            v = spec["vertices"]
            if not np.issubdtype(v.dtype, np.floating):
                raise ValueError(f"Item {i} ({name}): 'vertices' must be floating dtype")
            if not np.isfinite(v).all():
                raise ValueError(
                    f"Item {i} ({name}): 'vertices' must be finite (no nan or inf)"
                )

            if st == STRUCTURE_SURFACE_MESH:
                if "faces" not in spec:
                    raise ValueError(f"Item {i} ({name}): surface_mesh requires 'faces'")
                faces = spec["faces"]
                if not isinstance(faces, np.ndarray) or faces.ndim != 2 or faces.shape[1] != 3:
                    raise ValueError(
                        f"Item {i} ({name}): 'faces' must be ndarray of shape (F, 3), got {getattr(faces, 'shape', None)}"
                    )
                if not np.issubdtype(faces.dtype, np.integer):
                    raise ValueError(f"Item {i} ({name}): 'faces' must have integer dtype")
                vmax = v.shape[1]
                if faces.size and (faces.min() < 0 or faces.max() >= vmax):
                    raise ValueError(
                        f"Item {i} ({name}): face indices out of range for V={vmax}"
                    )
            else:
                if "edges" not in spec:
                    raise ValueError(f"Item {i} ({name}): curve_network requires 'edges'")
                edges = spec["edges"]
                if not isinstance(edges, np.ndarray) or edges.ndim != 2 or edges.shape[1] != 2:
                    raise ValueError(
                        f"Item {i} ({name}): 'edges' must be ndarray of shape (E, 2), got {getattr(edges, 'shape', None)}"
                    )
                if not np.issubdtype(edges.dtype, np.integer):
                    raise ValueError(f"Item {i} ({name}): 'edges' must have integer dtype")
                vmax = v.shape[1]
                if edges.size and (edges.min() < 0 or edges.max() >= vmax):
                    raise ValueError(
                        f"Item {i} ({name}): edge indices out of range for V={vmax}"
                    )

            if "color" in spec and spec["color"] is not None:
                c = spec["color"]
                if not isinstance(c, np.ndarray) or c.shape != (3,):
                    raise ValueError(
                        f"Item {i} ({name}): 'color' must be ndarray of shape (3,), got {getattr(c, 'shape', None)}"
                    )
                if not (
                    np.issubdtype(c.dtype, np.floating) or np.issubdtype(c.dtype, np.integer)
                ):
                    raise ValueError(
                        f"Item {i} ({name}): 'color' must be integer or floating dtype"
                    )

        else:
            val = spec["values"]
            if not np.issubdtype(val.dtype, np.floating):
                raise ValueError(f"Item {i}: 'values' must be floating dtype")
            if not np.isfinite(val).all():
                raise ValueError(f"Item {i}: 'values' must be finite (no nan or inf)")

    return t0

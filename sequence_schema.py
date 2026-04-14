"""
Pickle format for polyscope sequence viewer (normative spec for producers).

Top-level object
----------------
``list`` of structure specification dicts (order = draw order).

Common fields (every dict)
--------------------------
structure_type : str
    ``STRUCTURE_SURFACE_MESH`` (``\"surface_mesh\"``) or
    ``STRUCTURE_CURVE_NETWORK`` (``\"curve_network\"``).
name : str
    Stable identifier, unique among structures of the same ``structure_type``.
vertices : numpy.ndarray, float, shape (T, V, 3)
    Time-varying 3D positions. **T must match every other structure.** All values must be
    **finite** (no ``nan``, no ``inf``); non-finite coordinates are invalid for the viewer.

surface_mesh extras
-------------------
faces : numpy.ndarray, integer, shape (F, 3)
    Triangle vertex indices for each frame (same topology for all t). Mesh geometry uses
    ``vertices`` only; those positions must be finite (no NaN or infinity).

curve_network extras
--------------------
edges : numpy.ndarray, integer, shape (E, 2)
    Segments between vertex indices 0 .. V-1.

Optional (both types)
---------------------
color : numpy.ndarray, shape (3,)
    Constant RGB. Use floats in ``[0, 1]``, or integers in ``[0, 255]`` (values ``> 1`` are
    normalized by dividing by 255). Passed to Polyscope as ``color=`` when registering.

Validation
----------
The viewer checks: non-empty list, required keys per type, ``vertices.ndim == 3``,
that **all vertex coordinates are finite** (no NaN or infinity), shared T, integer-like
faces/edges with plausible shapes, name uniqueness per type, and optional ``color`` shape
``(3,)`` (integer or float).
"""

from __future__ import annotations

import numpy as np

STRUCTURE_SURFACE_MESH = "surface_mesh"
STRUCTURE_CURVE_NETWORK = "curve_network"

_STRUCTURE_TYPES = frozenset({STRUCTURE_SURFACE_MESH, STRUCTURE_CURVE_NETWORK})


def validate_sequences(specs: list) -> int:
    """
    Validate a loaded sequence list. Returns the shared frame count T.

    Raises
    ------
    TypeError, ValueError
        If the structure is invalid (including non-finite ``vertices``).
    """
    if not isinstance(specs, list):
        raise TypeError(f"Expected a list of structure specs, got {type(specs).__name__}")
    if len(specs) == 0:
        raise ValueError("Sequence list is empty")

    names_mesh: set[str] = set()
    names_curve: set[str] = set()
    t0: int | None = None

    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise TypeError(f"Item {i} must be a dict, got {type(spec).__name__}")

        st = spec.get("structure_type")
        if st not in _STRUCTURE_TYPES:
            raise ValueError(
                f"Item {i}: structure_type must be one of {sorted(_STRUCTURE_TYPES)}, got {st!r}"
            )

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

        if "vertices" not in spec:
            raise ValueError(f"Item {i} ({name}): missing 'vertices'")
        v = spec["vertices"]
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Item {i} ({name}): 'vertices' must be numpy.ndarray")
        if v.ndim != 3 or v.shape[2] != 3:
            raise ValueError(
                f"Item {i} ({name}): 'vertices' must have shape (T, V, 3), got {v.shape}"
            )
        if not np.issubdtype(v.dtype, np.floating):
            raise ValueError(f"Item {i} ({name}): 'vertices' must be floating dtype")
        if not np.isfinite(v).all():
            raise ValueError(
                f"Item {i} ({name}): 'vertices' must be finite (no nan or inf)"
            )

        t, _v, _ = v.shape
        if t0 is None:
            t0 = t
        elif t != t0:
            raise ValueError(
                f"Item {i} ({name}): T={t} differs from first structure T={t0}"
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

    assert t0 is not None
    return t0

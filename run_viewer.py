"""
Load a sequence pickle and play it in Polyscope using a frame_tick loop.

Usage:
    python run_viewer.py path/to/sequences.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import polyscope as ps
import polyscope.imgui as imgui

from sequence_schema import (
    STRUCTURE_CURVE_NETWORK,
    STRUCTURE_SURFACE_MESH,
    validate_sequences,
)

# At playback_speed=1.0, advance ~this many sequence frames per wall second.
REFERENCE_FPS = 30.0


@dataclass
class ViewerState:
    pickle_path: str = ""
    frame_idx: int = 0
    playing: bool = False
    playback_speed: float = 1.0
    frame_accum: float = 0.0
    t_frames: int = 0
    specs: list = field(default_factory=list)
    mesh_registered: set = field(default_factory=set)
    curve_registered: set = field(default_factory=set)
    last_t: float | None = None
    reload_requested: bool = False
    # Per-structure: subtract vertex mean (centroid) each frame, then add i * x_offset on x.
    lock_center: bool = False
    x_offset: float = 0.0


def _color_kw(spec: dict) -> dict:
    if "color" not in spec or spec["color"] is None:
        return {}
    c = np.asarray(spec["color"], dtype=np.float64).reshape(3)
    if np.issubdtype(spec["color"].dtype, np.integer) and c.max() > 1.0:
        c = c / 255.0
    return {"color": (float(c[0]), float(c[1]), float(c[2]))}


def _transform_vertices(state: ViewerState, verts: np.ndarray, structure_index: int) -> np.ndarray:
    """Apply center-lock and/or per-structure x spacing (i × offset); copy only if needed."""
    dx = float(structure_index) * state.x_offset
    if not state.lock_center and dx == 0.0:
        return np.ascontiguousarray(verts, dtype=np.float64)
    out = np.ascontiguousarray(verts, dtype=np.float64).copy()
    if state.lock_center:
        out -= out.mean(axis=0)
    if dx != 0.0:
        out[:, 0] += dx
    return out


def update_structures_for_frame(state: ViewerState) -> None:
    t = state.frame_idx
    for i, spec in enumerate(state.specs):
        st = spec["structure_type"]
        name = spec["name"]
        verts = _transform_vertices(state, spec["vertices"][t], i)
        ck = _color_kw(spec)

        if st == STRUCTURE_SURFACE_MESH:
            faces = spec["faces"]
            if name in state.mesh_registered:
                ps.get_surface_mesh(name).update_vertex_positions(verts)
            else:
                ps.register_surface_mesh(name, verts, faces, smooth_shade=True, **ck)
                state.mesh_registered.add(name)
        elif st == STRUCTURE_CURVE_NETWORK:
            edges = spec["edges"]
            if name in state.curve_registered:
                ps.get_curve_network(name).update_node_positions(verts)
            else:
                ps.register_curve_network(name, verts, edges, **ck)
                state.curve_registered.add(name)


def advance_playhead(state: ViewerState, dt: float) -> None:
    if not state.playing or state.t_frames <= 0:
        return
    state.frame_accum += dt * state.playback_speed * REFERENCE_FPS
    while state.frame_accum >= 1.0:
        state.frame_accum -= 1.0
        state.frame_idx = (state.frame_idx + 1) % state.t_frames


def reload_pickle(state: ViewerState) -> None:
    with open(state.pickle_path, "rb") as f:
        specs = pickle.load(f)
    t_frames = validate_sequences(specs)
    ps.remove_all_structures()
    state.mesh_registered.clear()
    state.curve_registered.clear()
    state.specs = specs
    state.t_frames = t_frames
    tmax = max(0, t_frames - 1)
    state.frame_idx = min(state.frame_idx, tmax)
    state.frame_accum = 0.0


# Bottom bar height (fixed; avoids ImVec2 from GetIO().DisplaySize in bindings).
_PLAYBACK_PANEL_HEIGHT = 200.0


def build_user_callback(state: ViewerState):
    def callback():
        win_w, win_h = ps.get_window_size()
        h = min(_PLAYBACK_PANEL_HEIGHT, float(win_h))
        imgui.SetNextWindowPos(
            (0.0, float(win_h) - h),
            imgui.ImGuiCond_Always,
        )
        imgui.SetNextWindowSize(
            (float(win_w), h),
            imgui.ImGuiCond_Always,
        )
        flags = (
            imgui.ImGuiWindowFlags_NoTitleBar
            | imgui.ImGuiWindowFlags_NoResize
            | imgui.ImGuiWindowFlags_NoMove
            | imgui.ImGuiWindowFlags_NoCollapse
            | imgui.ImGuiWindowFlags_NoScrollbar
        )
        imgui.Begin("Sequence playback", True, flags)

        tmax = max(0, state.t_frames - 1)
        imgui.Text(f"Frame {state.frame_idx + 1} / {state.t_frames}")
        imgui.SameLine()
        # Single toggle: shows the action clicking will perform.
        if imgui.Button("Pause" if state.playing else "Play"):
            state.playing = not state.playing
        imgui.SameLine()
        if imgui.Button("Reload"):
            state.reload_requested = True

        imgui.Text("Timeline")
        # Full-width slider (PushItemWidth -1 = use all remaining width on this line).
        imgui.PushItemWidth(-1.0)
        changed, new_frame = imgui.SliderInt("##timeline", state.frame_idx, 0, tmax)
        imgui.PopItemWidth()
        if changed:
            state.frame_idx = new_frame
            state.frame_accum = 0.0

        # Narrower speed control (label + short slider).
        imgui.Text("Speed")
        imgui.SameLine()
        imgui.PushItemWidth(160.0)
        # 1.0 = REFERENCE_FPS sequence frames per second; lower = slower.
        _, state.playback_speed = imgui.SliderFloat(
            "##speed",
            state.playback_speed,
            0.05,
            1.0,
            format="%.2f",
        )
        imgui.PopItemWidth()

        _, state.lock_center = imgui.Checkbox("Center each structure at origin", state.lock_center)
        imgui.Text("X spacing (structure i gets i × value)")
        imgui.PushItemWidth(220.0)
        _, state.x_offset = imgui.SliderFloat(
            "##x_offset",
            state.x_offset,
            -2.0,
            2.0,
            format="%.3f",
        )
        imgui.PopItemWidth()

        imgui.End()

    return callback


def run_loop(state: ViewerState) -> None:
    ps.set_user_callback(build_user_callback(state))

    while not ps.window_requests_close():
        if state.reload_requested:
            reload_pickle(state)
            state.reload_requested = False

        now = time.perf_counter()
        if state.last_t is None:
            dt = 0.0
        else:
            dt = now - state.last_t
        state.last_t = now

        advance_playhead(state, dt)
        update_structures_for_frame(state)
        ps.frame_tick()


def main() -> None:
    p = argparse.ArgumentParser(description="View Polyscope sequence pickles.")
    p.add_argument(
        "pickle_path",
        help="path to the sequence pickle file",
    )
    args = p.parse_args()

    try:
        with open(args.pickle_path, "rb") as f:
            specs = pickle.load(f)
    except ModuleNotFoundError as e:
        err = str(e).lower()
        if "numpy" in err or "_core" in err:
            raise SystemExit(
                "Pickle failed to load: NumPy version mismatch.\n"
                "Files saved with NumPy 2.x need NumPy 2.0+ to unpickle "
                "(e.g. pip install 'numpy>=2' in this env).\n"
                "Alternatively, re-save the pickle using the same NumPy as the viewer.\n"
                f"Original error: {e}"
            ) from e
        raise

    t_frames = validate_sequences(specs)

    ps.set_max_fps(60)
    # Newer Polyscope builds expose this; older bindings only honor set_max_fps.
    if hasattr(ps, "set_frame_tick_limit_fps_mode"):
        ps.set_frame_tick_limit_fps_mode("block_to_hit_target")
    ps.set_ground_plane_mode("none")
    ps.init()

    state = ViewerState(
        pickle_path=os.path.abspath(args.pickle_path),
        t_frames=t_frames,
        specs=specs,
    )
    try:
        run_loop(state)
    finally:
        ps.unshow()


if __name__ == "__main__":
    main()


from __future__ import annotations

import copy

import numpy as np

from ..dynamics.maneuvers import phasing_maneuver, plane_change, raise_orbit



_DEFAULTS = {
    "altitude_step": 150.0,
    "phase_step": 0.5,
    "inclination_step": 0.1,
    "lookahead_steps": 200,
    "lookahead_stride": 5,
}
_DEFAULT_BAND = (150.0, 1200.0)   


def _mparams(context) -> dict:
    
    cfg = getattr(context, "config", None) or {}
    m = dict(_DEFAULTS)
    m.update(cfg.get("maneuvers", {}) or {})
    return m


def _altitude_band(context) -> tuple:
    
    cfg = getattr(context, "config", None) or {}
    swarm = cfg.get("swarm", {}) or {}
    return (
        float(swarm.get("min_altitude", _DEFAULT_BAND[0])),
        float(swarm.get("max_altitude", _DEFAULT_BAND[1])),
    )


def enumerate_candidate_maneuvers(orbit, context):
    
    p = _mparams(context)
    lo, hi = _altitude_band(context)
    R = context.asteroid.R
    alt = orbit.semi_major_axis - R

    cands = [("stay", orbit, 0.0)]

    astep = float(p["altitude_step"])
    if alt + astep <= hi:
        cands.append(("raise", *raise_orbit(orbit, astep)))
    if alt - astep >= lo:
        cands.append(("lower", *raise_orbit(orbit, -astep)))

    pstep = float(p["phase_step"])
    cands.append(("phase+", *phasing_maneuver(orbit, pstep)))
    cands.append(("phase-", *phasing_maneuver(orbit, -pstep)))

    istep = float(p["inclination_step"])
    if orbit.inclination + istep < np.pi:
        cands.append(("incl+", *plane_change(orbit, istep)))
    if orbit.inclination - istep > 0.0:
        cands.append(("incl-", *plane_change(orbit, -istep)))

    return cands


def estimate_new_coverage(orbit, context, horizon_steps=None, stride=None) -> float:
    
    p = _mparams(context)
    horizon = int(p["lookahead_steps"] if horizon_steps is None else horizon_steps)
    stride = max(1, int(p["lookahead_stride"] if stride is None else stride))

    sensor = context.sensor
    surface_map = context.surface_map
    dt = context.dt

    orb = copy.copy(orbit)                 
    ast = copy.copy(context.asteroid)      
    predicted = surface_map.coverage.copy()

    gain = 0.0
    n_samples = max(1, horizon // stride)
    for _ in range(n_samples):
        pos = orb.position
        cells = sensor.footprint_cells(pos, ast)
        if cells.size:
            q = sensor.quality_at(pos, ast)
            if q > 0.0:
                cap = 1.0 - predicted[cells]
                add = np.clip(np.minimum(q, cap), 0.0, None)
                gain += float(add.sum())
                predicted[cells] += add
        for _s in range(stride):
            orb.propagate(dt)
            ast.update(dt)

    return gain


from __future__ import annotations

import numpy as np

from .two_body import KeplerOrbit

_TWO_PI = 2.0 * np.pi


def _carry(orbit, new_orbit, dv):
    
    new_orbit.delta_v = orbit.delta_v + dv
    return new_orbit, dv


def apply_impulse(orbit, dv_vector):
    
    dv_vector = np.asarray(dv_vector, dtype=float)
    r = orbit.position
    v = orbit.velocity + dv_vector
    new = KeplerOrbit.from_state(orbit.mu, r, v)
    return _carry(orbit, new, float(np.linalg.norm(dv_vector)))


def hohmann_transfer(mu, r1, r2):
    
    mu, r1, r2 = float(mu), float(r1), float(r2)
    a_t = 0.5 * (r1 + r2)                       
    v1 = np.sqrt(mu / r1)                       
    v2 = np.sqrt(mu / r2)                       
    v_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))   
    v_apo = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))    
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    return {
        "dv1": dv1,
        "dv2": dv2,
        "dv_total": dv1 + dv2,
        "t_transfer": np.pi * np.sqrt(a_t ** 3 / mu),
    }


def raise_orbit(orbit, delta_altitude):
    
    r1 = orbit.semi_major_axis
    r2 = r1 + float(delta_altitude)
    if r2 <= 0.0:
        raise ValueError("raise_orbit: resulting radius must be positive.")
    h = hohmann_transfer(orbit.mu, r1, r2)
    arrival = (orbit.argp + orbit.nu + np.pi) % _TWO_PI
    new = KeplerOrbit.from_elements(
        mu=orbit.mu, a=r2, e=0.0,
        inclination=orbit.inclination, raan=orbit.raan, argp=0.0, nu=arrival,
    )
    return _carry(orbit, new, h["dv_total"])


def phasing_maneuver(orbit, delta_true_anomaly, n_rev: int = 1):
    
    mu = orbit.mu
    r = orbit.semi_major_axis
    v_circ = np.sqrt(mu / r)
    T = orbit.period
    
    t_phase = T * (1.0 - delta_true_anomaly / (_TWO_PI * n_rev))
    if t_phase <= 0.0:
        raise ValueError("phasing_maneuver: phase shift too large for n_rev.")
    a_phase = (mu * (t_phase / _TWO_PI) ** 2) ** (1.0 / 3.0)
    v_phase = np.sqrt(mu * (2.0 / r - 1.0 / a_phase))
    dv = 2.0 * abs(v_phase - v_circ)
    new = KeplerOrbit.from_elements(
        mu=mu, a=r, e=0.0,
        inclination=orbit.inclination, raan=orbit.raan, argp=0.0,
        nu=(orbit.argp + orbit.nu + delta_true_anomaly) % _TWO_PI,
    )
    return _carry(orbit, new, dv)


def plane_change(orbit, delta_inclination):
    
    di = float(delta_inclination)
    v = float(np.linalg.norm(orbit.velocity))
    dv = 2.0 * v * abs(np.sin(0.5 * di))
    new = KeplerOrbit.from_elements(
        mu=orbit.mu, a=orbit.semi_major_axis, e=orbit.eccentricity,
        inclination=orbit.inclination + di, raan=orbit.raan,
        argp=orbit.argp, nu=orbit.nu,
    )
    return _carry(orbit, new, dv)

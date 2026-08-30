
from __future__ import annotations

import numpy as np


_EPS_E = 1e-11   
_EPS_I = 1e-11   
_TWO_PI = 2.0 * np.pi


class CircularOrbit:
    

    def __init__(self, mu, radius, altitude, inclination=0.0, raan=0.0, phase0=0.0):
        self.a = float(radius) + float(altitude)      
        self.mu = float(mu)
        self.n = np.sqrt(self.mu / self.a ** 3)        
        self.inclination = float(inclination)
        self.raan = float(raan)
        self.theta = float(phase0)                     
        self.delta_v = 0.0

    def _rotation(self) -> np.ndarray:
        
        cO, sO = np.cos(self.raan), np.sin(self.raan)
        ci, si = np.cos(self.inclination), np.sin(self.inclination)
        r_raan = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
        r_inc = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
        return r_raan @ r_inc

    @property
    def position(self) -> np.ndarray:
        
        in_plane = self.a * np.array([np.cos(self.theta), np.sin(self.theta), 0.0])
        return self._rotation() @ in_plane

    def propagate(self, dt: float) -> None:
        
        self.theta = (self.theta + self.n * dt) % _TWO_PI





def solve_kepler_elliptic(M, e, tol: float = 1e-12, max_iter: int = 100) -> float:
    
    M = (M + np.pi) % _TWO_PI - np.pi          
    e = float(e)
    E = M if e < 0.8 else np.pi * np.sign(M or 1.0)
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def true_to_mean(nu, e) -> float:
    
    e = float(e)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                         np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    return E - e * np.sin(E)


def mean_to_true(M, e) -> float:
    
    e = float(e)
    E = solve_kepler_elliptic(M, e)
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                            np.sqrt(1.0 - e) * np.cos(E / 2.0))





def elements_to_state(mu, a, e=0.0, inclination=0.0, raan=0.0, argp=0.0, nu=0.0):
    
    mu = float(mu)
    p = a * (1.0 - e * e)                       
    r = p / (1.0 + e * np.cos(nu))

    
    r_pqw = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inclination), np.sin(inclination)
    cw, sw = np.cos(argp), np.sin(argp)
    r3_raan = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    r1_inc = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    r3_argp = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])
    q = r3_raan @ r1_inc @ r3_argp

    return q @ r_pqw, q @ v_pqw


def state_to_elements(mu, r, v):
    
    mu = float(mu)
    r_vec = np.asarray(r, dtype=float)
    v_vec = np.asarray(v, dtype=float)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)   
    n = np.linalg.norm(n_vec)

    e_vec = ((v * v - mu / r) * r_vec - r * vr * v_vec) / mu
    e = np.linalg.norm(e_vec)

    energy = 0.5 * v * v - mu / r
    a = np.inf if abs(energy) < 1e-15 else -mu / (2.0 * energy)

    inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    equatorial = n < _EPS_I
    circular = e < _EPS_E

    if not equatorial:
        raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0.0:
            raan = _TWO_PI - raan
    else:
        raan = 0.0

    if circular:
        
        argp = 0.0
        if not equatorial:                     
            nu = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0))
            if r_vec[2] < 0.0:
                nu = _TWO_PI - nu
        else:                                  
            nu = np.arccos(np.clip(r_vec[0] / r, -1.0, 1.0))
            if r_vec[1] < 0.0:
                nu = _TWO_PI - nu
    else:
        if not equatorial:
            argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
            if e_vec[2] < 0.0:
                argp = _TWO_PI - argp
        else:                                  
            argp = np.arccos(np.clip(e_vec[0] / e, -1.0, 1.0))
            if e_vec[1] < 0.0:
                argp = _TWO_PI - argp
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if vr < 0.0:
            nu = _TWO_PI - nu

    return a, e, inclination, raan, argp, nu % _TWO_PI


def propagate_elements(elements, mu, dt):
    
    a, e, inclination, raan, argp, nu = elements
    n = np.sqrt(mu / a ** 3)
    M = true_to_mean(nu, e) + n * dt
    nu_new = mean_to_true(M, e) % _TWO_PI
    return a, e, inclination, raan, argp, nu_new





def stumpff_c(z: float) -> float:
    
    if z > 1e-9:
        sz = np.sqrt(z)
        return (1.0 - np.cos(sz)) / z
    if z < -1e-9:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1.0) / (-z)
    return 0.5 - z / 24.0                        


def stumpff_s(z: float) -> float:
    
    if z > 1e-9:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz ** 3
    if z < -1e-9:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz ** 3
    return 1.0 / 6.0 - z / 120.0                 


def propagate_kepler(state, mu, dt, tol: float = 1e-10, max_iter: int = 100):
    
    r0_vec = np.asarray(state[0], dtype=float)
    v0_vec = np.asarray(state[1], dtype=float)
    mu = float(mu)
    dt = float(dt)

    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    if r0 == 0.0:
        raise ValueError("propagate_kepler: zero position vector.")

    sqrt_mu = np.sqrt(mu)
    vr0 = np.dot(r0_vec, v0_vec) / r0
    alpha = 2.0 / r0 - v0 * v0 / mu             

    chi = sqrt_mu * abs(alpha) * dt             
    for _ in range(max_iter):
        z = alpha * chi * chi
        c = stumpff_c(z)
        s = stumpff_s(z)
        f = (
            r0 * vr0 / sqrt_mu * chi * chi * c
            + (1.0 - alpha * r0) * chi ** 3 * s
            + r0 * chi
            - sqrt_mu * dt
        )
        df = (
            r0 * vr0 / sqrt_mu * chi * (1.0 - alpha * chi * chi * s)
            + (1.0 - alpha * r0) * chi * chi * c
            + r0
        )
        ratio = f / df
        chi -= ratio
        if abs(ratio) < tol:
            break

    z = alpha * chi * chi
    c = stumpff_c(z)
    s = stumpff_s(z)

    ff = 1.0 - chi * chi / r0 * c
    gg = dt - chi ** 3 / sqrt_mu * s
    r_vec = ff * r0_vec + gg * v0_vec
    r = np.linalg.norm(r_vec)

    fdot = sqrt_mu / (r * r0) * (alpha * chi ** 3 * s - chi)
    gdot = 1.0 - chi * chi / r * c
    v_vec = fdot * r0_vec + gdot * v0_vec

    return r_vec, v_vec





class KeplerOrbit:
    

    def __init__(self, mu, a, e=0.0, inclination=0.0, raan=0.0, argp=0.0, nu=0.0):
        self.mu = float(mu)
        self.a = float(a)
        self.e = float(e)
        self.inclination = float(inclination)
        self.raan = float(raan)
        self.argp = float(argp)
        self.nu = float(nu)
        self.delta_v = 0.0

    
    @classmethod
    def from_elements(cls, mu, a, e=0.0, inclination=0.0, raan=0.0, argp=0.0, nu=0.0):
        
        return cls(mu, a, e, inclination, raan, argp, nu)

    @classmethod
    def from_state(cls, mu, r, v):
        
        a, e, i, raan, argp, nu = state_to_elements(mu, r, v)
        return cls(mu, a, e, i, raan, argp, nu)

    
    @property
    def elements(self):
        
        return (self.a, self.e, self.inclination, self.raan, self.argp, self.nu)

    @property
    def n(self) -> float:
        
        return np.sqrt(self.mu / abs(self.a) ** 3)

    @property
    def semi_major_axis(self) -> float:
        return self.a

    @property
    def eccentricity(self) -> float:
        return self.e

    @property
    def period(self) -> float:
        
        if self.e >= 1.0 or self.a <= 0.0:
            raise ValueError("period is defined only for bound (elliptic) orbits.")
        return _TWO_PI * np.sqrt(self.a ** 3 / self.mu)

    @property
    def specific_energy(self) -> float:
        
        return -self.mu / (2.0 * self.a)

    def _state(self):
        return elements_to_state(self.mu, self.a, self.e, self.inclination,
                                 self.raan, self.argp, self.nu)

    @property
    def position(self) -> np.ndarray:
        
        return self._state()[0]

    @property
    def velocity(self) -> np.ndarray:
        
        return self._state()[1]

    
    @property
    def r(self) -> np.ndarray:
        return self.position

    @property
    def v(self) -> np.ndarray:
        return self.velocity

    
    def propagate(self, dt: float) -> None:
        
        if self.e < 1.0:
            self.a, self.e, self.inclination, self.raan, self.argp, self.nu = \
                propagate_elements(self.elements, self.mu, dt)
        else:
            r, v = propagate_kepler(self._state(), self.mu, dt)
            self.a, self.e, self.inclination, self.raan, self.argp, self.nu = \
                state_to_elements(self.mu, r, v)


import numpy as np
import pytest

from astroswarm.dynamics import (
    CircularOrbit,
    KeplerOrbit,
    elements_to_state,
    state_to_elements,
    propagate_elements,
    propagate_kepler,
    solve_kepler_elliptic,
    mean_to_true,
    true_to_mean,
    stumpff_c,
    stumpff_s,
)


MU = 3.5e-5


def make_eccentric():
    
    return KeplerOrbit.from_elements(
        mu=MU, a=1000.0, e=0.3, inclination=1.2, raan=0.4, argp=0.7, nu=0.9
    )



@pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.7, 0.9])
@pytest.mark.parametrize("M", np.linspace(-3.0, 3.0, 7))
def test_kepler_solver_satisfies_equation(M, e):
    E = solve_kepler_elliptic(M, e)
    lhs = E - e * np.sin(E)
    
    diff = (lhs - M + np.pi) % (2 * np.pi) - np.pi
    assert diff == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("e", [0.0, 0.2, 0.6, 0.85])
def test_mean_true_are_inverses(e):
    for nu in np.linspace(-3.0, 3.0, 9):
        M = true_to_mean(nu, e)
        nu_back = mean_to_true(M, e)
        d = (nu_back - nu + np.pi) % (2 * np.pi) - np.pi
        assert d == pytest.approx(0.0, abs=1e-10)



def test_stumpff_values_at_zero():
    assert stumpff_c(0.0) == pytest.approx(0.5)
    assert stumpff_s(0.0) == pytest.approx(1.0 / 6.0)


def test_stumpff_continuous_across_zero():
    for z in (-1e-6, -1e-10, 1e-10, 1e-6):
        assert stumpff_c(z) == pytest.approx(0.5, abs=1e-6)
        assert stumpff_s(z) == pytest.approx(1.0 / 6.0, abs=1e-6)



def _assert_state_roundtrip(mu, a, e, i, raan, argp, nu):
    r, v = elements_to_state(mu, a, e, i, raan, argp, nu)
    a2, e2, i2, raan2, argp2, nu2 = state_to_elements(mu, r, v)
    r2, v2 = elements_to_state(mu, a2, e2, i2, raan2, argp2, nu2)
    
    assert np.allclose(r2, r, atol=1e-6 * np.linalg.norm(r))
    assert np.allclose(v2, v, atol=1e-6 * np.linalg.norm(v))


def test_roundtrip_general_orbit():
    _assert_state_roundtrip(MU, 1000.0, 0.3, 1.2, 0.4, 0.7, 0.9)


def test_roundtrip_circular_inclined():
    _assert_state_roundtrip(MU, 900.0, 0.0, 1.0, 0.4, 0.0, 0.9)


def test_roundtrip_circular_equatorial():
    _assert_state_roundtrip(MU, 900.0, 0.0, 0.0, 0.0, 0.0, 1.3)


def test_roundtrip_elliptic_equatorial():
    _assert_state_roundtrip(MU, 1100.0, 0.25, 0.0, 0.0, 0.6, 1.1)


def test_state_to_elements_recovers_a_e():
    r, v = elements_to_state(MU, 1234.0, 0.42, 0.7, 0.3, 0.5, 2.0)
    a, e, i, _, _, _ = state_to_elements(MU, r, v)
    assert a == pytest.approx(1234.0, rel=1e-9)
    assert e == pytest.approx(0.42, rel=1e-9)
    assert i == pytest.approx(0.7, rel=1e-9)



def test_element_and_universal_propagation_agree():
    orb = make_eccentric()
    r0, v0 = orb.r.copy(), orb.v.copy()
    for dt in (0.0, 1234.0, 1.0e6, 7.3e6, -5.0e5):
        
        els = propagate_elements(orb.elements, MU, dt)
        r_el, v_el = elements_to_state(MU, *els)
        
        r_uv, v_uv = propagate_kepler((r0, v0), MU, dt)
        assert np.allclose(r_el, r_uv, atol=1e-6 * np.linalg.norm(r_uv))
        assert np.allclose(v_el, v_uv, atol=1e-6 * np.linalg.norm(v_uv))



def test_full_period_returns_to_start():
    orb = make_eccentric()
    r0, v0 = orb.r.copy(), orb.v.copy()
    orb.propagate(orb.period)
    assert np.allclose(orb.r, r0, atol=1e-6 * np.linalg.norm(r0))
    assert np.allclose(orb.v, v0, atol=1e-6 * np.linalg.norm(v0))


def test_energy_and_angular_momentum_conserved_along_orbit():
    orb = make_eccentric()
    T = orb.period
    e0 = orb.specific_energy
    h0 = np.cross(orb.r, orb.v)
    for frac in np.linspace(0.0, 1.0, 13):
        els = propagate_elements(orb.elements, MU, frac * T)
        r, v = elements_to_state(MU, *els)
        ek = 0.5 * np.dot(v, v) - MU / np.linalg.norm(r)
        assert ek == pytest.approx(e0, rel=1e-9)
        assert np.allclose(np.cross(r, v), h0, rtol=1e-9, atol=0)


def test_half_period_reaches_apoapsis():
    a, e = 1000.0, 0.3
    orb = KeplerOrbit.from_elements(mu=MU, a=a, e=e, nu=0.0)   
    assert np.linalg.norm(orb.r) == pytest.approx(a * (1 - e))
    orb.propagate(orb.period / 2.0)
    assert np.linalg.norm(orb.r) == pytest.approx(a * (1 + e), rel=1e-8)



def test_reversibility():
    orb = make_eccentric()
    r0, v0 = orb.r.copy(), orb.v.copy()
    orb.propagate(4321.0)
    orb.propagate(-4321.0)
    assert np.allclose(orb.r, r0, atol=1e-6 * np.linalg.norm(r0))
    assert np.allclose(orb.v, v0, atol=1e-6 * np.linalg.norm(v0))


def test_from_state_recovers_orbit():
    
    orb = make_eccentric()
    r, v = orb.r.copy(), orb.v.copy()
    rebuilt = KeplerOrbit.from_state(MU, r, v)
    assert np.allclose(rebuilt.position, r, atol=1e-6 * np.linalg.norm(r))
    assert np.allclose(rebuilt.velocity, v, atol=1e-6 * np.linalg.norm(v))



def test_circular_orbit_is_special_case_of_kepler():
    R, alt, inc, raan, phase0 = 500.0, 400.0, 1.2, 0.4, 0.9
    a = R + alt
    circ = CircularOrbit(mu=MU, radius=R, altitude=alt,
                         inclination=inc, raan=raan, phase0=phase0)
    kep = KeplerOrbit.from_elements(mu=MU, a=a, e=0.0,
                                    inclination=inc, raan=raan, argp=0.0, nu=phase0)
    assert np.allclose(kep.position, circ.position, atol=1e-6 * a)
    for _ in range(5):
        circ.propagate(1000.0)
        kep.propagate(1000.0)
        assert np.allclose(kep.position, circ.position, atol=1e-4 * a)


def test_kepler_orbit_reports_consistent_elements():
    a = 1234.0
    orb = KeplerOrbit.from_elements(mu=MU, a=a, e=0.25, inclination=0.5, nu=1.3)
    assert orb.semi_major_axis == pytest.approx(a, rel=1e-9)
    assert orb.eccentricity == pytest.approx(0.25, rel=1e-12)
    assert orb.specific_energy == pytest.approx(-MU / (2.0 * a), rel=1e-9)





from astroswarm.dynamics import (
    apply_impulse,
    hohmann_transfer,
    raise_orbit,
    phasing_maneuver,
    plane_change,
)


def make_circular(a=900.0, inclination=1.2, raan=0.4, nu=0.9):
    return KeplerOrbit.from_elements(mu=MU, a=a, e=0.0,
                                     inclination=inclination, raan=raan, nu=nu)



def test_hohmann_matches_textbook_formula():
    r1, r2 = 900.0, 1400.0
    h = hohmann_transfer(MU, r1, r2)
    
    a_t = 0.5 * (r1 + r2)
    v1, v2 = np.sqrt(MU / r1), np.sqrt(MU / r2)
    vp = np.sqrt(MU * (2 / r1 - 1 / a_t))
    va = np.sqrt(MU * (2 / r2 - 1 / a_t))
    assert h["dv1"] == pytest.approx(vp - v1, rel=1e-12)
    assert h["dv2"] == pytest.approx(v2 - va, rel=1e-12)
    assert h["dv_total"] == pytest.approx((vp - v1) + (v2 - va), rel=1e-12)
    assert h["t_transfer"] == pytest.approx(np.pi * np.sqrt(a_t ** 3 / MU), rel=1e-12)


def test_raise_orbit_changes_altitude_and_costs_textbook_dv():
    orb = make_circular(a=900.0)
    new, dv = raise_orbit(orb, 500.0)          
    assert new.semi_major_axis == pytest.approx(1400.0, rel=1e-12)
    assert new.eccentricity == pytest.approx(0.0, abs=1e-9)
    assert dv == pytest.approx(hohmann_transfer(MU, 900.0, 1400.0)["dv_total"], rel=1e-12)
    assert dv > 0.0


def test_lower_orbit_also_works():
    orb = make_circular(a=1400.0)
    new, dv = raise_orbit(orb, -500.0)         
    assert new.semi_major_axis == pytest.approx(900.0, rel=1e-12)
    assert dv == pytest.approx(hohmann_transfer(MU, 1400.0, 900.0)["dv_total"], rel=1e-12)
    assert dv > 0.0



def test_apply_impulse_preserves_position_and_adds_velocity():
    orb = make_eccentric()
    r0, v0 = orb.r.copy(), orb.v.copy()
    dv_vec = np.array([1e-5, -2e-5, 0.5e-5])
    new, dv = apply_impulse(orb, dv_vec)
    assert np.allclose(new.position, r0, atol=1e-6 * np.linalg.norm(r0))   
    assert np.allclose(new.velocity, v0 + dv_vec, atol=1e-9)
    assert dv == pytest.approx(np.linalg.norm(dv_vec), rel=1e-12)


def test_zero_impulse_is_noop():
    orb = make_circular()
    new, dv = apply_impulse(orb, [0.0, 0.0, 0.0])
    assert dv == pytest.approx(0.0)
    assert np.allclose(new.position, orb.position, atol=1e-9)



def test_plane_change_dv_and_inclination():
    orb = make_circular(a=1000.0, inclination=0.5)
    di = 0.2
    new, dv = plane_change(orb, di)
    v = np.linalg.norm(orb.velocity)
    assert new.inclination == pytest.approx(0.5 + di, rel=1e-9)
    assert new.semi_major_axis == pytest.approx(1000.0, rel=1e-9)
    assert dv == pytest.approx(2 * v * np.sin(di / 2), rel=1e-9)



def test_phasing_zero_shift_is_free():
    orb = make_circular(a=1000.0)
    new, dv = phasing_maneuver(orb, 0.0)
    assert dv == pytest.approx(0.0, abs=1e-9)
    assert new.semi_major_axis == pytest.approx(1000.0, rel=1e-9)


def test_phasing_shifts_phase_keeps_orbit_shape():
    orb = make_circular(a=1000.0, nu=0.5)
    d = 0.3
    new, dv = phasing_maneuver(orb, d, n_rev=1)
    
    assert new.semi_major_axis == pytest.approx(1000.0, rel=1e-9)
    assert new.eccentricity == pytest.approx(0.0, abs=1e-9)
    assert new.inclination == pytest.approx(orb.inclination, rel=1e-9)
    assert dv > 0.0
    
    T = orb.period
    a_ph = (MU * (T * (1 - d / (2 * np.pi)) / (2 * np.pi)) ** 2) ** (1 / 3)
    v_ph = np.sqrt(MU * (2 / 1000.0 - 1 / a_ph))
    assert dv == pytest.approx(2 * abs(v_ph - np.sqrt(MU / 1000.0)), rel=1e-9)



def test_delta_v_accumulates_across_maneuvers():
    orb = make_circular(a=900.0)
    assert orb.delta_v == 0.0
    o1, dv1 = raise_orbit(orb, 300.0)
    o2, dv2 = plane_change(o1, 0.1)
    o3, dv3 = phasing_maneuver(o2, 0.2)
    assert o1.delta_v == pytest.approx(dv1, rel=1e-12)
    assert o2.delta_v == pytest.approx(dv1 + dv2, rel=1e-12)
    assert o3.delta_v == pytest.approx(dv1 + dv2 + dv3, rel=1e-12)
    
    assert orb.delta_v == 0.0


def test_maneuvers_do_not_mutate_input():
    orb = make_circular(a=1000.0)
    a0, i0, nu0 = orb.semi_major_axis, orb.inclination, orb.nu
    raise_orbit(orb, 200.0)
    plane_change(orb, 0.3)
    phasing_maneuver(orb, 0.4)
    assert orb.semi_major_axis == a0 and orb.inclination == i0 and orb.nu == nu0

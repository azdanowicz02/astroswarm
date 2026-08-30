
from astroswarm.analysis.plots import plot_coverage_comparison
from astroswarm.simulation import MetricsRecorder


class _FakeMap:
    def __init__(self, cov, qual=0.3):
        self._cov, self._qual = cov, qual
    def coverage_fraction(self):
        return self._cov
    def mean_quality(self):
        return self._qual
    def mean_reobservations(self):
        return 0.0


def _recorder(cov_series, target=0.95):
    m = MetricsRecorder(target_coverage=target)
    for t, cov in enumerate(cov_series):
        m.record(float(t) * 10.0, _FakeMap(cov), delta_v=float(t))
    return m


def test_plot_coverage_comparison_writes_file(tmp_path):
    
    fast = _recorder([0.0, 0.5, 0.9, 0.97])
    slow = _recorder([0.0, 0.2, 0.4, 0.6, 0.8, 0.96])
    out = tmp_path / "cmp.png"
    path = plot_coverage_comparison(
        {"greedy": fast, "random": slow}, out_path=out, target=0.95)
    assert path == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_coverage_comparison_without_target(tmp_path):
    r = _recorder([0.0, 0.3, 0.6])
    out = tmp_path / "cmp2.png"
    path = plot_coverage_comparison({"one": r}, out_path=out)  
    assert path.exists() and path.stat().st_size > 0

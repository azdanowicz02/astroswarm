
from astroswarm.utils.config import load_config


def test_default_config_has_expected_sections():
    cfg = load_config("configs/default.yaml")
    for section in ("asteroid", "sensor", "simulation", "orbit"):
        assert section in cfg


def test_merge_overrides_nested_keys():
    cfg = load_config("configs/default.yaml", "configs/swarm.yaml")
    
    assert "swarm" in cfg and "decision_weights" in cfg and "pheromone" in cfg
    
    assert "asteroid" in cfg

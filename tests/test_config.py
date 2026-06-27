from src.common.config import load_config


def test_default_config_locks_phase0_decisions():
    cfg = load_config()
    assert cfg["sport"] == "soccer"
    assert cfg["history"]["provider"] == "football-data.co.uk"
    assert cfg["live"]["market_source"] == "polymarket"
    assert cfg["vector_store"]["provider"] == "qdrant"
    assert cfg["timing"]["strict_before_kickoff"] is True

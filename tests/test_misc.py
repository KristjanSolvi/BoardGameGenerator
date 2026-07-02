import random

from gamegen.config import Config
from gamegen.inspiration import (FORBIDDEN_GAMES, MECHANIC_SEEDS,
                                 sample_inspiration)


def test_inspiration_deterministic_and_diverse():
    a = sample_inspiration(random.Random(42))
    b = sample_inspiration(random.Random(42))
    assert a.seeds == b.seeds
    assert "goal_type" in a.seeds
    assert 2 <= len(a.seeds) <= 3
    c = sample_inspiration(random.Random(43))
    different = any(
        sample_inspiration(random.Random(s)).seeds != a.seeds
        for s in range(43, 60)
    )
    assert different
    assert len(a.forbidden_games) >= 30
    assert a.forbidden_games == FORBIDDEN_GAMES


def test_seed_lists_nonempty():
    for category, seeds in MECHANIC_SEEDS.items():
        assert len(seeds) >= 4, category


def test_config_defaults_merge(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("backend: codex\nmodel: gpt-5.5\nseed: 7\n")
    cfg = Config.load(cfg_file)
    assert cfg.seed == 7
    assert cfg.limits["revision_cycles"] == 4  # default filled in
    assert cfg.playtest["move_cap"] == 400

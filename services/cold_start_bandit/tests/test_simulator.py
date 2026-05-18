"""End-to-end simulator test.

The bandit should beat random — i.e. its regret ratio over enough rounds
should be well below 1.0. We use a small/fast configuration so the test
finishes in a few seconds.
"""

from app.simulator import SimConfig, run_simulation


def test_simulator_runs_and_beats_random():
    cfg = SimConfig(rounds=400, n_clusters=4, videos_per_cluster=10, seed=0)
    result = run_simulation(cfg)
    assert result["rounds"] == 400
    # Bandit should be doing meaningfully better than a worst-case ratio of 1.0.
    assert result["regret_ratio"] < 0.5
    # And it should have populated arms.
    assert result["arms_used"] >= 1

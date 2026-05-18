from app.toxicity import StreamingToxicity


def test_empty_returns_none():
    assert StreamingToxicity().classify("") is None


def test_clean_returns_none():
    assert StreamingToxicity().classify("hello world this is fine") is None


def test_insult_classified_as_toxicity():
    r = StreamingToxicity().classify("you idiot")
    assert r is not None
    assert r.category == "toxicity"


def test_harassment_higher_priority_than_insult():
    """If both fire, the highest-severity result wins."""
    r = StreamingToxicity().classify("you stupid moron kys")
    assert r is not None
    assert r.category == "harassment"
    assert r.severity > 0.9

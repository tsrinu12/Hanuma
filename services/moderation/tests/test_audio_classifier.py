from app.audio_classifier import AudioClassifier
from app.schemas import AudioSegment


def test_clean_speech_no_flag():
    """Normal voice content: moderate loudness, high voicing -> no flag."""
    seg = AudioSegment(
        start_s=0.0, end_s=2.0,
        features=[-18.0, 4.0, -10.0, 0.1, 0.9],
    )
    clf = AudioClassifier(profile="lite")
    assert clf.classify([seg]) == []


def test_gunshot_proxy_loud_burst():
    """Very loud peak + low voicing -> violent burst flag."""
    seg = AudioSegment(
        start_s=10.0, end_s=11.0,
        features=[-20.0, 8.0, -1.0, 0.4, 0.05],
    )
    clf = AudioClassifier(profile="lite")
    flags = clf.classify([seg])
    assert any(f.category == "violence" and f.severity > 0.0 for f in flags)


def test_screaming_detected():
    """Sustained loud + high voicing -> harassment (screaming)."""
    seg = AudioSegment(
        start_s=5.0, end_s=7.0,
        features=[-5.0, 3.0, -3.0, 0.2, 0.85],
    )
    clf = AudioClassifier(profile="lite")
    flags = clf.classify([seg])
    assert any(f.category == "harassment" for f in flags)


def test_missing_features_skipped():
    seg = AudioSegment(start_s=0.0, end_s=1.0, features=None)
    clf = AudioClassifier(profile="lite")
    assert clf.classify([seg]) == []

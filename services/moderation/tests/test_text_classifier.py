from app.schemas import TranscriptSegment
from app.text_classifier import TextClassifier


def _t(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start_s=start, end_s=end, text=text)


def test_clean_text_no_flags():
    clf = TextClassifier(profile="lite")
    flags = clf.classify([_t(0, 5, "this is a perfectly normal sentence about cooking")])
    assert flags == []


def test_insult_flagged_as_toxicity():
    clf = TextClassifier(profile="lite")
    flags = clf.classify([_t(0, 5, "you are such a moron")])
    assert any(f.category == "toxicity" for f in flags)
    # spans propagate
    assert flags[0].start_s == 0.0
    assert flags[0].end_s == 5.0


def test_harassment_high_severity():
    clf = TextClassifier(profile="lite")
    flags = clf.classify([_t(0, 1, "kys")])
    assert any(f.category == "harassment" and f.severity > 0.9 for f in flags)


def test_violence_detected():
    clf = TextClassifier(profile="lite")
    flags = clf.classify([_t(0, 1, "I will kill them")])
    assert any(f.category == "violence" for f in flags)


def test_each_segment_independent():
    clf = TextClassifier(profile="lite")
    flags = clf.classify(
        [
            _t(0, 5, "normal content here"),
            _t(5, 10, "you are stupid"),
            _t(10, 15, "more normal content"),
        ]
    )
    assert len(flags) == 1
    assert flags[0].start_s == 5.0

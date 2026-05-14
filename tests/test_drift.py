from app.drift import build_reference, compute_drift, drift_state


def _events(texts, labels, lang="en"):
    return [
        {"text_len": len(t), "text_lang": lang, "label": labels[i]}
        for i, t in enumerate(texts)
    ]


def test_identical_window_is_not_drift():
    drift_state.token_buffer.clear()
    texts = ["the movie was great"] * 60 + ["the movie was bad"] * 60
    labels = ["POSITIVE"] * 60 + ["NEGATIVE"] * 60
    ref = build_reference(texts, labels)
    for t in texts:
        drift_state.add(t)
    window = _events(texts, labels, lang="en")
    out = compute_drift(window, ref)
    assert out["overall_status"] == "ok"
    triggered = [s for s in out["signals"] if s["triggered"]]
    assert triggered == []


def test_foreign_language_window_is_drift():
    drift_state.token_buffer.clear()
    ref_texts = ["the movie was great"] * 100
    ref_labels = ["POSITIVE"] * 100
    ref = build_reference(ref_texts, ref_labels)
    foreign = ["bonjour le monde tres bien"] * 100
    for t in foreign:
        drift_state.add(t)
    window = _events(foreign, ["POSITIVE"] * 100, lang="fr")
    out = compute_drift(window, ref)
    assert out["overall_status"] == "drift_detected"
    names_triggered = {s["name"] for s in out["signals"] if s["triggered"]}
    assert "non_english_ratio" in names_triggered


def test_insufficient_data_when_no_window():
    out = compute_drift([], {"lengths": [1, 2, 3], "top_100_tokens": [], "label_positive_ratio": 0.5})
    assert out["overall_status"] == "insufficient_data"

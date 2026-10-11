from eval.text_pipeline import chunk_text, clean_text


def test_clean_strips_control_chars():
    raw = "Hello\x00world  \n\n\nextra"
    out = clean_text(raw)
    assert "world" in out
    assert "\x00" not in out


def test_chunk_non_empty():
    text = "параграф один\n\n" + ("word " * 200)
    parts = chunk_text(text)
    assert len(parts) >= 1
    assert all(len(p) > 0 for p in parts)

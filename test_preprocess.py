from preprocess import clean_text, is_hinglish


def test_is_hinglish_detects_code_mixed():
    assert is_hinglish("aap kahan ho?") is True
    assert is_hinglish("mein market ja raha hoon") is True
    assert is_hinglish("hi mam, kya haal hai?") is True


def test_is_hinglish_false_for_english():
    assert is_hinglish("Hello, are we still meeting at 6?") is False
    assert is_hinglish("WIN cash now!!!") is False


def test_clean_text_hinglish_optional_off_by_default():
    # Default: Hinglish stop words are NOT removed (backward compatible).
    assert clean_text("aap kahan ho") == "aap kahan ho"


def test_clean_text_removes_hinglish_when_requested():
    assert clean_text("dost kahan hai", remove_hinglish=True) == "dost"
    assert clean_text("mein market ja raha hoon", remove_hinglish=True) == "market ja"


def test_clean_text_english_unchanged_with_hinglish_flag():
    assert clean_text("Hello, meeting today?", remove_hinglish=True) == "hello meeting today"
    assert clean_text("WIN cash now!!!", remove_hinglish=True) == "win cash"

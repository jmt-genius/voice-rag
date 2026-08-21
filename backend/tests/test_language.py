from app.config import language_filter


def test_ui_codes_map_to_index_languages():
    assert language_filter("ta-IN") == "tam_Taml"
    assert language_filter("hi-IN") == "hin_Deva"
    assert language_filter("en-IN") == "en"
    assert language_filter("bn-IN") == "ben_Beng"


def test_index_language_values_pass_through():
    assert language_filter("ben_Beng") == "ben_Beng"
    assert language_filter("tam_Taml") == "tam_Taml"


def test_unknown_or_missing_code_searches_all():
    assert language_filter(None) is None
    assert language_filter("fr-FR") is None
    assert language_filter("") is None
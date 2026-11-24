from datetime import datetime

from config import VARIABLES
from i18n import LANGS, STRINGS, fmt_day, fmt_day_time, t


def test_same_keys_in_every_language():
    keys = {lang: set(STRINGS[lang]) for lang in LANGS}
    first = keys["pt"]
    for lang, k in keys.items():
        assert k == first, f"{lang}: missing {first - k}, extra {k - first}"


def test_every_variable_and_group_is_translated():
    for lang in LANGS:
        for v in VARIABLES:
            for field in ("label", "short", "help"):
                assert f"var.{v.col}.{field}" in STRINGS[lang]
            assert f"group.{v.group}" in STRINGS[lang]


def test_placeholders_match():
    import string
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}
    for key, pt in STRINGS["pt"].items():
        assert fields(pt) == fields(STRINGS["en"][key]), key


def test_t_formats_and_falls_back():
    assert t("good_hours", "en", n=3) == "📋 Good Fishing Hours (3)"
    assert t("good_hours", "xx", n=3) == "📋 Boas horas de pesca (3)"
    assert t("no.such.key", "en") == "no.such.key"


def test_day_formatting():
    d = datetime(2026, 11, 24, 7, 5)  # a Tuesday
    assert fmt_day(d, "pt") == "Ter 24/11"
    assert fmt_day(d, "en") == "Tue 24/11"
    assert fmt_day_time(d, "pt") == "Ter 24/11 07:05"

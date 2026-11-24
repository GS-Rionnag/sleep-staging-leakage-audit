import pytest

from sleepaudit.records import (
    build_manifest,
    load_manifest,
    pair_files,
    parse_name,
    select_recordings,
)


def test_parse_name():
    assert parse_name("SC4001E0-PSG.edf") == (0, 1)
    assert parse_name("SC4022EJ-Hypnogram.edf") == (2, 2)
    with pytest.raises(ValueError):
        parse_name("ST7011J0-PSG.edf")


def test_pair_scorer_codes_differ():
    recs = pair_files(
        [
            "SC4021E0-PSG.edf",
            "SC4021EH-Hypnogram.edf",
            "SC4022E0-PSG.edf",
            "SC4022EJ-Hypnogram.edf",
        ]
    )
    assert [r.stem for r in recs] == ["SC4021E", "SC4022E"]
    assert recs[1].hyp_file == "SC4022EJ-Hypnogram.edf"
    assert (recs[1].subject, recs[1].night) == (2, 2)


def test_pair_missing_hypnogram_raises():
    with pytest.raises(ValueError, match="SC4021E"):
        pair_files(["SC4021E0-PSG.edf"])


def _nights(recs):
    out: dict[int, set[int]] = {}
    for r in recs:
        out.setdefault(r.subject, set()).add(r.night)
    return out


def test_sc20_shape():
    recs = select_recordings("sc20")
    nights = _nights(recs)
    assert len(recs) == 39
    assert len(nights) == 20
    assert nights[13] == {1}


def test_sc78_shape():
    recs = select_recordings("sc78")
    nights = _nights(recs)
    assert len(recs) == 153
    assert len(nights) == 78
    assert set(range(83)) - set(nights) == {39, 68, 69, 78, 79}
    assert nights[36] == {2} and nights[52] == {2}
    assert sum(len(n) == 2 for n in nights.values()) == 75


def test_explicit_subset():
    assert len(select_recordings([0, 1, 2, 3])) == 8


def test_build_manifest_filters_st():
    h = "a" * 64
    sums = f"{h}  sleep-cassette/SC4001E0-PSG.edf\n{h}  sleep-telemetry/ST7011J0-PSG.edf\n"
    df = build_manifest(sums, "sleep-cassette/SC4001E0-PSG.edf\nsleep-telemetry/ST7011J0-PSG.edf\n")
    assert len(df) == 1
    row = df.iloc[0]
    assert row.filename == "SC4001E0-PSG.edf"
    assert row.kind == "psg"
    assert (row.subject, row.night) == (0, 1)
    assert row.sha256 == h
    assert list(df.columns) == ["filename", "kind", "subject", "night", "sha256", "size"]


def test_manifest_columns():
    assert list(load_manifest().columns) == [
        "filename", "kind", "subject", "night", "sha256", "size",
    ]

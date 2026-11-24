import hashlib

import pytest
import requests

from sleepaudit.download import ChecksumError, edf_duration_seconds, fetch_file

GOOD = b"good bytes"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
NAME = "SC4001E0-PSG.edf"


class FakeGetter:
    def __init__(self, handler):
        self.handler = handler
        self.urls: list[str] = []

    def __call__(self, url, headers):
        self.urls.append(url)
        return self.handler(url, headers)


def test_checksum_mismatch_leaves_no_file(tmp_path):
    get = FakeGetter(lambda url, headers: b"bad")
    with pytest.raises(ChecksumError):
        fetch_file(NAME, GOOD_SHA, tmp_path, get)
    assert not (tmp_path / NAME).exists()
    assert not (tmp_path / f"{NAME}.part").exists()


def test_fallback_used_on_primary_error(tmp_path):
    def handler(url, headers):
        if "physionet.org" in url:
            raise requests.HTTPError("503")
        return GOOD

    get = FakeGetter(handler)
    path = fetch_file(NAME, GOOD_SHA, tmp_path, get)
    assert path.read_bytes() == GOOD
    assert len(get.urls) == 2
    assert get.urls[0].startswith("https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/")
    assert get.urls[1].startswith("https://physionet-open.s3.amazonaws.com/")


def test_existing_good_file_skipped(tmp_path):
    get = FakeGetter(lambda url, headers: GOOD)
    fetch_file(NAME, GOOD_SHA, tmp_path, get)
    calls = len(get.urls)
    fetch_file(NAME, GOOD_SHA, tmp_path, get)
    assert len(get.urls) == calls


def test_edf_duration():
    header = bytearray(b" " * 256)
    header[236:244] = b"84      "
    header[244:252] = b"30      "
    assert edf_duration_seconds(bytes(header)) == 2520.0

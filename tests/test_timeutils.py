from datetime import datetime, timedelta, timezone

import pytest

from src.common.timeutils import now_utc, to_utc


def test_parses_z_suffix():
    assert to_utc("2026-06-27T18:00:00Z") == datetime(2026, 6, 27, 18, tzinfo=timezone.utc)


def test_converts_offset_to_utc():
    # 18:00 at -03:00 is 21:00 UTC
    assert to_utc("2026-06-27T18:00:00-03:00").hour == 21


def test_rejects_naive_datetime():
    with pytest.raises(ValueError):
        to_utc(datetime(2026, 6, 27, 18, 0, 0))


def test_assume_tz_escape_hatch():
    est = timezone(timedelta(hours=-5))
    assert to_utc(datetime(2026, 6, 27, 18), assume_tz=est).hour == 23


def test_now_utc_is_aware():
    assert now_utc().tzinfo is not None

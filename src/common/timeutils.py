"""UTC time handling. Naive (tz-less) timestamps are the classic source of silent
leakage (off-by-hours comparisons), so we reject them at the boundary."""
from __future__ import annotations

from datetime import datetime, timezone, tzinfo

UTC = timezone.utc


def now_utc() -> datetime:
    return datetime.now(UTC)


def to_utc(value: datetime | str, *, assume_tz: tzinfo | None = None) -> datetime:
    """Coerce a datetime or ISO-8601 string to a tz-aware UTC datetime.

    Naive input is rejected unless `assume_tz` is given — better to fail loudly
    than to compare an unknown-zone timestamp against kickoff and leak.
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError(f"expected datetime or ISO string, got {type(value).__name__}")
    if value.tzinfo is None:
        if assume_tz is None:
            raise ValueError("naive datetime rejected; pass tz-aware input or assume_tz=")
        value = value.replace(tzinfo=assume_tz)
    return value.astimezone(UTC)

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.ingest.news_guardian import assign_date_with_cutoff, validate_cutoff_assignments


def test_assign_date_cutoff_edges():
    tz = ZoneInfo("Europe/London")

    # exactly at cutoff -> same day
    dt_at = datetime(2024, 11, 4, 16, 30, 0, tzinfo=tz)
    assert assign_date_with_cutoff(dt_at, cutoff_hhmm="16:30").isoformat() == "2024-11-04"

    # 1 second after cutoff -> next day
    dt_after = datetime(2024, 11, 4, 16, 30, 1, tzinfo=tz)
    assert assign_date_with_cutoff(dt_after, cutoff_hhmm="16:30").isoformat() == "2024-11-05"

    # before cutoff -> same day
    dt_before = datetime(2024, 11, 4, 9, 15, 0, tzinfo=tz)
    assert assign_date_with_cutoff(dt_before, cutoff_hhmm="16:30").isoformat() == "2024-11-04"


def test_validate_cutoff_assignments_detects_violation():
    tz = ZoneInfo("Europe/London")
    pub = datetime(2024, 11, 4, 17, 0, 0, tzinfo=tz)  # after cutoff

    df = pd.DataFrame(
        [
            {
                "ticker": "HSBA.L",
                "published_at_london": pub.isoformat(),
                "assigned_date": "2024-11-04",  # WRONG; should be 2024-11-05
            }
        ]
    )

    with pytest.raises(ValueError):
        validate_cutoff_assignments(df, cutoff_hhmm="16:30")

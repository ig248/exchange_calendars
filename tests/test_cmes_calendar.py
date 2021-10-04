from __future__ import annotations
from collections import abc

import pytest
import pandas as pd
from pytz import UTC

from exchange_calendars.exchange_calendar_cmes import CMESExchangeCalendar
from exchange_calendars import ExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBaseNew, Answers


class TestCMESCalendar(ExchangeCalendarTestBaseNew):
    @pytest.fixture(scope="class", params=["left", "right"])
    def all_calendars_with_answers(
        self, request, calendars, answers
    ) -> abc.Iterator[ExchangeCalendar, Answers]:
        """Parameterized calendars and answers for each side."""
        yield (calendars[request.param], answers[request.param])

    @pytest.fixture(scope="class")
    def calendar_cls(self) -> abc.Iterator[ExchangeCalendar]:
        yield CMESExchangeCalendar

    @pytest.fixture(scope="class")
    def max_session_hours(self) -> abc.Iterator[int | float]:
        yield 24

    # Additional tests

    def test_2016_holidays(self, default_calendar):
        # good friday, christmas, new years
        for date in ["2016-03-25", "2016-12-26", "2016-01-02"]:
            assert not default_calendar.is_session(
                pd.Timestamp(date, tz=UTC), _parse=False
            )

    def test_2016_early_closes(self, default_calendar):
        cal = default_calendar
        # TODO HERERE, WOULD THIS BE QUICKER IF COMPARED TWO INDICES?
        for date in [
            "2016-01-18",  # mlk day
            "2016-02-15",  # presidents
            "2016-05-30",  # mem day
            "2016-07-04",  # july 4
            "2016-09-05",  # labor day
            "2016-11-24",  # thanksgiving
        ]:
            dt = pd.Timestamp(date, tz=UTC)
            assert dt in cal.early_closes
            assert cal.closes[dt].tz_localize(UTC).tz_convert(cal.tz).hour == 12

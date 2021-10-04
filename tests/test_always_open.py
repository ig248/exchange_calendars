from __future__ import annotations
from collections import abc

import pytest
import pandas as pd
import pandas.testing as tm
from pytz import UTC

from exchange_calendars.always_open import AlwaysOpenCalendar
from exchange_calendars import ExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBaseNew, Answers


class TestAlwaysOpenCalendar(ExchangeCalendarTestBaseNew):
    @pytest.fixture(scope="class", params=["left", "right"])
    def all_calendars_with_answers(
        self, request, calendars, answers
    ) -> abc.Iterator[ExchangeCalendar, Answers]:
        """Parameterized calendars and answers for each side."""
        yield (calendars[request.param], answers[request.param])

    @pytest.fixture(scope="class")
    def calendar_cls(self) -> abc.Iterator[ExchangeCalendar]:
        yield AlwaysOpenCalendar

    @pytest.fixture(scope="class")
    def max_session_hours(self) -> abc.Iterator[int | float]:
        yield 24

    # Additional tests

    def test_open_every_day(self, default_calendar_with_answers):
        cal, ans = default_calendar_with_answers
        dates = pd.date_range(*ans.sessions_range, tz=UTC)
        tm.assert_index_equal(cal.all_sessions, dates)

    def test_open_every_minute(self, calendars, answers, one_minute):
        cal, ans = calendars["left"], answers["left"]
        minutes = pd.date_range(*ans.trading_minutes_range, freq="min", tz=UTC)
        tm.assert_index_equal(cal.all_minutes, minutes)

        cal = calendars["right"]
        minutes += one_minute
        tm.assert_index_equal(cal.all_minutes, minutes)

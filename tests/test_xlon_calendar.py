from __future__ import annotations
from collections import abc

import pytest

from exchange_calendars.exchange_calendar_xlon import XLONExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBaseNew
from .test_utils import T


class TestXLONCalendar(ExchangeCalendarTestBaseNew):
    @pytest.fixture(scope="class")
    def calendar_cls(self) -> abc.Iterator[XLONExchangeCalendar]:
        yield XLONExchangeCalendar

    @pytest.fixture(scope="class")
    def max_session_hours(self) -> abc.Iterator[int | float]:
        yield 8.5

    # Additional tests

    def test_2012(self, default_calendar):
        cal = default_calendar
        expected_holidays_2012 = [
            T("2012-01-02"),  # New Year's observed
            T("2012-04-06"),  # Good Friday
            T("2012-04-09"),  # Easter Monday
            T("2012-05-07"),  # May Day
            T("2012-06-04"),  # Spring Bank Holiday
            T("2012-08-27"),  # Summer Bank Holiday
            T("2012-12-25"),  # Christmas
            T("2012-12-26"),  # Boxing Day
        ]

        for session in expected_holidays_2012:
            assert session not in cal.all_sessions

        early_closes_2012 = [
            T("2012-12-24"),  # Christmas Eve
            T("2012-12-31"),  # New Year's Eve
        ]

        for session in early_closes_2012:
            assert session in cal.early_closes

    def test_special_holidays(self, default_calendar):
        cal = default_calendar
        special_holidays = [
            T("2002-06-03"),  # Spring Bank 2002
            T("2002-06-04"),  # Golden Jubilee
            T("2011-04-29"),  # Royal Wedding
            T("2012-06-04"),  # Spring Bank 2012
            T("2012-06-05"),  # Diamond Jubilee
            T("2020-05-08"),  # VE Day
        ]

        for holiday in special_holidays:
            assert holiday not in cal.all_sessions

    def test_special_non_holidays(self, default_calendar):
        # May Bank Holiday was instead observed on VE Day in 2020.
        assert T("2020-05-04") in default_calendar.all_sessions

    def test_specific_early_closes(self, default_calendar):
        early_closes = [
            # In Dec 2010, Christmas Eve and NYE are on Friday.
            T("2010-12-24"),
            T("2010-12-31"),
            # In Dec 2011, Christmas Eve and NYE were both on a Saturday,
            # so preceding Fridays (the 23rd and 30th) are early closes.
            T("2011-12-23"),
            T("2011-12-30"),
        ]

        for session in early_closes:
            assert session in default_calendar.early_closes

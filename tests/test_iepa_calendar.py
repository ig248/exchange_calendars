import datetime

import pytest

from exchange_calendars.exchange_calendar_iepa import IEPAExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBaseNew


class TestIEPACalendar(ExchangeCalendarTestBaseNew):
    @pytest.fixture(scope="class")
    def calendar_cls(self):
        yield IEPAExchangeCalendar

    @pytest.fixture(scope="class")
    def max_session_hours(self):
        yield 22

    @pytest.fixture(scope="class")
    def regular_holidays_sample(self):
        # new year's, good friday, christmas
        yield ["2016-01-01", "2016-03-25", "2016-12-26"]

    @pytest.fixture(scope="class")
    def adhoc_holidays_sample(self):
        yield ["2012-10-29"]  # hurricane sandy (day one)

    @pytest.fixture(scope="class")
    def sessions_sample(self):
        yield ["2012-10-30"]  # hurricane sandy day two - exchange open

    @pytest.fixture(scope="class")
    def early_closes_sample(self):
        yield [
            "2016-01-18",  # mlk
            "2016-02-15",  # presidents
            "2016-05-30",  # mem day
            "2016-07-04",  # independence day
            "2016-09-05",  # labor
            "2016-11-24",  # thanksgiving
        ]

    # Calendar-specific tests

    def test_early_close_time(self, default_calendar, early_closes_sample):
        cal = default_calendar
        for early_close in early_closes_sample:
            local_close = cal.closes[early_close].tz_localize("UTC").tz_convert(cal.tz)
            assert local_close.time() == datetime.time(13, 0)

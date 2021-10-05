import datetime

import pytest

from exchange_calendars.exchange_calendar_cmes import CMESExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBaseNew


class TestCMESCalendar(ExchangeCalendarTestBaseNew):
    @pytest.fixture(scope="class", params=["left", "right"])
    def all_calendars_with_answers(self, request, calendars, answers):
        yield (calendars[request.param], answers[request.param])

    @pytest.fixture(scope="class")
    def calendar_cls(self):
        yield CMESExchangeCalendar

    @pytest.fixture(scope="class")
    def max_session_hours(self):
        yield 24

    @pytest.fixture(scope="class")
    def regular_holidays_sample(self):
        # good friday, christmas, new years
        yield ["2016-03-25", "2016-12-26", "2016-01-02"]

    @pytest.fixture(scope="class")
    def early_closes_sample(self):
        yield [
            "2016-01-18",  # mlk day
            "2016-02-15",  # presidents
            "2016-05-30",  # mem day
            "2016-07-04",  # july 4
            "2016-09-05",  # labor day
            "2016-11-24",  # thanksgiving
        ]

    # Calendar-specific tests

    def test_early_close_time(self, default_calendar, early_closes_sample):
        cal = default_calendar
        for early_close in early_closes_sample:
            close_time = cal.closes[early_close].tz_localize("UTC").tz_convert(cal.tz)
            assert close_time.time() == datetime.time(12, 0)

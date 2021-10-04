from __future__ import annotations
from datetime import time
from collections import abc

import pytest
import pandas as pd

from exchange_calendars.exchange_calendar_xhkg import XHKGExchangeCalendar

from .test_exchange_calendar import ExchangeCalendarTestBaseNew
from .test_utils import T


class TestXHKGCalendar(ExchangeCalendarTestBaseNew):
    @pytest.fixture(scope="class")
    def calendar_cls(self) -> abc.Iterator[XHKGExchangeCalendar]:
        yield XHKGExchangeCalendar

    @pytest.fixture(scope="class")
    def max_session_hours(self) -> abc.Iterator[int | float]:
        yield 6.5

    @pytest.fixture(scope="class")
    def start_bound(self) -> abc.Iterator[pd.Timestamp | None]:
        """Earliest date for which calendar can be instantiated, or None if
        there is no start bound."""
        yield T("1960-01-01")

    @pytest.fixture(scope="class")
    def end_bound(self) -> abc.Iterator[pd.Timestamp | None]:
        """Latest date for which calendar can be instantiated, or None if
        there is no end bound."""
        yield T("2049-12-31")

    # Additional tests

    def test_lunar_new_year_2003(self, default_calendar):
        # NOTE: Lunar Month 12 2002 is the 12th month of the lunar year that begins
        # in 2002; this month actually takes place in January 2003.

        # Lunar Month 12 2002
        #   January-January
        # Su Mo Tu We Th Fr Sa
        #                 3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        #  Lunar Month 1 2003
        #    February-March
        # Su Mo Tu We Th Fr Sa
        #                    1
        #  2  3  4  5  6  7  8
        #  9 10 11 12 13 14 15
        # 16 17 18 19 20 21 22
        # 23 24 25 26 27 28  1
        #  2

        # Prior to 2011, lunar new years eve is a holiday if new years is a Saturday.
        holidays = [
            T("2003-01-31"),
            T("2003-02-03"),
        ]
        for holiday in holidays:
            assert holiday not in default_calendar.all_sessions

    def test_lunar_new_year_2018(self, default_calendar):
        # NOTE: Lunar Month 12 2017 is the 12th month of the lunar year that begins
        # in 2017; this month actually takes place in January and February 2018.

        # Lunar Month 12 2017
        #   January-February
        # Su Mo Tu We Th Fr Sa
        #          17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31  1  2  3
        #  4  5  6  7  8  9 10
        # 11 12 13 14 15

        #  Lunar Month 1 2018
        #    February-March
        # Su Mo Tu We Th Fr Sa
        #                16 17
        # 18 19 20 21 22 23 24
        # 25 26 27 28  1  2  3
        #  4  5  6  7  8  9 10
        # 11 12 13 14 15 16
        cal = default_calendar
        holidays = [
            T("2018-02-16"),
            T("2018-02-19"),
        ]
        for holiday in holidays:
            assert holiday not in cal.all_sessions

        early_close = pd.Timestamp("2018-02-15 12:00", tz="Asia/Hong_Kong")
        assert cal.closes.loc["2018-02-15"].tz_localize("UTC") == early_close

    def test_full_year_with_lunar_leap_year(self, default_calendar):
        # 2017 Lunar month 6 will be a leap month (double length). This
        # affects when all the lunisolar holidays after the 6th month occur.

        #                         Gregorian Calendar
        #                                2017
        #
        #        January               February                 March
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7             1  2  3  4             1  2  3  4
        #  8  9 10 11 12 13 14    5  6  7  8  9 10 11    5  6  7  8  9 10 11
        # 15 16 17 18 19 20 21   12 13 14 15 16 17 18   12 13 14 15 16 17 18
        # 22 23 24 25 26 27 28   19 20 21 22 23 24 25   19 20 21 22 23 24 25
        # 29 30 31               26 27 28               26 27 28 29 30 31
        #
        #         April                   May                   June
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #                    1       1  2  3  4  5  6                1  2  3
        #  2  3  4  5  6  7  8    7  8  9 10 11 12 13    4  5  6  7  8  9 10
        #  9 10 11 12 13 14 15   14 15 16 17 18 19 20   11 12 13 14 15 16 17
        # 16 17 18 19 20 21 22   21 22 23 24 25 26 27   18 19 20 21 22 23 24
        # 23 24 25 26 27 28 29   28 29 30 31            25 26 27 28 29 30
        # 30
        #         July                  August                September
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #                    1          1  2  3  4  5                   1  2
        #  2  3  4  5  6  7  8    6  7  8  9 10 11 12    3  4  5  6  7  8  9
        #  9 10 11 12 13 14 15   13 14 15 16 17 18 19   10 11 12 13 14 15 16
        # 16 17 18 19 20 21 22   20 21 22 23 24 25 26   17 18 19 20 21 22 23
        # 23 24 25 26 27 28 29   27 28 29 30 31         24 25 26 27 28 29 30
        # 30 31
        #        October               November               December
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7             1  2  3  4                   1  2
        #  8  9 10 11 12 13 14    5  6  7  8  9 10 11    3  4  5  6  7  8  9
        # 15 16 17 18 19 20 21   12 13 14 15 16 17 18   10 11 12 13 14 15 16
        # 22 23 24 25 26 27 28   19 20 21 22 23 24 25   17 18 19 20 21 22 23
        # 29 30 31               26 27 28 29 30         24 25 26 27 28 29 30
        #                                               31

        #                           Lunar Calendar
        #                                2017
        #
        #    Lunar Month 1          Lunar Month 2          Lunar Month 3
        #   January-February        February-March          March-April
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #                   28                                28 29 30 31  1
        # 29 30 31  1  2  3  4   26 27 28  1  2  3  4    2  3  4  5  6  7  8
        #  5  6  7  8  9 10 11    5  6  7  8  9 10 11    9 10 11 12 13 14 15
        # 12 13 14 15 16 17 18   12 13 14 15 16 17 18   16 17 18 19 20 21 22
        # 19 20 21 22 23 24 25   19 20 21 22 23 24 25   23 24 25
        #                        26 27
        #
        #    Lunar Month 4          Lunar Month 5         Lunar Month 6(+)
        #      April-May               May-June             June-August
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #          26 27 28 29                  26 27                     24
        # 30  1  2  3  4  5  6   28 29 30 31  1  2  3   25 26 27 28 29 30  1
        #  7  8  9 10 11 12 13    4  5  6  7  8  9 10    2  3  4  5  6  7  8
        # 14 15 16 17 18 19 20   11 12 13 14 15 16 17    9 10 11 12 13 14 15
        # 21 22 23 24 25         18 19 20 21 22 23      16 17 18 19 20 21 22
        #                                               23 24 25 26 27 28 29
        #                                               30 31  1  2  3  4  5
        #                                                6  7  8  9 10 11 12
        #                                               13 14 15 16 17 18 19
        #                                               20 21
        #
        #    Lunar Month 7          Lunar Month 8          Lunar Month 9
        #   August-September      September-October       October-November
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #       22 23 24 25 26            20 21 22 23                  20 21
        # 27 28 29 30 31  1  2   24 25 26 27 28 29 30   22 23 24 25 26 27 28
        #  3  4  5  6  7  8  9    1  2  3  4  5  6  7   29 30 31  1  2  3  4
        # 10 11 12 13 14 15 16    8  9 10 11 12 13 14    5  6  7  8  9 10 11
        # 17 18 19               15 16 17 18 19         12 13 14 15 16 17
        #
        #    Lunar Month 10         Lunar Month 11         Lunar Month 12
        #  November-December       December-January       January-February
        # Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
        #                   18      18 19 20 21 22 23            17 18 19 20
        # 19 20 21 22 23 24 25   24 25 26 27 28 29 30   21 22 23 24 25 26 27
        # 26 27 28 29 30  1  2   31  1  2  3  4  5  6   28 29 30 31  1  2  3
        #  3  4  5  6  7  8  9    7  8  9 10 11 12 13    4  5  6  7  8  9 10
        # 10 11 12 13 14 15 16   14 15 16               11 12 13 14 15
        # 17
        full_holidays = [
            T("2017-01-02"),  # New Year's Day (Sunday to Monday observance)
            T("2017-01-30"),  # Lunar New Year
            T("2017-01-31"),  # Lunar New Year
            T("2017-04-04"),  # Qingming Festival (off qingming solar term, not lunar)
            T("2017-04-14"),  # Good Friday
            T("2017-04-17"),  # Easter Monday
            T("2017-05-01"),  # Labour Day
            T("2017-05-03"),  # Buddha's Birthday (The 8th day of the 4th lunar month)
            # Tuen Ng Festival (also known as Dragon Boat Festival. The 5th day
            # of the 5th lunar month, then Sunday to Monday observance)
            T("2017-05-30"),
            T("2017-10-02"),  # National Day (Sunday to Monday observance)
            # The day following the Mid-Autumn Festival (Mid-Autumn Festival
            # is the 15th day of the 8th lunar month. This market holiday is
            # next day because the festival is celebrated at night)
            T("2017-10-05"),
            T("2017-12-25"),  # Christmas Day
            T("2017-12-26"),  # The day after Christmas
        ]

        early_closes = [
            T("2017-01-27"),  # Lunar New Year's Eve
            # Christmas Eve and New Year's Eve are both Sunday this year
        ]

        cal = default_calendar
        closes = cal.session_closes_in_range("2017-01-01", "2017-12-31")

        for holiday in full_holidays:
            assert holiday not in cal.all_sessions

        for early_close in early_closes:
            close_time = early_close.tz_convert(None).tz_localize(
                "Asia/Hong_Kong"
            ) + pd.Timedelta(hours=12)
            assert close_time == closes[early_close]

        local_time_close = closes.drop(early_closes).dt.tz_convert(
            "Asia/Hong_Kong",
        )
        assert {time(16)} == set(local_time_close.dt.time)

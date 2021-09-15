#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from abc import ABC, abstractproperty
from collections import OrderedDict
import functools
import warnings

import numpy as np
import pandas as pd
import toolz
from numpy import searchsorted
from pandas import DataFrame, date_range
from pandas.tseries.holiday import AbstractHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pytz import UTC

from exchange_calendars import errors
from .calendar_helpers import (
    NP_NAT,
    compute_all_minutes,
    one_minute_later,
    one_minute_earlier,
    next_divider_idx,
    previous_divider_idx,
    Session,
    Date,
    Minute,
    parse_timestamp,
    parse_session,
    parse_date,
)
from .utils.memoize import lazyval
from .utils.pandas_utils import days_at_time
from .pandas_extensions.offsets import MultipleWeekmaskCustomBusinessDay

GLOBAL_DEFAULT_START = pd.Timestamp.now(tz=UTC).floor("D") - pd.DateOffset(years=20)
# Give an aggressive buffer for logic that needs to use the next trading
# day or minute.
GLOBAL_DEFAULT_END = pd.Timestamp.now(tz=UTC).floor("D") + pd.DateOffset(years=1)

NANOS_IN_MINUTE = 60000000000
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY)
WEEKENDS = (SATURDAY, SUNDAY)


def selection(arr, start, end):
    predicates = []
    if start is not None:
        predicates.append(start.tz_localize(UTC) <= arr)
    if end is not None:
        predicates.append(arr < end.tz_localize(UTC))

    if not predicates:
        return arr

    return arr[np.all(predicates, axis=0)]


def _group_times(all_days, times, tz, offset=0):
    if times is None:
        return None
    elements = [
        days_at_time(selection(all_days, start, end), time, tz, offset)
        for (start, time), (end, _) in toolz.sliding_window(
            2, toolz.concatv(times, [(None, None)])
        )
    ]
    return elements[0].append(elements[1:])


class deprecate:
    """Decorator for deprecated/renamed ExchangeCalendar methods."""

    def __init__(
        self,
        deprecated_release: str = "3.4",
        removal_release: str = "4.0",
        alt_method: str = "",
        renamed: bool = True,
    ):
        self.deprecated_release = "release " + deprecated_release
        self.removal_release = "release " + removal_release
        self.alt_method = alt_method
        self.renamed = renamed
        if renamed:
            assert alt_method, "pass `alt_method` if renaming"

    def __call__(self, f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            warnings.warn(self._message(f), FutureWarning)
            return f(*args, **kwargs)

        return wrapped_f

    def _message(self, f):
        msg = (
            f"`{f.__name__}` was deprecated in {self.deprecated_release}"
            f" and will be removed in {self.removal_release}."
        )
        if self.alt_method:
            if self.renamed:
                msg += (
                    f" The method has been renamed `{self.alt_method}`."
                    f" NB parameter names may also have changed (see "
                    f" documentation for `{self.alt_method}`)."
                )
            else:
                msg += f" Use `{self.alt_method}`."
        return msg


class ExchangeCalendar(ABC):
    """Representation of timing information of a single market exchange.

    The timing information comprises sessions, open/close times and, for
    exchanges that observe an intraday break, break_start/break_end times.

    For exchanges that do not observe an intraday break a session
    represents a contiguous set of minutes. Where an exchange observes
    an intraday break a session represents two contiguous sets of minutes
    separated by the intraday break.

    Each session has a label that is midnight UTC. It is important to note
    that a session label should not be considered a specific point in time,
    and that midnight UTC is just being used for convenience.

    For each session, we store the open and close time together with, for
    those exchanges with breaks, the break start and break end. All times
    are defined as UTC.

    Parameters
    ----------
    start : default: later of 20 years ago or first supported start date.
        First calendar session will be `start`, if `start` is a session, or
        first session after `start`.

    end : default: earliest of 1 year from 'today' or last supported end date.
        Last calendar session will be `end`, if `end` is a session, or last
        session before `end`.

    side : default: "both" ("left" for 24 hour calendars)
        Define which of session open/close and break start/end should
            be treated as a trading minute:
        "left" - treat session open and break_start as trading minutes,
            do not treat session close or break_end as trading minutes.
        "right" - treat session close and break_end as trading minutes,
            do not treat session open or break_start as tradng minutes.
        "both" - treat all of session open, session close, break_start
            and break_end as trading minutes.
        "neither" - treat none of session open, session close,
            break_start or break_end as trading minutes.

    Raises
    ------
    ValueError
        If `start` is earlier than the earliest supported start date.
        If `end` is later than the latest supported end date.
        If `start` parses to a later date than `end`.

    Notes
    -----
    Exchange calendars were originally defined for the Zipline package from
    Quantopian under the package 'trading_calendars'. Since 2021 they have
    been maintained under the 'exchange_calendars' package (a fork of
    'trading_calendars') by an active community of contributing users.

    Some calendars have defined start and end bounds within which
    contributors have endeavoured to ensure the calendar's accuracy and
    outside of which the calendar would not be accurate. These bounds
    are enforced such that passing `start` or `end` as dates that are
    out-of-bounds will raise a ValueError. The bounds of each calendar are
    exposed via the `bound_start` and `bound_end` properties.

    Many calendars do not have bounds defined (in these cases `bound_start`
    and/or `bound_end` return None). These calendars can be created through
    any date range although it should be noted that the earlier the start
    date, the greater the potential for inaccuracies.

    In all cases, no guarantees are offered as to the accuracy of any
    calendar.


    Internal method parameters:

        _parse: bool
            Determines if a `minute` or `session` parameter should be
            parsed (default True). Passed as False:
                - internally to prevent double parsing.
                - by tests for efficiency.
    """

    _LEFT_SIDES = ["left", "both"]
    _RIGHT_SIDES = ["right", "both"]

    def __init__(
        self,
        start: Date | None = None,
        end: Date | None = None,
        side: str | None = None,
    ):
        side = side if side is not None else self.default_side
        if side not in self.valid_sides:
            raise ValueError(
                f"`side` must be in {self.valid_sides} although received as {side}."
            )
        self._side = side

        if start is None:
            start = self.default_start
        else:
            start = parse_date(start, "start")
            if self.bound_start is not None and start < self.bound_start:
                raise ValueError(self._bound_start_error_msg(start))

        if end is None:
            end = self.default_end
        else:
            end = parse_date(end, "end")
            if self.bound_end is not None and end > self.bound_end:
                raise ValueError(self._bound_end_error_msg(end))

        if start >= end:
            raise ValueError(
                "`start` must be earlier than `end` although `start` parsed as"
                f" '{start}' and `end` as '{end}'."
            )

        # Midnight in UTC for each trading day.
        _all_days = date_range(start, end, freq=self.day, tz=UTC)
        if _all_days.empty:
            raise errors.NoSessionsError(calendar_name=self.name, start=start, end=end)

        # `DatetimeIndex`s of standard opens/closes for each day.
        self._opens = _group_times(
            _all_days,
            self.open_times,
            self.tz,
            self.open_offset,
        )
        self._break_starts = _group_times(
            _all_days,
            self.break_start_times,
            self.tz,
        )
        self._break_ends = _group_times(
            _all_days,
            self.break_end_times,
            self.tz,
        )
        self._closes = _group_times(
            _all_days,
            self.close_times,
            self.tz,
            self.close_offset,
        )

        # Apply special offsets first
        self._calculate_and_overwrite_special_offsets(_all_days, start, end)

        # `Series`s mapping sessions with nonstandard opens/closes to
        # the open/close time.
        _special_opens = self._calculate_special_opens(start, end)
        _special_closes = self._calculate_special_closes(start, end)

        # Overwrite the special opens and closes on top of the standard ones.
        _overwrite_special_dates(_all_days, self._opens, _special_opens)
        _overwrite_special_dates(_all_days, self._closes, _special_closes)
        _remove_breaks_for_special_dates(
            _all_days,
            self._break_starts,
            _special_closes,
        )
        _remove_breaks_for_special_dates(
            _all_days,
            self._break_ends,
            _special_closes,
        )

        if self._break_starts is None:
            break_starts = None
        else:
            break_starts = self._break_starts.tz_localize(None)
        if self._break_ends is None:
            break_ends = None
        else:
            break_ends = self._break_ends.tz_localize(None)
        self.schedule = DataFrame(
            index=_all_days,
            data=OrderedDict(
                [
                    ("market_open", self._opens.tz_localize(None)),
                    ("break_start", break_starts),
                    ("break_end", break_ends),
                    ("market_close", self._closes.tz_localize(None)),
                ]
            ),
            dtype="datetime64[ns]",
        )

        self.market_opens_nanos = self.schedule.market_open.values.astype(np.int64)

        self.market_break_starts_nanos = self.schedule.break_start.values.astype(
            np.int64
        )

        self.market_break_ends_nanos = self.schedule.break_end.values.astype(np.int64)

        self.market_closes_nanos = self.schedule.market_close.values.astype(np.int64)

        _check_breaks_match(
            self.market_break_starts_nanos, self.market_break_ends_nanos
        )

        self.first_trading_session = _all_days[0]
        self.last_trading_session = _all_days[-1]

        self._late_opens = pd.DatetimeIndex(
            _special_opens.map(lambda x: self.minute_index_to_session_labels(x, "both"))
        )
        self._early_closes = pd.DatetimeIndex(
            _special_closes.map(
                lambda x: self.minute_index_to_session_labels(x, "both")
            )
        )

    # Methods and properties that define calendar and which should be
    # overriden or extended, if and as required, by subclass.

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def bound_start(self) -> pd.Timestamp | None:
        """Earliest date from which calendar can be constructed.

        Returns
        -------
        pd.Timestamp or None
            Earliest date from which calendar can be constructed. Must have
            tz as "UTC". None if no limit.

        Notes
        -----
        To impose a constraint on the earliest date from which a calendar
        can be constructed subclass should override this method and
        optionally override `_bound_start_error_msg`.
        """
        return None

    @property
    def bound_end(self) -> pd.Timestamp | None:
        """Latest date to which calendar can be constructed.

        Returns
        -------
        pd.Timestamp or None
            Latest date to which calendar can be constructed. Must have tz
            as "UTC". None if no limit.

        Notes
        -----
        To impose a constraint on the latest date to which a calendar can
        be constructed subclass should override this method and optionally
        override `_bound_end_error_msg`.
        """
        return None

    def _bound_start_error_msg(self, start: pd.Timestamp) -> str:
        """Return error message to handle `start` being out-of-bounds.

        See Also
        --------
        bound_start
        """
        return (
            f"The earliest date from which calendar {self.name} can be"
            f" evaluated is {self.bound_start}, although received `start` as"
            f" {start}."
        )

    def _bound_end_error_msg(self, end: pd.Timestamp) -> str:
        """Return error message to handle `end` being out-of-bounds.

        See Also
        --------
        bound_end
        """
        return (
            f"The latest date to which calendar {self.name} can be evaluated"
            f" is {self.bound_end}, although received `end` as {end}."
        )

    @property
    def default_start(self) -> pd.Timestamp:
        if self.bound_start is None:
            return GLOBAL_DEFAULT_START
        else:
            return max(GLOBAL_DEFAULT_START, self.bound_start)

    @property
    def default_end(self) -> pd.Timestamp:
        if self.bound_end is None:
            return GLOBAL_DEFAULT_END
        else:
            return min(GLOBAL_DEFAULT_END, self.bound_end)

    @abstractproperty
    def tz(self):
        raise NotImplementedError()

    @abstractproperty
    def open_times(self):
        """
        Returns a list of tuples of (start_date, open_time).  If the open
        time is constant throughout the calendar, use None for the start_date.
        """
        raise NotImplementedError()

    @property
    def break_start_times(self):
        """
        Returns a optional list of tuples of (start_date, break_start_time).
        If the break start time is constant throughout the calendar, use None
        for the start_date. If there is no break, return `None`.
        """
        return None

    @property
    def break_end_times(self):
        """
        Returns a optional list of tuples of (start_date, break_end_time).  If
        the break end time is constant throughout the calendar, use None for
        the start_date. If there is no break, return `None`.
        """
        return None

    @abstractproperty
    def close_times(self):
        """
        Returns a list of tuples of (start_date, close_time).  If the close
        time is constant throughout the calendar, use None for the start_date.
        """
        raise NotImplementedError()

    @property
    def weekmask(self):
        """
        String indicating the days of the week on which the market is open.

        Default is '1111100' (i.e., Monday-Friday).

        See Also
        --------
        numpy.busdaycalendar
        """
        return "1111100"

    @property
    def open_offset(self):
        return 0

    @property
    def close_offset(self):
        return 0

    @property
    def regular_holidays(self):
        """
        Returns
        -------
        pd.AbstractHolidayCalendar: a calendar containing the regular holidays
        for this calendar
        """
        return None

    @property
    def adhoc_holidays(self):
        """
        Returns
        -------
        list: A list of tz-naive timestamps representing unplanned closes.
        """
        return []

    @property
    def special_opens(self):
        """
        A list of special open times and corresponding HolidayCalendars.

        Returns
        -------
        list: List of (time, AbstractHolidayCalendar) tuples
        """
        return []

    @property
    def special_opens_adhoc(self):
        """
        Returns
        -------
        list: List of (time, DatetimeIndex) tuples that represent special
         closes that cannot be codified into rules.
        """
        return []

    @property
    def special_closes(self):
        """
        A list of special close times and corresponding HolidayCalendars.

        Returns
        -------
        list: List of (time, AbstractHolidayCalendar) tuples
        """
        return []

    @property
    def special_closes_adhoc(self):
        """
        Returns
        -------
        list: List of (time, DatetimeIndex) tuples that represent special
         closes that cannot be codified into rules.
        """
        return []

    @property
    def special_weekmasks(self):
        """
        Returns
        -------
        list: List of (date, date, str) tuples that represent special
         weekmasks that applies between dates.
        """
        return []

    @property
    def special_offsets(self):
        """
        Returns
        -------
        list: List of (timedelta, timedelta, timedelta, timedelta, AbstractHolidayCalendar) tuples
         that represent special open, break_start, break_end, close offsets
         and corresponding HolidayCalendars.
        """
        return []

    @property
    def special_offsets_adhoc(self):
        """
        Returns
        -------
        list: List of (timedelta, timedelta, timedelta, timedelta, DatetimeIndex) tuples
         that represent special open, break_start, break_end, close offsets
         and corresponding DatetimeIndexes.
        """
        return []

    # ------------------------------------------------------------------
    # -- NO method below this line should be overriden on a subclass! --
    # ------------------------------------------------------------------

    # Methods and properties that define calendar (continued...).

    @lazyval
    def day(self):
        if self.special_weekmasks:
            return MultipleWeekmaskCustomBusinessDay(
                holidays=self.adhoc_holidays,
                calendar=self.regular_holidays,
                weekmask=self.weekmask,
                weekmasks=self.special_weekmasks,
            )
        else:
            return CustomBusinessDay(
                holidays=self.adhoc_holidays,
                calendar=self.regular_holidays,
                weekmask=self.weekmask,
            )

    @property
    def valid_sides(self) -> list[str]:
        """List of valid `side` options."""
        if self.close_times == self.open_times:
            return ["left", "right"]
        else:
            return ["both", "left", "right", "neither"]

    @property
    def default_side(self) -> str:
        """Default `side` option."""
        if self.close_times == self.open_times:
            return "right"
        else:
            return "both"

    @property
    def side(self) -> str:
        """Side on which sessions are closed.

        Returns
        -------
        str
            "left" - Session open and break_start are trading minutes.
                Session close and break_end are not trading minutes.
            "right" - Session close and break_end are trading minutes,
                Session open and break_start are not tradng minutes.
            "both" - Session open, session close, break_start and
                break_end are all trading minutes.
            "neither" - Session open, session close, break_start and
                break_end are all not trading minutes.

        Notes
        -----
        Subclasses should NOT override this method.
        """
        return self._side

    # Properties covering all sessions.

    @property
    def all_sessions(self) -> pd.DatetimeIndex:
        return self.schedule.index

    @property
    def opens(self) -> pd.Series:
        return self.schedule.market_open

    @property
    def closes(self) -> pd.Series:
        return self.schedule.market_close

    @property
    def late_opens(self) -> pd.DatetimeIndex:
        return self._late_opens

    @property
    def early_closes(self) -> pd.DatetimeIndex:
        return self._early_closes

    @property
    def break_starts(self) -> pd.Series:
        return self.schedule.break_start

    @property
    def break_ends(self) -> pd.Series:
        return self.schedule.break_end

    @functools.lru_cache(maxsize=1)  # cache last request
    def _first_minute_nanos(self, side: str | None = None) -> np.ndarray:
        side = side if side is not None else self.side
        if side in self._LEFT_SIDES:
            return self.market_opens_nanos
        else:
            return one_minute_later(self.market_opens_nanos)

    @functools.lru_cache(maxsize=1)  # cache last request
    def _last_minute_nanos(self, side: str | None = None) -> np.ndarray:
        side = side if side is not None else self.side
        if side in self._RIGHT_SIDES:
            return self.market_closes_nanos
        else:
            return one_minute_earlier(self.market_closes_nanos)

    @functools.lru_cache(maxsize=1)  # cache last request
    def _last_am_minute_nanos(self, side: str | None = None) -> np.ndarray:
        side = side if side is not None else self.side
        if side in self._RIGHT_SIDES:
            return self.market_break_starts_nanos
        else:
            return one_minute_earlier(self.market_break_starts_nanos)

    @functools.lru_cache(maxsize=1)  # cache last request
    def _first_pm_minute_nanos(self, side: str | None = None) -> np.ndarray:
        side = side if side is not None else self.side
        if side in self._LEFT_SIDES:
            return self.market_break_ends_nanos
        else:
            return one_minute_later(self.market_break_ends_nanos)

    def _minutes_as_series(self, nanos: np.ndarray, name: str) -> pd.Series:
        """Convert trading minute nanos to pd.Series."""
        ser = pd.Series(pd.DatetimeIndex(nanos), index=self.all_sessions)
        ser.name = name
        return ser

    @property
    def all_first_minutes(self) -> pd.Series:
        """First trading minute of each session."""
        return self._minutes_as_series(self._first_minute_nanos(), "first_minutes")

    @property
    def all_last_minutes(self) -> pd.Series:
        """Last trading minute of each session."""
        return self._minutes_as_series(self._last_minute_nanos(), "last_minutes")

    @property
    def all_last_am_minutes(self) -> pd.Series:
        """Last am trading minute of each session."""
        return self._minutes_as_series(self._last_am_minute_nanos(), "last_am_minutes")

    @property
    def all_first_pm_minutes(self) -> pd.Series:
        """First pm trading minute of each session."""
        return self._minutes_as_series(
            self._first_pm_minute_nanos(), "first_pm_minutes"
        )

    # Properties covering all minutes.

    def _all_minutes(self, side: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            compute_all_minutes(
                self.market_opens_nanos,
                self.market_break_starts_nanos,
                self.market_break_ends_nanos,
                self.market_closes_nanos,
                side,
            ),
            tz="UTC",
        )

    @lazyval
    def all_minutes(self) -> pd.DatetimeIndex:
        """All trading minutes."""
        return self._all_minutes(self.side)

    @lazyval
    def all_minutes_nanos(self) -> np.ndarray:
        """All trading minutes as nanoseconds."""
        return self.all_minutes.values.astype(np.int64)

    # Calendar properties.

    @property
    def first_session(self) -> pd.Timestamp:
        return self.all_sessions[0]

    @property
    def last_session(self) -> pd.Timestamp:
        return self.all_sessions[-1]

    @property
    def first_session_open(self) -> pd.Timestamp:
        """Open time of calendar's first session."""
        return self.opens[0]

    @property
    def last_session_close(self) -> pd.Timestamp:
        """Close time of calendar's last session."""
        return self.closes[-1]

    @property
    def first_trading_minute(self) -> pd.Timestamp:
        """Calendar's first trading minute."""
        return pd.Timestamp(self.all_minutes_nanos[0], tz="UTC")

    @property
    def last_trading_minute(self) -> pd.Timestamp:
        """Calendar's last trading minute."""
        return pd.Timestamp(self.all_minutes_nanos[-1], tz="UTC")

    def has_breaks(
        self,
        start: Date | None = None,
        end: Date | None = None,
    ) -> bool:
        """Query if at least one session of a calendar has a break.

        Parameters
        ----------
        start : optional
            Limit query to sessions from `start`.

        end : optional
            Limit query to sessions through `end`.

        Returns
        -------
        bool
            True if any calendar session has a break, false otherwise.
        """
        start = None if start is None else parse_date(start, "start")
        end = None if end is None else parse_date(end, "end")
        return self.break_starts[start:end].notna().any()

    # Methods that interrogate a given session.

    def session_open(self, session_label: Session) -> pd.Timestamp:
        session_label = parse_session(self, session_label, "session_label")
        return self.schedule.at[session_label, "market_open"].tz_localize(UTC)

    def session_break_start(self, session_label: Session) -> pd.Timestamp:
        session_label = parse_session(self, session_label, "session_label")
        break_start = self.schedule.at[session_label, "break_start"]
        if not pd.isnull(break_start):
            # older versions of pandas need this guard
            break_start = break_start.tz_localize(UTC)

        return break_start

    def session_break_end(self, session_label: Session) -> pd.Timestamp:
        session_label = parse_session(self, session_label, "session_label")
        break_end = self.schedule.at[session_label, "break_end"]
        if not pd.isnull(break_end):
            # older versions of pandas need this guard
            break_end = break_end.tz_localize(UTC)

        return break_end

    def session_close(self, session_label: Session) -> pd.Timestamp:
        session_label = parse_session(self, session_label, "session_label")
        return self.schedule.at[session_label, "market_close"].tz_localize(UTC)

    def open_and_close_for_session(
        self, session_label: Session
    ) -> tuple(pd.Timestamp, pd.Timestamp):
        """
        Returns a tuple of timestamps of the open and close of the session
        represented by the given label.

        Parameters
        ----------
        session_label
            The session whose open and close are desired.

        Returns
        -------
        (Timestamp, Timestamp)
            The open and close for the given session.
        """
        session_label = parse_session(self, session_label, "session_label")
        return (
            self.session_open(session_label),
            self.session_close(session_label),
        )

    def break_start_and_end_for_session(
        self, session_label: Session
    ) -> tuple(pd.Timestamp, pd.Timestamp):
        """
        Returns a tuple of timestamps of the break start and end of the session
        represented by the given label.

        Parameters
        ----------
        session_label: pd.Timestamp
            The session whose break start and end are desired.

        Returns
        -------
        (Timestamp, Timestamp)
            The break start and end for the given session.
        """
        session_label = parse_session(self, session_label, "session_label")
        return (
            self.session_break_start(session_label),
            self.session_break_end(session_label),
        )

    def _get_session_minute_from_nanos(
        self, session: Session, nanos: np.ndarray, _parse: bool
    ) -> pd.Timestamp:
        if _parse:
            session = parse_session(self, session, "session")
        idx = self.all_sessions.get_loc(session)
        return pd.Timestamp(nanos[idx], tz="UTC")

    def session_first_minute(
        self, session: Session, _parse: bool = True
    ) -> pd.Timestamp:
        """Return first trading minute of a given session."""
        nanos = self._first_minute_nanos()
        return self._get_session_minute_from_nanos(session, nanos, _parse)

    def session_last_minute(
        self, session: Session, _parse: bool = True
    ) -> pd.Timestamp:
        """Return last trading minute of a given session."""
        nanos = self._last_minute_nanos()
        return self._get_session_minute_from_nanos(session, nanos, _parse)

    def session_last_am_minute(
        self, session: Session, _parse: bool = True
    ) -> pd.Timestamp | pd.NaT:  # Literal[pd.NaT] - when move to min 3.8
        """Return last trading minute of am subsession of a given session."""
        nanos = self._last_am_minute_nanos()
        return self._get_session_minute_from_nanos(session, nanos, _parse)

    def session_first_pm_minute(
        self, session: Session, _parse: bool = True
    ) -> pd.Timestamp | pd.NaT:  # Literal[pd.NaT] - when move to min 3.8
        """Return first trading minute of pm subsession of a given session."""
        nanos = self._first_pm_minute_nanos()
        return self._get_session_minute_from_nanos(session, nanos, _parse)

    def session_first_and_last_minute(
        self,
        session: Session,
        _parse: bool = True,
    ) -> tuple(pd.Timestamp, pd.Timestamp):
        """Return first and last trading minutes of a given session."""
        if _parse:
            session = parse_session(self, session, "session")
        idx = self.all_sessions.get_loc(session)
        first = pd.Timestamp(self._first_minute_nanos()[idx], tz="UTC")
        last = pd.Timestamp(self._last_minute_nanos()[idx], tz="UTC")
        return (first, last)

    def session_has_break(self, session: Session) -> bool:
        """Query if a given session has a break.

        Parameters
        ----------
        session :
            Session to query.

        Returns
        -------
        bool
            True if `session` has a break, false otherwise.
        """
        session = parse_session(self, session, "session")
        return pd.notna(self.session_break_start(session))

    def next_session_label(
        self, session_label: Session, _parse: bool = True
    ) -> pd.Timestamp:
        """Return session that immediately follows a given session.

        Parameters
        ----------
        session_label
            Session whose next session is desired.

        Raises
        ------
        ValueError
            If `session_label` is the last calendar session.

        See Also
        --------
        date_to_session_label
        """
        if _parse:
            session_label = parse_session(self, session_label, "session_label")
        idx = self.schedule.index.get_loc(session_label)
        try:
            return self.schedule.index[idx + 1]
        except IndexError as err:
            if idx == len(self.schedule.index) - 1:
                raise ValueError(
                    "There is no next session as this is the end"
                    " of the exchange calendar."
                ) from err
            else:
                raise

    def previous_session_label(
        self, session_label: Session, _parse: bool = True
    ) -> pd.Timestamp:
        """Return session that immediately preceeds a given session.

        Parameters
        ----------
        session_label
            Session whose previous session is desired.

        Raises
        ------
        ValueError
            If `session_label` is the first calendar session.

        See Also
        --------
        date_to_session_label
        """
        if _parse:
            session_label = parse_session(self, session_label, "session_label")
        idx = self.schedule.index.get_loc(session_label)
        if idx == 0:
            raise ValueError(
                "There is no previous session as this is the"
                " beginning of the exchange calendar."
            )
        return self.schedule.index[idx - 1]

    def minutes_for_session(
        self, session_label: Session, _parse: bool = True
    ) -> pd.DatetimeIndex:
        """Return trading minutes corresponding to a given session.

        Parameters
        ----------
        session_label
            Session for which require trading minutes.

        Returns
        -------
        pd.DateTimeIndex
            Trading minutes for `session`.
        """
        if _parse:
            session_label = parse_session(self, session_label, "session_label")
        first, last = self.session_first_and_last_minute(session_label, _parse=False)
        return self.minutes_in_range(start_minute=first, end_minute=last)

    # Methods that interrogate a date.

    def is_session(self, dt):
        """
        Given a dt, returns whether it's a valid session label.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt that is being tested.

        Returns
        -------
        bool
            Whether the given dt is a valid session label.
        """
        return dt in self.schedule.index

    def date_to_session_label(
        self,
        date: Date,
        direction: str = "none",
        _parse: bool = True,
    ) -> pd.Timestamp:
        """Return a session label corresponding to a given date.

        Parameters
        ----------
        date
            Date for which require session label. Can be a date that does not
            represent an actual session (see `direction`).

        direction : default: "none"
            Defines behaviour if `date` does not represent a session:
                "next" - return first session label following `date`.
                "previous" - return first session label prior to `date`.
                "none" - raise ValueError.

        Returns
        -------
        pd.Timestamp (midnight UTC)
            Label of the corresponding session.

        See Also
        --------
        next_session_label
        previous_session_label
        """
        if _parse:
            date = parse_date(date, "date")
        if self.is_session(date):
            return date
        elif direction in ["next", "previous"]:
            if direction == "previous" and date < self.first_session:
                raise ValueError(
                    "Cannot get a session label prior to the first calendar"
                    f" session ('{self.first_session}'). Consider passing"
                    f" `direction` as 'next'."
                )
            if direction == "next" and date > self.last_session:
                raise ValueError(
                    "Cannot get a session label later than the last calendar"
                    f" session ('{self.last_session}'). Consider passing"
                    f" `direction` as 'previous'."
                )
            idx = self.all_sessions.values.astype(np.int64).searchsorted(date.value)
            if direction == "previous":
                idx -= 1
            return self.all_sessions[idx]
        elif direction == "none":
            raise ValueError(
                f"`date` '{date}' is not a session label. Consider passing"
                " a `direction`."
            )
        else:
            raise ValueError(
                f"'{direction}' is not a valid `direction`. Valid `direction`"
                ' values are "next", "previous" and "none".'
            )

    # Methods that interrogate a given minute (trading or non-trading).

    def is_trading_minute(self, minute: Minute, _parse: bool = True) -> bool:
        """Query if a given minute is a trading minute.

        Minutes during breaks are not considered trading minutes.

        Note: `self.side` determines whether exchange will be considered
        open or closed on session open, session close, break start and
        break end.

        Parameters
        ----------
        minute
            Minute being queried.

        Returns
        -------
        bool
            Boolean indicting if `minute` is a trading minute.

        See Also
        --------
        is_open_on_minute
        """
        if _parse:
            minute = parse_timestamp(
                minute, "minute", raise_oob=True, calendar=self
            ).value
        else:
            minute = minute.value

        idx = self.all_minutes_nanos.searchsorted(minute)
        numpy_bool = minute == self.all_minutes_nanos[idx]
        return bool(numpy_bool)

    def is_break_minute(self, minute: Minute, _parse: bool = True) -> bool:
        """Query if a given minute is within a break.

        Note: `self.side` determines whether either, both or one of break
        start and break end are treated as break minutes.

        Parameters
        ----------
        minute
            Minute being queried.

        Returns
        -------
        bool
            Boolean indicting if `minute` is a break minute.
        """
        if _parse:
            minute = parse_timestamp(
                minute, "minute", raise_oob=True, calendar=self
            ).value
        else:
            minute = minute.value

        session_idx = np.searchsorted(self._first_minute_nanos(), minute) - 1
        break_start = self._last_am_minute_nanos()[session_idx]
        break_end = self._first_pm_minute_nanos()[session_idx]
        # NaT comparisions evalute as False
        numpy_bool = break_start < minute < break_end
        return bool(numpy_bool)

    def is_open_on_minute(
        self, dt: Minute, ignore_breaks: bool = False, _parse: bool = True
    ) -> bool:
        """Query if exchange is open on a given minute.

        Note: `self.side` determines whether exchange will be considered
        open or closed on session open, session close, break start and
        break end.

        Parameters
        ----------
        dt
            Minute being queried.

        ignore_breaks
            Should exchange be considered open during any break?
                True - treat exchange as open during any break.
                False - treat exchange as closed during any break.

        Returns
        -------
        bool
            Boolean indicting if exchange is open on `dt`.

        See Also
        --------
        is_trading_minute
        """
        if _parse:
            minute = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        else:
            minute = dt

        is_trading_minute = self.is_trading_minute(minute, _parse=_parse)
        if is_trading_minute or not ignore_breaks:
            return is_trading_minute
        else:
            # not a trading minute although should return True if in break
            return self.is_break_minute(minute, _parse=_parse)

    def next_open(self, dt: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return next open that follows a given minute.

        If `dt` is a session open, the next session's open will be
        returned.

        Parameters
        ----------
        dt
            Minute for which to get the next open.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the next open.
        """
        if _parse:
            dt = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        try:
            idx = next_divider_idx(self.market_opens_nanos, dt.value)
        except IndexError:
            if dt.tz_convert(None) >= self.opens[-1]:
                raise ValueError(
                    "Minute cannot be the last open or later (received `dt`"
                    f" parsed as '{dt}'.)"
                ) from None
            else:
                raise

        return pd.Timestamp(self.market_opens_nanos[idx], tz=UTC)

    def next_close(self, dt: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return next close that follows a given minute.

        If `dt` is a session close, the next session's close will be
        returned.

        Parameters
        ----------
        dt
            Minute for which to get the next close.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the next close.
        """
        if _parse:
            dt = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        try:
            idx = next_divider_idx(self.market_closes_nanos, dt.value)
        except IndexError:
            if dt.tz_convert(None) == self.closes[-1]:
                raise ValueError(
                    "Minute cannot be the last close (received `dt` parsed as"
                    f" '{dt}'.)"
                ) from None
            else:
                raise
        return pd.Timestamp(self.market_closes_nanos[idx], tz=UTC)

    def previous_open(self, dt: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return previous open that preceeds a given minute.

        If `dt` is a session open, the previous session's open will be
        returned.

        Parameters
        ----------
        dt
            Minute for which to get the previous open.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the previous open.
        """
        if _parse:
            dt = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        try:
            idx = previous_divider_idx(self.market_opens_nanos, dt.value)
        except ValueError:
            if dt.tz_convert(None) == self.opens[0]:
                raise ValueError(
                    "Minute cannot be the first open (received `dt` parsed as"
                    f" '{dt}'.)"
                ) from None
            else:
                raise

        return pd.Timestamp(self.market_opens_nanos[idx], tz=UTC)

    def previous_close(self, dt: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return previous close that preceeds a given minute.

        If `dt` is a session close, the previous session's close will be
        returned.

        Parameters
        ----------
        dt
            Minute for which to get the previous close.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the previous close.
        """
        if _parse:
            dt = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        try:
            idx = previous_divider_idx(self.market_closes_nanos, dt.value)
        except ValueError:
            if dt.tz_convert(None) <= self.closes[0]:
                raise ValueError(
                    "Minute cannot be the first close or earlier (received"
                    f" `dt` parsed as '{dt}'.)"
                ) from None
            else:
                raise

        return pd.Timestamp(self.market_closes_nanos[idx], tz=UTC)

    def next_minute(self, dt: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return trading minute that immediately follows a given minute.

        Parameters
        ----------
        dt
            Minute for which to get next trading minute. Minute can be a
            trading or a non-trading minute.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the next minute.
        """
        if _parse:
            dt = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        try:
            idx = next_divider_idx(self.all_minutes_nanos, dt.value)
        except IndexError:
            # dt > last_trading_minute handled via parsing
            if dt == self.last_trading_minute:
                raise ValueError(
                    "Minute cannot be the last trading minute or later"
                    f" (received `dt` parsed as '{dt}'.)"
                ) from None
        return self.all_minutes[idx]

    def previous_minute(self, dt: Minute, _parse: bool = True) -> pd.Timestamp:
        """Return trading minute that immediately preceeds a given minute.

        Parameters
        ----------
        dt
            Minute for which to get previous trading minute. Minute can be
            a trading or a non-trading minute.

        Returns
        -------
        pd.Timestamp
            UTC timestamp of the previous minute.
        """
        if _parse:
            dt = parse_timestamp(dt, "dt", raise_oob=True, calendar=self)
        try:
            idx = previous_divider_idx(self.all_minutes_nanos, dt.value)
        except ValueError:
            # dt < first_trading_minute handled via parsing
            if dt == self.first_trading_minute:
                raise ValueError(
                    "Minute cannot be the first trading minute or earlier"
                    f" (received `dt` parsed as '{dt}'.)"
                ) from None
        return self.all_minutes[idx]

    def minute_to_session_label(
        self,
        dt: Minute,
        direction: str = "next",
        _parse: bool = True,
    ) -> pd.Timestamp:
        """Get session corresponding with a trading or break minute.

        Parameters
        ----------
        dt
            Minute for which require corresponding session.

        direction
            How to resolve session in event that `dt` is not a trading
            or break minute:
                "next" (default) - return first session subsequent to
                    `dt`.
                "previous" - return first session prior to `dt`.
                "none" - raise ValueError.

        Returns
        -------
        pd.Timestamp (midnight UTC)
            Corresponding session label.

        Raises
        ------
        ValueError
            If `dt` is not a trading minute and `direction` is "none".
        """
        if _parse:
            minute = parse_timestamp(dt, "dt").value
        else:
            minute = dt.value

        if minute < self.first_trading_minute.value:
            # Resolve call here.
            if direction == "next":
                return self.first_session
            else:
                raise ValueError(
                    "Received `minute` as '{0}' although this is earlier than the"
                    " calendar's first trading minute ({1}). Consider passing"
                    " `direction` as 'next' to get first session label.".format(
                        pd.Timestamp(minute, tz="UTC"), self.first_trading_minute
                    )
                )

        if minute > self.last_trading_minute.value:
            # Resolve call here.
            if direction == "previous":
                return self.last_session
            else:
                raise ValueError(
                    "Received `minute` as '{0}' although this is later than the"
                    " calendar's last trading minute ({1}). Consider passing"
                    " `direction` as 'previous' to get last session label.".format(
                        pd.Timestamp(minute, tz="UTC"), self.last_trading_minute
                    )
                )

        idx = np.searchsorted(self._last_minute_nanos(), minute)
        current_or_next_session = self.schedule.index[idx]

        if direction == "next":
            return current_or_next_session
        elif direction == "previous":
            if not self.is_open_on_minute(minute, ignore_breaks=True):
                return self.schedule.index[idx - 1]
        elif direction == "none":
            if not self.is_open_on_minute(minute, ignore_breaks=True):
                # if the exchange is closed, blow up
                raise ValueError(
                    "Received `minute` as '{0}' although this is not an exchange"
                    " minute. Consider passing `direction` as 'next' or"
                    " 'previous'.".format(pd.Timestamp(minute, tz="UTC"))
                )
        else:
            # invalid direction
            raise ValueError("Invalid direction parameter: " "{0}".format(direction))

        return current_or_next_session

    # Methods that evaluate or interrogate a range of minutes.

    def minutes_in_range(
        self, start_minute: Minute, end_minute: Minute, _parse: bool = True
    ) -> pd.DatetimeIndex:
        """Return all trading minutes between given minutes.

        Parameters
        ----------
        start_minute
            Minute representing start of desired range. Can be a trading
            minute or non-trading minute.

        end_minute
            Minute representing end of desired range. Can be a trading
            minute or non-trading minute.
        """
        if _parse:
            start_minute = parse_timestamp(
                start_minute, "start_minute", raise_oob=True, calendar=self
            )
            end_minute = parse_timestamp(
                end_minute, "end_minute", raise_oob=True, calendar=self
            )

        start_idx = searchsorted(self.all_minutes_nanos, start_minute.value)
        end_idx = searchsorted(self.all_minutes_nanos, end_minute.value)

        if end_minute.value == self.all_minutes_nanos[end_idx]:
            # if the end minute is a market minute, increase by 1
            end_idx += 1

        return self.all_minutes[start_idx:end_idx]

    def minutes_window(self, start_dt: pd.Timestamp, count: int):
        start_dt_nanos = start_dt.value
        start_idx = self.all_minutes_nanos.searchsorted(start_dt_nanos)

        # searchsorted finds the index of the minute **on or after** start_dt.
        # If the latter, push back to the prior minute.
        if self.all_minutes_nanos[start_idx] != start_dt_nanos:
            start_idx -= 1

        if start_idx < 0 or start_idx >= len(self.all_minutes_nanos):
            raise KeyError("Can't start minute window at {}".format(start_dt))

        end_idx = start_idx + count

        if start_idx > end_idx:
            return self.all_minutes[(end_idx + 1) : (start_idx + 1)]
        else:
            return self.all_minutes[start_idx:end_idx]

    def minute_index_to_session_labels(
        self,
        index: pd.DatetimeIndex,
        side: str | None = None,
    ) -> pd.DatetimeIndex:
        """Return session labels corresponding to multiple market minutes.

        For the purpose of this method market minutes are considered as:
            - Trading minutes as determined by `side` (default `self.side`).
            - All minutes of any breaks.

        Parameters
        ----------
        index
            Sorted DatetimeIndex representing market minutes for which to get
            corresponding session labels.

        side : default: as `self.side`
            Override `self.side` to define which side of sessions should be
            considered as market minutes for the purpose of this call.

        Returns
        -------
        pd.DatetimeIndex (indices UTC midnight)
            Session labels corresponding to market minutes given in `index`.

        Raises
        ------
        ValueError
            If any indice of `index` is not a market minute.
        """
        if not index.is_monotonic_increasing:
            raise ValueError(
                "Non-ordered index passed to minute_index_to_session_labels."
            )
        # Find the indices of the previous first session minute and the next
        # last session minute for each minute.
        index_nanos = index.values.astype(np.int64)
        first_min_nanos = self._first_minute_nanos(side)
        last_min_nanos = self._last_minute_nanos(side)
        prev_first_mins_idxs = (
            first_min_nanos.searchsorted(index_nanos, side="right") - 1
        )
        next_last_mins_idxs = last_min_nanos.searchsorted(index_nanos, side="left")

        # If they don't match, the minute is outside the trading day. Barf.
        mismatches = prev_first_mins_idxs != next_last_mins_idxs
        if mismatches.any():
            # Show the first bad minute in the error message.
            bad_ix = np.flatnonzero(mismatches)[0]
            example = index[bad_ix]

            prev_session_idx = prev_first_mins_idxs[bad_ix]
            prev_first_min = pd.Timestamp(first_min_nanos[prev_session_idx], tz="UTC")
            prev_last_min = pd.Timestamp(last_min_nanos[prev_session_idx], tz="UTC")
            next_first_min = pd.Timestamp(
                first_min_nanos[prev_session_idx + 1], tz="UTC"
            )
            next_last_min = pd.Timestamp(last_min_nanos[prev_session_idx + 1], tz="UTC")

            raise ValueError(
                f"{mismatches.sum()} non-trading minutes in"
                f" minute_index_to_session_labels:\nFirst Bad Minute: {example}\n"
                f"Previous Session: {prev_first_min} -> {prev_last_min}\n"
                f"Next Session: {next_first_min} -> {next_last_min}"
            )

        return self.schedule.index[prev_first_mins_idxs]

    # Methods that evaluate or interrogate a range of sessions.

    def sessions_in_range(
        self, start_session_label: Date, end_session_label: Date, _parse: bool = True
    ) -> pd.DatetimeIndex:
        """Return sessions within a given range.

        Parameters
        ----------
        start_session_label
            Date from which (and inclusive of) to include sessions in
            range.

        end_session_label
            Date through which (and inclusive of) to include sessions in
            range.

        Returns
        -------
        pd.DatetimeIndex
            Sessions from `start_session_label` through `end_session_label`.
        """
        if _parse:
            start_session_label = parse_date(
                start_session_label,
                "start_session_label",
                raise_oob=True,
                calendar=self,
            )
            end_session_label = parse_date(
                end_session_label,
                "end_session_label",
                raise_oob=True,
                calendar=self,
            )
        indexer = self.all_sessions.slice_indexer(
            start_session_label, end_session_label
        )
        return self.all_sessions[indexer]

    def sessions_window(
        self, session_label: Session, count: int, _parse: bool = True
    ) -> pd.DatetimeIndex:
        """Return block of given size of consecutive sessions.

        Parameters
        ----------
        session_label
            Session representing the first (if `count` positive) or last
            (if `count` negative) session of session block.

        count
            Number of sessions to include in block in addition to
                `session_label` (i.e. 0 will return block of length 1 with
                `session_label` as only value).
            Positive to return block of sessions from `session_label`
            Negative to return block of sessions to `session_label`.
        """
        if _parse:
            session_label = parse_session(self, session_label, "session_label")
        start_idx = self.all_sessions.get_loc(session_label)
        end_idx = start_idx + count
        if end_idx < 0:
            raise ValueError(
                f"Sessions window cannot begin before the first calendar session"
                f" ({self.first_session}). `count` cannot be lower than"
                f" {count - end_idx} for `session` '{session_label}'."
            )
        elif end_idx >= len(self.all_sessions):
            raise ValueError(
                f"Sessions window cannot end after the last calendar session"
                f" ({self.last_session}). `count` cannot be higher than"
                f" {count - (end_idx - len(self.all_sessions) + 1)} for"
                f" `session` '{session_label}'."
            )
        return self.all_sessions[min(start_idx, end_idx) : max(start_idx, end_idx) + 1]

    def session_distance(
        self,
        start_session_label: Date,
        end_session_label: Date,
        _parse: bool = True,
    ) -> int:
        """Return the number of sessions between two dates.

        Parameters
        ----------
        start_session_label
            Date from which (inclusive) to evaluate distance.

        end_session_label
            Date through which (inclusive) to evaluate distance.

        Returns
        -------
        int
            Number of sessions between `start_session_label` and
            `end_session_label` (both inclusive). If `start_session` is
            after `end_session` then return will be negated.
        """
        # out-of-bounds dates handled via parsing
        if _parse:
            start_session = parse_date(
                start_session_label, "start_session_label", True, self
            )
            end_session = parse_date(end_session_label, "end_session_label", True, self)
        else:
            start_session, end_session = start_session_label, end_session_label

        negate = end_session < start_session
        if negate:
            start_session, end_session = end_session, start_session
        start_idx = self.all_sessions.searchsorted(start_session)
        end_idx = self.all_sessions.searchsorted(end_session, side="right")
        return start_idx - end_idx if negate else end_idx - start_idx

    def minutes_for_sessions_in_range(
        self,
        start_session_label: Session,
        end_session_label: Session,
        _parse: bool = True,
    ) -> pd.DatetimeIndex:
        """Return trading minutes over a range of sessions.

        Parameters
        ----------
        start_session_label
            First session of range.

        end_session_label
            Last session of range.

        Returns
        -------
        pd.DatetimeIndex
            Trading minutes over range.
        """
        if _parse:
            start_session_label = parse_session(
                self, start_session_label, "start_session_label"
            )
            end_session_label = parse_session(
                self, end_session_label, "end_session_label"
            )
        first_minute = self.session_first_minute(start_session_label)
        last_minute = self.session_last_minute(end_session_label)
        return self.minutes_in_range(first_minute, last_minute)

    def session_opens_in_range(self, start_session_label, end_session_label):
        return self.schedule.loc[
            start_session_label:end_session_label,
            "market_open",
        ].dt.tz_localize(UTC)

    def session_closes_in_range(self, start_session_label, end_session_label):
        return self.schedule.loc[
            start_session_label:end_session_label,
            "market_close",
        ].dt.tz_localize(UTC)

    @lazyval
    def _minutes_per_session(self):
        close_to_open_diff = self.schedule.market_close - self.schedule.market_open
        break_diff = (self.schedule.break_end - self.schedule.break_start).fillna(
            pd.Timedelta(seconds=0)
        )
        diff = (close_to_open_diff - break_diff).astype("timedelta64[m]")
        return diff + 1

    def minutes_count_for_sessions_in_range(
        self, start_session: pd.Timestamp, end_session: pd.Timestamp
    ) -> int:
        """
        Parameters
        ----------
        start_session: pd.Timestamp
            The first session.

        end_session: pd.Timestamp
            The last session.

        Returns
        -------
        int: The total number of minutes for the contiguous chunk of sessions.
             between start_session and end_session, inclusive.
        """
        return int(self._minutes_per_session[start_session:end_session].sum())

    # Internal methods called by constructor.

    def _special_dates(self, calendars, ad_hoc_dates, start_date, end_date):
        """
        Compute a Series of times associated with special dates.

        Parameters
        ----------
        holiday_calendars : list[(datetime.time, HolidayCalendar)]
            Pairs of time and calendar describing when that time occurs. These
            are used to describe regularly-scheduled late opens or early
            closes.
        ad_hoc_dates : list[(datetime.time, list[pd.Timestamp])]
            Pairs of time and list of dates associated with the given times.
            These are used to describe late opens or early closes that occurred
            for unscheduled or otherwise irregular reasons.
        start_date : pd.Timestamp
            Start of the range for which we should calculate special dates.
        end_date : pd.Timestamp
            End of the range for which we should calculate special dates.

        Returns
        -------
        special_dates : pd.Series
            Series mapping trading sessions with special opens/closes to the
            special open/close for that session.
        """
        # List of Series for regularly-scheduled times.
        regular = [
            scheduled_special_times(
                calendar,
                start_date,
                end_date,
                time_,
                self.tz,
            )
            for time_, calendar in calendars
        ]

        # List of Series for ad-hoc times.
        ad_hoc = [
            pd.Series(
                index=pd.to_datetime(datetimes, utc=True),
                data=days_at_time(datetimes, time_, self.tz),
            )
            for time_, datetimes in ad_hoc_dates
        ]

        merged = regular + ad_hoc
        if not merged:
            # Concat barfs if the input has length 0.
            return pd.Series([], dtype="datetime64[ns, UTC]")

        result = pd.concat(merged).sort_index()
        result = result.loc[(result >= start_date) & (result <= end_date)]
        # exclude any special date that conincides with a holiday
        adhoc_holidays = pd.DatetimeIndex(self.adhoc_holidays, tz="UTC")
        result = result[~result.index.isin(adhoc_holidays)]
        reg_holidays = self.regular_holidays.holidays(
            start_date.tz_convert(None), end_date.tz_convert(None)
        )
        if not reg_holidays.empty:
            result = result[~result.index.isin(reg_holidays.tz_localize("UTC"))]
        return result

    def _calculate_special_opens(self, start, end):
        return self._special_dates(
            self.special_opens,
            self.special_opens_adhoc,
            start,
            end,
        )

    def _calculate_special_closes(self, start, end):
        return self._special_dates(
            self.special_closes,
            self.special_closes_adhoc,
            start,
            end,
        )

    def _overwrite_special_offsets(
        self,
        session_labels,
        opens_or_closes,
        calendars,
        ad_hoc_dates,
        start_date,
        end_date,
        strict=False,
    ):
        # Short circuit when nothing to apply.
        if opens_or_closes is None or not len(opens_or_closes):
            return

        len_m, len_oc = len(session_labels), len(opens_or_closes)
        if len_m != len_oc:
            raise ValueError(
                "Found misaligned dates while building calendar.\n"
                "Expected session_labels to be the same length as "
                "open_or_closes but,\n"
                "len(session_labels)=%d, len(open_or_closes)=%d" % (len_m, len_oc)
            )

        regular = []
        for offset, calendar in calendars:
            days = calendar.holidays(start_date, end_date)
            series = pd.Series(
                index=pd.DatetimeIndex(days, tz=UTC),
                data=offset,
            )
            regular.append(series)

        ad_hoc = []
        for offset, datetimes in ad_hoc_dates:
            series = pd.Series(
                index=pd.to_datetime(datetimes, utc=True),
                data=offset,
            )
            ad_hoc.append(series)

        merged = regular + ad_hoc
        if not merged:
            return pd.Series([], dtype="timedelta64[ns]")

        result = pd.concat(merged).sort_index()
        offsets = result.loc[(result.index >= start_date) & (result.index <= end_date)]

        # Find the array indices corresponding to each special date.
        indexer = session_labels.get_indexer(offsets.index)

        # -1 indicates that no corresponding entry was found.  If any -1s are
        # present, then we have special dates that doesn't correspond to any
        # trading day.
        if -1 in indexer and strict:
            bad_dates = list(offsets.index[indexer == -1])
            raise ValueError("Special dates %s are not trading days." % bad_dates)

        special_opens_or_closes = opens_or_closes[indexer] + offsets

        # Short circuit when nothing to apply.
        if not len(special_opens_or_closes):
            return

        # NOTE: This is a slightly dirty hack.  We're in-place overwriting the
        # internal data of an Index, which is conceptually immutable.  Since we're
        # maintaining sorting, this should be ok, but this is a good place to
        # sanity check if things start going haywire with calendar computations.
        opens_or_closes.values[indexer] = special_opens_or_closes.values

    def _calculate_and_overwrite_special_offsets(self, session_labels, start, end):
        _special_offsets = self.special_offsets
        _special_offsets_adhoc = self.special_offsets_adhoc

        _special_open_offsets = [
            (t[0], t[-1]) for t in _special_offsets if t[0] is not None
        ]
        _special_open_offsets_adhoc = [
            (t[0], t[-1]) for t in _special_offsets_adhoc if t[0] is not None
        ]
        _special_break_start_offsets = [
            (t[1], t[-1]) for t in _special_offsets if t[1] is not None
        ]
        _special_break_start_offsets_adhoc = [
            (t[1], t[-1]) for t in _special_offsets_adhoc if t[1] is not None
        ]
        _special_break_end_offsets = [
            (t[2], t[-1]) for t in _special_offsets if t[2] is not None
        ]
        _special_break_end_offsets_adhoc = [
            (t[2], t[-1]) for t in _special_offsets_adhoc if t[2] is not None
        ]
        _special_close_offsets = [
            (t[3], t[-1]) for t in _special_offsets if t[3] is not None
        ]
        _special_close_offsets_adhoc = [
            (t[3], t[-1]) for t in _special_offsets_adhoc if t[3] is not None
        ]

        self._overwrite_special_offsets(
            session_labels,
            self._opens,
            _special_open_offsets,
            _special_open_offsets_adhoc,
            start,
            end,
        )
        self._overwrite_special_offsets(
            session_labels,
            self._break_starts,
            _special_break_start_offsets,
            _special_break_start_offsets_adhoc,
            start,
            end,
        )
        self._overwrite_special_offsets(
            session_labels,
            self._break_ends,
            _special_break_end_offsets,
            _special_break_end_offsets_adhoc,
            start,
            end,
        )
        self._overwrite_special_offsets(
            session_labels,
            self._closes,
            _special_close_offsets,
            _special_close_offsets_adhoc,
            start,
            end,
        )

    # Deprecated methods to be removed in release 4.0.

    @deprecate(renamed=False)
    def execution_time_from_open(self, open_dates):
        return open_dates

    @deprecate(renamed=False)
    def execution_time_from_close(self, close_dates):
        return close_dates

    @deprecate(alt_method="minutes_for_session", renamed=False)
    def execution_minutes_for_session(self, session_label: Session) -> pd.DatetimeIndex:
        """
        Given a session label, return the execution minutes for that session.

        Parameters
        ----------
        session_label
            A session label whose session's minutes are desired.

        Returns
        -------
        pd.DateTimeIndex
            All the execution minutes for the given session.
        """
        return self.minutes_for_session(session_label)

    @deprecate(alt_method="minutes_for_sessions_in_range", renamed=False)
    def execution_minutes_for_sessions_in_range(self, start, stop):
        minutes = self.execution_minutes_for_session
        return pd.DatetimeIndex(
            np.concatenate(
                [minutes(session) for session in self.sessions_in_range(start, stop)]
            ),
            tz=UTC,
        )


def _check_breaks_match(market_break_starts_nanos, market_break_ends_nanos):
    """Checks that market_break_starts_nanos and market_break_ends_nanos

    Parameters
    ----------
    market_break_starts_nanos : np.ndarray
    market_break_ends_nanos : np.ndarray
    """
    nats_match = np.equal(
        NP_NAT == market_break_starts_nanos, NP_NAT == market_break_ends_nanos
    )
    if not nats_match.all():
        raise ValueError(
            """
            Mismatched market breaks
            Break starts:
            {0}
            Break ends:
            {1}
            """.format(
                market_break_starts_nanos[~nats_match],
                market_break_ends_nanos[~nats_match],
            )
        )


def scheduled_special_times(calendar, start, end, time, tz):
    """
    Returns a Series mapping each holiday (as a UTC midnight Timestamp)
    in ``calendar`` between ``start`` and ``end`` to that session at
    ``time`` (as a UTC Timestamp).
    """
    days = calendar.holidays(start, end)
    return pd.Series(
        index=pd.DatetimeIndex(days, tz=UTC),
        data=days_at_time(days, time, tz=tz),
    )


def _overwrite_special_dates(session_labels, opens_or_closes, special_opens_or_closes):
    """
    Overwrite dates in open_or_closes with corresponding dates in
    special_opens_or_closes, using session_labels for alignment.
    """
    # Short circuit when nothing to apply.
    if not len(special_opens_or_closes):
        return

    len_m, len_oc = len(session_labels), len(opens_or_closes)
    if len_m != len_oc:
        raise ValueError(
            "Found misaligned dates while building calendar.\n"
            "Expected session_labels to be the same length as "
            "open_or_closes but,\n"
            "len(session_labels)=%d, len(open_or_closes)=%d" % (len_m, len_oc)
        )

    # Find the array indices corresponding to each special date.
    indexer = session_labels.get_indexer(special_opens_or_closes.index)

    # -1 indicates that no corresponding entry was found.  If any -1s are
    # present, then we have special dates that doesn't correspond to any
    # trading day.
    if -1 in indexer:
        bad_dates = list(special_opens_or_closes[indexer == -1])
        raise ValueError("Special dates %s are not trading days." % bad_dates)

    # NOTE: This is a slightly dirty hack.  We're in-place overwriting the
    # internal data of an Index, which is conceptually immutable.  Since we're
    # maintaining sorting, this should be ok, but this is a good place to
    # sanity check if things start going haywire with calendar computations.
    opens_or_closes.values[indexer] = special_opens_or_closes.values


def _remove_breaks_for_special_dates(
    session_labels, break_start_or_end, special_opens_or_closes
):
    """
    Overwrite breaks in break_start_or_end with corresponding dates in
    special_opens_or_closes, using session_labels for alignment.
    """
    # Short circuit when we have no breaks
    if break_start_or_end is None:
        return

    # Short circuit when nothing to apply.
    if not len(special_opens_or_closes):
        return

    len_m, len_oc = len(session_labels), len(break_start_or_end)
    if len_m != len_oc:
        raise ValueError(
            "Found misaligned dates while building calendar.\n"
            "Expected session_labels to be the same length as break_starts,\n"
            "but len(session_labels)=%d, len(break_start_or_end)=%d" % (len_m, len_oc)
        )

    # Find the array indices corresponding to each special date.
    indexer = session_labels.get_indexer(special_opens_or_closes.index)

    # -1 indicates that no corresponding entry was found.  If any -1s are
    # present, then we have special dates that doesn't correspond to any
    # trading day.
    if -1 in indexer:
        bad_dates = list(special_opens_or_closes[indexer == -1])
        raise ValueError("Special dates %s are not trading days." % bad_dates)

    # NOTE: This is a slightly dirty hack.  We're in-place overwriting the
    # internal data of an Index, which is conceptually immutable.  Since we're
    # maintaining sorting, this should be ok, but this is a good place to
    # sanity check if things start going haywire with calendar computations.
    break_start_or_end.values[indexer] = NP_NAT


class HolidayCalendar(AbstractHolidayCalendar):
    def __init__(self, rules):
        super(HolidayCalendar, self).__init__(rules=rules)

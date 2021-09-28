from __future__ import annotations
import typing
import datetime
import contextlib

import numpy as np
import pandas as pd

from exchange_calendars import errors

if typing.TYPE_CHECKING:
    from exchange_calendars import ExchangeCalendar

NANOSECONDS_PER_MINUTE = int(6e10)

NP_NAT = pd.NaT.value

# Use Date type where input does not need to represent an actual session
# and will be parsed by parse_date
Date = typing.Union[pd.Timestamp, str, int, float, datetime.datetime]
# Use Session type where input should represent an actual session and will
# be parsed by parse_session
Session = Date
# Use Minute type where input does not need to represent an actual trading
# minute and will be parsed by parse_timestamp
Minute = typing.Union[pd.Timestamp, str, int, float, datetime.datetime]


def next_divider_idx(dividers: np.ndarray, minute_val: int) -> int:

    divider_idx = np.searchsorted(dividers, minute_val, side="right")
    target = dividers[divider_idx]

    if minute_val == target:
        # if dt is exactly on the divider, go to the next value
        return divider_idx + 1
    else:
        return divider_idx


def previous_divider_idx(dividers: np.ndarray, minute_val: int) -> int:

    divider_idx = np.searchsorted(dividers, minute_val)

    if divider_idx == 0:
        raise ValueError("Cannot go earlier in calendar!")

    return divider_idx - 1


def compute_all_minutes(
    opens_in_ns: np.ndarray,
    break_starts_in_ns: np.ndarray,
    break_ends_in_ns: np.ndarray,
    closes_in_ns: np.ndarray,
) -> np.ndarray:
    """
    Given arrays of opens and closes (in nanoseconds) and optionally
    break_starts and break ends, return an array of each minute between the
    opens and closes.

    NOTE: Add an extra minute to ending boundaries (break_start and close)
    so we include the last bar (arange doesn't include its stop).
    """
    pieces = []
    for open_time, break_start_time, break_end_time, close_time in zip(
        opens_in_ns, break_starts_in_ns, break_ends_in_ns, closes_in_ns
    ):
        if break_start_time != NP_NAT:
            pieces.append(
                np.arange(
                    open_time,
                    break_start_time + NANOSECONDS_PER_MINUTE,
                    NANOSECONDS_PER_MINUTE,
                )
            )
            pieces.append(
                np.arange(
                    break_end_time,
                    close_time + NANOSECONDS_PER_MINUTE,
                    NANOSECONDS_PER_MINUTE,
                )
            )
        else:
            pieces.append(
                np.arange(
                    open_time,
                    close_time + NANOSECONDS_PER_MINUTE,
                    NANOSECONDS_PER_MINUTE,
                )
            )
    out = np.concatenate(pieces).view("datetime64[ns]")
    return out


def parse_timestamp(
    timestamp: Date | Minute,
    param_name: str,
    utc: bool = True,
    raise_oob: bool = False,
    calendar: ExchangeCalendar | None = None,
) -> pd.Timestamp:
    """Parse input intended to represent either a date or a minute.

    Parameters
    ----------
    timestamp
        Input to be parsed as either a Date or a Minute. Must be valid
        input to pd.Timestamp.

    param_name
        Name of a parameter that was to receive a Date or Minute.

    utc : default: True
        True - convert / set timezone to "UTC".
        False - leave any timezone unchanged. Note, if timezone of
        `timestamp` is "UTC" then will remain as "UTC".

    raise_oob : default: False
        True to raise MinuteOutOfBounds if `timestamp` is earlier than the
        first trading minute or later than the last trading minute of
        `calendar`. Only use when `timestamp` represents a Minute (as
        opposed to a Date). If True then `calendar` must be passed.

    calendar
        ExchangeCalendar against which to evaluate out-of-bounds timestamps.
        Only requried if `raise_oob` True.

    Raises
    ------
    TypeError
        If `timestamp` is not of type [pd.Timestamp | str | int | float |
            datetime.datetime].

    ValueError
        If `timestamp` is not an acceptable single-argument input to
        pd.Timestamp.

    exchange_calendars.errors.MinuteOutOfBounds
        If `raise_oob` True and `timestamp` parses to a valid timestamp
        although timestamp is either before `calendar`'s first trading
        minute or after `calendar`'s last trading minute.
    """
    try:
        ts = pd.Timestamp(timestamp)
    except Exception as e:
        msg = (
            f"Parameter `{param_name}` receieved as '{timestamp}' although a Date or"
            f" Minute must be passed as a pd.Timestamp or a valid single-argument"
            f" input to pd.Timestamp."
        )
        if isinstance(e, TypeError):
            raise TypeError(msg) from e
        else:
            raise ValueError(msg) from e

    if utc:
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

    ts = ts.floor("T")

    if raise_oob:
        if calendar is None:
            raise ValueError("`calendar` must be passed if `raise_oob` is True.")
        if ts < calendar.first_trading_minute or ts > calendar.last_trading_minute:
            raise errors.MinuteOutOfBounds(calendar, ts, param_name)

    return ts


def parse_date(
    date: Date,
    param_name: str,
    raise_oob: bool = False,
    calendar: ExchangeCalendar | None = None,
) -> pd.Timestamp:
    """Parse input intended to represent a date.

     Parameters
     ----------
     date
         Input to be parsed as date. Must be valid input to pd.Timestamp
         and have a time component of 00:00.

     param_name
         Name of a parameter that was to receive a date.

    raise_oob : default: False
        True to raise DateOutOfBounds if `date` is earlier than the
        first session or later than the last session of `calendar`. NB if
        True then `calendar` must be passed.

    calendar
        ExchangeCalendar against which to evalute out-of-bounds dates.
        Only requried if `raise_oob` True.

    Returns
     -------
     pd.Timestamp
         pd.Timestamp (UTC with time component of 00:00).

     Raises
     ------
     Errors as `parse_timestamp` and additionally:

     ValueError
         If `date` time component is not 00:00.

         If `date` is timezone aware and timezone is not UTC.

    exchange_calendars.errors.DateOutOfBounds
        If `raise_oob` True and `date` parses to a valid timestamp although
        timestamp is before `calendar`'s first session or after
        `calendar`'s last session.
    """
    ts = parse_timestamp(date, param_name, utc=False)

    if not (ts.tz is None or ts.tz.zone == "UTC"):
        raise ValueError(
            f"Parameter `{param_name}` parsed as '{ts}' although a Date must be"
            f" timezone naive or have timezone as 'UTC'."
        )

    if not ts == ts.normalize():
        raise ValueError(
            f"Parameter `{param_name}` parsed as '{ts}' although a Date must have"
            f" a time component of 00:00."
        )

    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    if raise_oob:
        if calendar is None:
            raise ValueError("`calendar` must be passed if `raise_oob` is True.")
        if ts < calendar.first_session or ts > calendar.last_session:
            raise errors.DateOutOfBounds(calendar, ts, param_name)

    return ts


def parse_session(
    calendar: ExchangeCalendar, session: Session, param_name: str
) -> pd.Timestamp:
    """Parse input intended to represent a session label.

    Parameters
    ----------
    calendar
        Calendar against which to evaluate `session`.

    session
        Input to be parsed as session. Must be valid input to pd.Timestamp,
        have a time component of 00:00 and represent a session of
        `calendar`.

    param_name
        Name of a parameter that was to receive a session label.

    Returns
    -------
    pd.Timestamp
        pd.Timestamp (UTC with time component of 00:00) that represents a
        real session of `calendar`.

    Raises
    ------
    Errors as `parse_date` and additionally:

    exchange_calendars.errors.NotSessionError
        If `session` parses to a valid date although date does not
        represent a session of `calendar`.
    """
    ts = parse_date(session, param_name)
    # don't check via is_session to allow for more specific error message if
    # `session` is out-of-bounds
    if ts not in calendar.schedule.index:
        raise errors.NotSessionError(calendar, ts, param_name)
    return ts


class _TradingIndex:
    """Create a trading index.

    Credit to @Stryder-Git at pandas_market_calendars for showing the way
    with a vectorised solution to creating trading indices.

    Parameters
    ----------
    All parameters as ExchangeCalendar.trading_index
    """

    def __init__(
        self,
        calendar: ExchangeCalendar,
        start: Date,
        end: Date,
        period: pd.Timedelta,
        closed: str,  # Literal["left", "right", "both", "neither"] when min python 3.8
        force_close: bool,
        force_break_close: bool,
        curtail_overlaps: bool,
    ):
        self.closed = closed
        self.force_break_close = force_break_close
        self.force_close = force_close
        self.curtail_overlaps = curtail_overlaps

        # get session bound values over requested range
        slice_start = calendar.all_sessions.searchsorted(start)
        slice_end = calendar.all_sessions.searchsorted(end, side="right")
        slce = slice(slice_start, slice_end)

        self.interval_nanos = period.value
        self.dtype = np.int64 if self.interval_nanos < 3000000000 else np.int32

        self.opens = calendar.market_opens_nanos[slce]
        self.closes = calendar.market_closes_nanos[slce]
        self.break_starts = calendar.market_break_starts_nanos[slce]
        self.break_ends = calendar.market_break_ends_nanos[slce]

        self.mask = self.break_starts != pd.NaT.value  # break mask
        self.has_break = self.mask.any()

        self.defaults = {
            "closed": self.closed,
            "force_close": self.force_close,
            "force_break_close": self.force_break_close,
        }

    @property
    def closed_right(self) -> bool:
        return self.closed in ["right", "both"]

    @property
    def closed_left(self) -> bool:
        return self.closed in ["left", "both"]

    def verify_non_overlapping(self):
        """Raise IndicesOverlapError if indices will overlap."""
        if not self.closed_right:
            return

        def _check(
            start_nanos: np.ndarray, end_nanos: np.ndarray, next_start_nanos: np.ndarray
        ):
            """Raise IndicesOverlap Error if indices would overlap.

            `next_start_nanos` describe start of (sub)session that follows and could
            overlap with (sub)session described by `start_nanos` and `end_nanos`.

            All inputs should be of same length.
            """
            num_intervals = np.ceil((end_nanos - start_nanos) / self.interval_nanos)
            right = start_nanos + num_intervals * self.interval_nanos
            if self.closed == "right" and (right > next_start_nanos).any():
                raise errors.IndicesOverlapError()
            if self.closed == "both" and (right >= next_start_nanos).any():
                raise errors.IndicesOverlapError()

        if self.has_break:
            if not self.force_break_close:
                _check(
                    self.opens[self.mask],
                    self.break_starts[self.mask],
                    self.break_ends[self.mask],
                )

        if not self.force_close:
            opens, closes, next_opens = (
                self.opens[:-1],
                self.closes[:-1],
                self.opens[1:],
            )
            _check(opens, closes, next_opens)
            if self.has_break:
                mask = self.mask[:-1]
                _check(self.break_ends[:-1][mask], closes[mask], next_opens[mask])

    def _create_index_for_sessions(
        self,
        start_nanos: np.ndarray,
        end_nanos: np.ndarray,
        force_close: bool,
        limit_nanos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create nano array of indices for sessions of given bounds."""
        if start_nanos.size == 0:
            return start_nanos

        # evaluate number of indices for each session
        num_intervals = (end_nanos - start_nanos) / self.interval_nanos
        num_indices = np.ceil(num_intervals).astype("int")

        if force_close:
            if self.closed_right:
                on_freq = (num_intervals == num_indices).all()
                if not on_freq:
                    num_indices -= 1  # add the close later
            else:
                on_freq = False

        if self.closed == "both":
            num_indices += 1
        elif self.closed == "neither":
            num_indices -= 1

        # by session, evaluate a range of int such that indices of a session
        # could be evaluted from [ session_open + (freq * i) for i in range ]
        start = 0 if self.closed_left else 1
        func = np.vectorize(lambda stop: np.arange(start, stop), otypes=[np.ndarray])
        stop = num_indices if self.closed_left else num_indices + 1
        ranges = np.concatenate(func(stop), axis=0, dtype=self.dtype)

        # evaluate index as nano array
        base = start_nanos.repeat(num_indices)
        index = base + ranges * self.interval_nanos

        # this is an ugly little clause to cover a very specific rare
        # circumstance. See comment towards start of `_trading_index`
        if limit_nanos is not None:
            limit_nanos = limit_nanos.repeat(num_indices)
            mask = index > limit_nanos
            index[mask] = limit_nanos[mask]

        if force_close and not on_freq:
            index = np.concatenate((index, end_nanos))
            index.sort()

        return index

    def _trading_index(self) -> np.ndarray:
        """Create trading index as nano array."""

        if self.has_break:
            # Create arrays of each session/subsession start and end time ...

            # sessions with breaks

            # limit only passed for index_am. It's there to provide for a rare
            # circumstance when creating the 'right' side of `trading_index_intervals`
            # with `self.force_close` = True, `self.force_break_close` = False,
            # for a calendar with breaks and for a `period` sufficiently high that the
            # right side of the last interval of the am session will exceed the day
            # close. Limiting the right side of this last am interval to the day close
            # ensures that the ordering of the left and right indices remain in sync
            # (to the contrary the right side gets ahead of itself).
            # NB not relevant for 'trading_index' as `verify_non_overlapping` will
            # raise an error in circumstances when limit would otherwise be enforced.
            if self.force_close and not self.force_break_close:
                limit = self.closes[self.mask]
            else:
                limit = None
            index_am = self._create_index_for_sessions(
                self.opens[self.mask],
                self.break_starts[self.mask],
                self.force_break_close,
                limit,
            )

            index_pm = self._create_index_for_sessions(
                self.break_ends[self.mask], self.closes[self.mask], self.force_close
            )

            # sessions without a break
            index_day = self._create_index_for_sessions(
                self.opens[~self.mask], self.closes[~self.mask], self.force_close
            )

            # put it all together
            index = np.concatenate((index_am, index_pm, index_day))
            index.sort()
        else:
            index = self._create_index_for_sessions(
                self.opens, self.closes, self.force_close
            )

        return index

    def trading_index(self) -> pd.DatetimeIndex:
        """Create trading index as a DatetimeIndex."""
        self.verify_non_overlapping()
        index = self._trading_index()
        return pd.DatetimeIndex(index, tz="UTC")

    @contextlib.contextmanager
    def _override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        yield
        for k, v in self.defaults.items():
            setattr(self, k, v)

    def trading_index_intervals(self) -> pd.IntervalIndex:
        """Create trading index as a pd.IntervalIndex."""
        with self._override_defaults(
            closed="left", force_close=False, force_break_close=False
        ):
            left = self._trading_index()

        if not (self.force_close or self.force_break_close):
            right = left + self.interval_nanos
        else:
            with self._override_defaults(closed="right"):
                right = self._trading_index()

        overlaps_next = right[:-1] > left[1:]
        if overlaps_next.any():
            if self.curtail_overlaps:
                right[:-1][overlaps_next] = left[1:][overlaps_next]
            else:
                raise errors.IntervalsOverlapError()

        left = pd.DatetimeIndex(left, tz="UTC")
        right = pd.DatetimeIndex(right, tz="UTC")
        return pd.IntervalIndex.from_arrays(left, right, self.closed)


# TODO post resolving PR #71:
#   Move `trading_index` to method of Exchange Calendar
#   Bring over / revise tests:
#       include hypothesis fuzz tests - mark so they do not run by default
#       include tests with known input / output for default test suite
def trading_index(
    self,
    start: Date,
    end: Date,
    period: pd.Timedelta | str,
    intervals: bool = True,
    closed: str = "left",  # when move to min 3.8 Literal["left", "right", "both", "neither"]
    force_close: bool = False,
    force_break_close: bool = False,
    curtail_overlaps: bool = False,
    parse: bool = True,
    _parse: bool = True,
) -> pd.DatetimeIndex | pd.IntervalIndex:
    """Create a trading index.

    Create a trading index of given `period` over a given range of dates.

    Execution time is related to the number of indices created. The longer
    the range of dates covered and/or the shorter the period (i.e. higher
    the frequency), the longer the execution. Whilst an index with 4000
    indices might be created in a couple of miliseconds, a high frequency
    index with 2 million indices might take a second or two.

    `self.side` and which minutes are treated as trading minutes is
    irrelevant in the evaluation of the trading index.

    Parameters
    ----------
    start
        Start of session range over which to create index.

    end
        End of session range over which to create index.

    period
        If `intervals` is True, the length of each interval. If `intervals`
        is False, the distance between indices. Period described by a
        pd.Timedelta or str acceptable as a single input to pd.Timedelta.
        `period` cannot be greater than 1 day.

        Examples of valid `period` input:
            pd.Timedelta(minutes=15), pd.Timedelta(minutes=15, hours=2)
            '15min', '15T', '1H', '4h', '1d', '30s', '2s', '500ms'.
        Examples of invalid `period` input:
            '15minutes', '2d'.

    intervals : default: True
        True to return trading index as a pd.IntervalIndex with indices
        representing explicit intervals.

        False to return trading index as a pd.DatetimeIndex with indices
        that implicitely represent a period according to `closed`.

        If `period` is '1d' then trading index will always be returned
        as a pd.DatetimeIndex.

    closed : {"left", "right", "both", "neither"}
        (ignored if `period` is '1d'.)

        If `intervals` is True:
            The side that intervals should be closed on. Must be either
            "left" or "right" (any time during a session must belong to
            one interval and one interval only).

        If `intervals` is False, then on which side of each period an
        indice should be defined. The first and last indices of each
        (sub)session will be defined according to:
            "left" - include left side of first period, do not include
                right side of last period.
            "right" - do not include left side of first period, include
                right side of last period.
            "both" - include both left side of first period and right side
                of last period.
            "neither" - do not include either left side of first period or
                right side of last period.
        NB if `period` is not a factor of the (sub)session length then
        "right" or "both" will result in an indice being defined after the
        (sub)session close. See `force_close` and `force_break_close`.

    force_close : default: False
        (ignored if `period` is '1d'.)
        (irrelevant if `intervals` is False and `closed` is "left" or
        "neither".)

        Defines behaviour if right side of a session's last period falls
        after the session close.

        True - define right side of last period as session close.

        False - define right side of last period after session close.
        In this case the period represented by the indice will include a
        non-trading period.

    force_break_close : default: False
        (ignored if `period` is '1d'.)
        (irrelevant if `intervals` is False and `closed` is "left" or
        "neither.)

        Defines behaviour if right side of last pre-break period falls
        after the start of the break.

        True - define right side of this period as break start.

        False - define right side of this period after break start.
        In this case the period represented by the indice will include a
        non-trading period.

    curtail_overlaps : default: False
        (ignored if `period` is '1d'.)
        (irrelevant if (`intervals` is False) or (`force_close` and
        `force_break_close` are True).)

        What action to take in the event that a period ends after the
        start of the next period. (This can occur if `period` is longer
        than a break or the period between one session's close and the
        next session's open.)

            If True, the right of the earlier of two overlapping periods
            will be curtailed to the left of the latter period. (NB
            consequently the period length will not be constant for all
            periods.)

            If False, will raise IntervalsOverlapError.

    parse : default: True
        Determines if `start` and `end` values are parsed. If these inputs
        are passed as pd.Timestamp with no time component and tz as UTC
        then can pass `parse` as False to save around 500µs on the
        execution.

    Returns
    -------
    pd.IntervalIndex or pd.DatetimeIndex
        Trading index.

        If `intervals` is False or `period` is '1d' then returned as a
            pd.DatetimeIndex.
        If `intervals` is True (default) returned as pd.IntervalIndex.

    Raises
    ------
    exchange_calendars.errors.IntervalsOverlapError
        If `intervals` is True and right side of one or more indices
        would fall after the left of the subsequent indice. This can occur
        if `period` is longer than a break or period between one
        session's close and the next session's open.

    exchange_calendars.errors.IntervalsOverlapError
        If `intervals` is False and an indice would otherwise fall to the
        right of (later than) the subsequent indice. This can occur if
        `period` is longer than a break or period between one session's
        close and the next session's open.

    Notes
    -----
    `_parse` included for compatibility. Input will not be parsed if either
        `parse` or `_parse` are False.

    Credit to @Stryder-Git at pandas_market_calendars for showing the way
    with a vectorised solution to creating trading indices (a variation of
    which is employed within the underlying _TradingIndex class).
    """
    if parse and _parse:
        start = parse_date(start, "start", True, self)
        end = parse_date(end, "end", True, self)

    if not isinstance(period, pd.Timedelta):
        try:
            period = pd.Timedelta(period)
        except ValueError:
            msg = (
                f"`period` receieved as '{period}' although takes type 'pd.Timedelta'"
                " or a type 'str' that is valid as a single input to 'pd.Timedelta'."
                " Examples of valid input: pd.Timestamp('15T'), '15min', '15T', '1H',"
                " '4h', '1d', '5s', 500ms'."
            )
            raise ValueError(msg) from None

    if period > pd.Timedelta(1, "D"):
        msg = (
            f"`period` cannot be greater than one day although received as '{period}'."
        )
        raise ValueError(msg)

    if period == pd.Timedelta(1, "D"):
        return self.sessions_in_range(start, end)

    if intervals and closed in ["both", "neither"]:
        raise ValueError(f"If `intervals` is True then `closed` cannot be '{closed}'.")

    # method exposes public methods of _TradingIndex.
    _trading_index = _TradingIndex(
        self,
        start,
        end,
        period,
        closed,
        force_close,
        force_break_close,
        curtail_overlaps,
    )

    if not intervals:
        return _trading_index.trading_index()
    else:
        return _trading_index.trading_index_intervals()

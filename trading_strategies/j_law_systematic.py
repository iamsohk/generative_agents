"""Pure Python implementation of the J Law momentum rule set.

The original proof-of-concept for this project relied heavily on
``pandas`` DataFrames.  The execution environment for the kata, however,
ships without the pandas dependency, which meant the previous version of
the module could not even be imported.  This rewrite keeps the public
surface area focused on the rule logic while expressing all calculations
with standard library collections.  The behaviour matches the high-level
requirements laid out in the user prompt: universe construction, breakout
entries, position sizing, stop placement, and the main exit tactics.

Data model
==========

``JLawSystematicStrategy`` operates on dictionaries of pre-cleaned OHLCV
records.  Each symbol has an associated :class:`SymbolHistory` that
stores the daily bars and derives the rolling statistics required by the
strategy (moving averages, returns, ATR, etc.).  Benchmarks and industry
proxies are handled by :class:`TimeSeries` helper objects so that
relative-strength comparisons remain straightforward.

The end result is a strategy class that can be unit tested without any
third-party dependencies while still remaining faithful to the
mechanical specification of the J Law framework.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DateLike = date | datetime


def _to_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _moving_average(values: List[float], window: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    if window <= 0:
        return result
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        if idx >= window - 1:
            result[idx] = running / window
    return result


def _rolling_max(values: List[float], window: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    for idx in range(window - 1, len(values)):
        start = idx - window + 1
        result[idx] = max(values[start : idx + 1])
    return result


def _pct_change(values: List[float], periods: int) -> List[Optional[float]]:
    result: List[Optional[float]] = [None] * len(values)
    if periods <= 0:
        return result
    for idx in range(periods, len(values)):
        prev = values[idx - periods]
        if prev != 0:
            result[idx] = values[idx] / prev - 1.0
    return result


def _true_range(high: List[float], low: List[float], close: List[float]) -> List[float]:
    result = [0.0] * len(close)
    for idx, (hi, lo) in enumerate(zip(high, low)):
        if idx == 0:
            result[idx] = hi - lo
        else:
            prev_close = close[idx - 1]
            result[idx] = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
    return result


@dataclass
class DailyBar:
    """Represents a single end-of-day bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    market_cap: float
    industry: str
    exchange: str
    dollar_volume: Optional[float] = None
    is_spac: bool = False
    borrow_available: bool = True

    def resolved_dollar_volume(self) -> float:
        return float(self.dollar_volume if self.dollar_volume is not None else self.close * self.volume)


class SymbolHistory:
    """Holds time-series data and derived metrics for one symbol."""

    def __init__(self, bars: Iterable[DailyBar]) -> None:
        ordered = sorted(bars, key=lambda bar: bar.date)
        if not ordered:
            raise ValueError("SymbolHistory requires at least one bar")

        self.dates = [bar.date for bar in ordered]
        self.open = [float(bar.open) for bar in ordered]
        self.high = [float(bar.high) for bar in ordered]
        self.low = [float(bar.low) for bar in ordered]
        self.close = [float(bar.close) for bar in ordered]
        self.volume = [float(bar.volume) for bar in ordered]
        self.market_cap = [float(bar.market_cap) for bar in ordered]
        self.dollar_volume = [float(bar.resolved_dollar_volume()) for bar in ordered]
        self.industry = [bar.industry for bar in ordered]
        self.exchange = [bar.exchange for bar in ordered]
        self.is_spac = [bool(bar.is_spac) for bar in ordered]
        self.borrow_available = [bool(bar.borrow_available) for bar in ordered]

        self.sma_10 = _moving_average(self.close, 10)
        self.sma_20 = _moving_average(self.close, 20)
        self.sma_50 = _moving_average(self.close, 50)
        self.sma_200 = _moving_average(self.close, 200)
        self.avg_volume_20 = _moving_average(self.volume, 20)
        self.avg_dollar_volume_20 = _moving_average(self.dollar_volume, 20)
        self.ret_21 = _pct_change(self.close, 21)
        self.ret_63 = _pct_change(self.close, 63)
        self.ret_126 = _pct_change(self.close, 126)
        self.rolling_high_50 = _rolling_max(self.close, 50)

        tr = _true_range(self.high, self.low, self.close)
        self.atr_14 = _moving_average(tr, 14)

    def index_for(self, session: date) -> Optional[int]:
        idx = bisect.bisect_left(self.dates, session)
        if idx < len(self.dates) and self.dates[idx] == session:
            return idx
        return None

    def slice(self, end_idx: int, lookback: int) -> slice:
        start = max(0, end_idx - lookback + 1)
        return slice(start, end_idx + 1)

    def row(self, idx: int) -> Dict[str, object]:
        return {
            "date": self.dates[idx],
            "open": self.open[idx],
            "high": self.high[idx],
            "low": self.low[idx],
            "close": self.close[idx],
            "volume": self.volume[idx],
            "market_cap": self.market_cap[idx],
            "dollar_volume": self.dollar_volume[idx],
            "industry": self.industry[idx],
            "exchange": self.exchange[idx],
            "is_spac": self.is_spac[idx],
            "borrow_available": self.borrow_available[idx],
            "sma_10": self.sma_10[idx],
            "sma_20": self.sma_20[idx],
            "sma_50": self.sma_50[idx],
            "sma_200": self.sma_200[idx],
            "avg_volume_20": self.avg_volume_20[idx],
            "avg_dollar_volume_20": self.avg_dollar_volume_20[idx],
            "ret_21": self.ret_21[idx],
            "ret_63": self.ret_63[idx],
            "ret_126": self.ret_126[idx],
            "rolling_high": self.rolling_high_50[idx],
            "atr_14": self.atr_14[idx],
        }


class TimeSeries:
    """Helper that stores a dated price series and derived returns."""

    def __init__(self, points: Sequence[Tuple[DateLike, float]]) -> None:
        ordered = sorted((_to_date(d), float(v)) for d, v in points)
        if not ordered:
            raise ValueError("TimeSeries requires data")
        self.dates = [d for d, _ in ordered]
        self.values = [v for _, v in ordered]
        self.ret_21 = _pct_change(self.values, 21)
        self.ret_63 = _pct_change(self.values, 63)
        self.ret_126 = _pct_change(self.values, 126)

    def index_for(self, session: date) -> Optional[int]:
        idx = bisect.bisect_left(self.dates, session)
        if idx < len(self.dates) and self.dates[idx] == session:
            return idx
        return None

    def value_on(self, session: date) -> Optional[float]:
        idx = self.index_for(session)
        if idx is None:
            return None
        return float(self.values[idx])

    def return_on(self, session: date, window: int) -> Optional[float]:
        idx = self.index_for(session)
        if idx is None:
            return None
        series = {21: self.ret_21, 63: self.ret_63, 126: self.ret_126}[window]
        value = series[idx]
        return None if math.isnan(value) else float(value)


@dataclass
class StrategyConfig:
    """Configuration bundle for :class:`JLawSystematicStrategy`."""

    min_price: float = 10.0
    min_market_cap: float = 2_000_000_000.0
    min_avg_dollar_volume: Optional[float] = 50_000_000.0
    min_avg_volume: int = 500_000
    relative_strength_ratio_short: float = 2.0
    relative_strength_ratio_medium: float = 2.0
    breakout_lookback: int = 50
    breakout_volume_multiplier: float = 1.5
    entry_slippage_limit: float = 0.01
    risk_fraction: float = 0.015
    stop_atr_multiple: float = 1.0
    fixed_stop_pct: float = 0.08
    max_position_fraction: float = 0.15
    max_industry_fraction: float = 0.40
    max_positions: int = 12
    min_positions: int = 5
    rs_weakness_days: int = 10
    partial_take_profit: float = 0.20
    trailing_stop_lookback: int = 50
    weekly_sma_lookback: int = 50
    allow_thematic_filter: bool = True
    blacklist: Sequence[str] = field(default_factory=tuple)
    allowed_exchanges: Sequence[str] = ("NYSE", "NASDAQ")


@dataclass
class Position:
    symbol: str
    entry_date: date
    entry_price: float
    shares: int
    stop: float
    industry: str
    scaled_out: bool = False
    rs_weak_days: int = 0
    last_updated: date = field(default_factory=date.today)

    def notional(self) -> float:
        return self.entry_price * self.shares


class JLawSystematicStrategy:
    """Systematic trading strategy inspired by the J Law framework."""

    def __init__(
        self,
        price_data: Dict[str, Iterable[DailyBar]],
        benchmark: TimeSeries,
        industry_proxies: Optional[Dict[str, TimeSeries]] = None,
        market_pressure: Optional[Dict[date, bool]] = None,
        config: Optional[StrategyConfig] = None,
    ) -> None:
        if not price_data:
            raise ValueError("price_data must contain at least one symbol")
        self.histories: Dict[str, SymbolHistory] = {
            symbol: SymbolHistory(bars) for symbol, bars in price_data.items()
        }
        self.benchmark = benchmark
        self.industry_proxies = industry_proxies or {}
        self.market_pressure = market_pressure or {}
        self.config = config or StrategyConfig()
        self.positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------
    # Helpers for data access
    # ------------------------------------------------------------------
    def _benchmark_returns(self, session: date) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        return (
            self.benchmark.return_on(session, 21),
            self.benchmark.return_on(session, 63),
            self.benchmark.return_on(session, 126),
        )

    def _industry_returns(self, industry: str, session: date) -> Tuple[Optional[float], Optional[float]]:
        series = self.industry_proxies.get(industry)
        if not series:
            return (None, None)
        return (series.return_on(session, 21), series.return_on(session, 63))

    # ------------------------------------------------------------------
    # Universe selection
    # ------------------------------------------------------------------
    def _universe_filter(self, session: date) -> Dict[str, Dict[str, object]]:
        cfg = self.config
        result: Dict[str, Dict[str, object]] = {}
        for symbol, history in self.histories.items():
            idx = history.index_for(session)
            if idx is None:
                continue
            row = history.row(idx)
            if row["close"] < cfg.min_price:
                continue
            if row["market_cap"] < cfg.min_market_cap:
                continue
            if row["exchange"] not in cfg.allowed_exchanges:
                continue
            avg_dollar_volume = row["avg_dollar_volume_20"]
            avg_volume = row["avg_volume_20"]
            if cfg.min_avg_dollar_volume is not None:
                if avg_dollar_volume is None or avg_dollar_volume < cfg.min_avg_dollar_volume:
                    continue
            else:
                if avg_volume is None or avg_volume < cfg.min_avg_volume:
                    continue
            if row.get("is_spac"):
                continue
            if not row.get("borrow_available", True):
                continue
            if symbol in cfg.blacklist:
                continue
            result[symbol] = row
        return result

    # ------------------------------------------------------------------
    # Signal filters
    # ------------------------------------------------------------------
    def _trend_momentum_filter(self, candidates: Dict[str, Dict[str, object]], session: date) -> Dict[str, Dict[str, object]]:
        cfg = self.config
        bm_ret_21, _, bm_ret_126 = self._benchmark_returns(session)
        filtered: Dict[str, Dict[str, object]] = {}
        for symbol, row in candidates.items():
            if row.get("sma_200") is None or row["close"] <= row["sma_200"]:
                continue
            if row.get("sma_10") is None or row.get("sma_20") is None:
                continue
            if row["sma_10"] <= row["sma_20"]:
                continue
            ret_21 = row.get("ret_21")
            ret_126 = row.get("ret_126")
            if ret_21 is None or bm_ret_21 in (None, 0.0):
                continue
            if ret_126 is None or bm_ret_126 in (None, 0.0):
                continue
            rs_1m = ret_21 / bm_ret_21
            rs_6m = ret_126 / bm_ret_126
            if rs_1m < cfg.relative_strength_ratio_short:
                continue
            if rs_6m < cfg.relative_strength_ratio_medium:
                continue
            row = dict(row)
            row["rs_1m"] = rs_1m
            row["rs_6m"] = rs_6m
            row["benchmark_ret_21"] = bm_ret_21
            row["benchmark_ret_63"] = self.benchmark.return_on(session, 63)
            row["benchmark_ret_126"] = bm_ret_126
            filtered[symbol] = row
        return filtered

    def _thematic_filter(self, candidates: Dict[str, Dict[str, object]], session: date) -> Dict[str, Dict[str, object]]:
        if not self.config.allow_thematic_filter or not self.industry_proxies:
            return candidates
        filtered: Dict[str, Dict[str, object]] = {}
        for symbol, row in candidates.items():
            industry = row.get("industry")
            if not industry:
                continue
            ind_ret_21, ind_ret_63 = self._industry_returns(industry, session)
            bench_21 = row.get("benchmark_ret_21")
            bench_63 = row.get("benchmark_ret_63")
            if ind_ret_21 is None or bench_21 is None or ind_ret_63 is None or bench_63 is None:
                continue
            if ind_ret_21 <= bench_21:
                continue
            if ind_ret_63 <= bench_63:
                continue
            row = dict(row)
            row["industry_ret_21"] = ind_ret_21
            row["industry_ret_63"] = ind_ret_63
            filtered[symbol] = row
        return filtered

    def _breakout_filter(self, candidates: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        cfg = self.config
        filtered: Dict[str, Dict[str, object]] = {}
        for symbol, row in candidates.items():
            breakout = row.get("rolling_high")
            if breakout is None:
                continue
            if row["close"] < breakout:
                continue
            if row["volume"] < cfg.breakout_volume_multiplier * (row.get("avg_volume_20") or 0.0):
                continue
            if row["close"] > breakout * (1 + cfg.entry_slippage_limit):
                continue
            filtered[symbol] = row
        return filtered

    # ------------------------------------------------------------------
    # Risk management helpers
    # ------------------------------------------------------------------
    def _initial_stop(self, row: Dict[str, object]) -> float:
        cfg = self.config
        base = max(row["low"], row.get("sma_20") or row["low"])
        atr = row.get("atr_14") or 0.0
        atr_stop = base - cfg.stop_atr_multiple * atr
        pct_stop = row["close"] * (1 - cfg.fixed_stop_pct)
        return max(atr_stop, pct_stop)

    def _position_size(self, equity: float, entry_price: float, stop_price: float) -> int:
        cfg = self.config
        risk_amount = equity * cfg.risk_fraction
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0
        shares = int(math.floor(risk_amount / risk_per_share))
        max_notional = equity * cfg.max_position_fraction
        if shares * entry_price > max_notional:
            shares = int(math.floor(max_notional / entry_price))
        return shares

    def _industry_exposure(self) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        for position in self.positions.values():
            exposure[position.industry] = exposure.get(position.industry, 0.0) + position.notional()
        return exposure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_daily(self, session: DateLike, equity: float) -> Dict[str, Dict[str, float]]:
        session_date = _to_date(session)
        orders: Dict[str, Dict[str, float]] = {}

        # Process exits before entries.
        orders.update(self._generate_exit_signals(session_date))

        if self.market_pressure.get(session_date, False):
            return orders

        if len(self.positions) >= self.config.max_positions:
            return orders

        universe = self._universe_filter(session_date)
        if not universe:
            return orders

        candidates = self._trend_momentum_filter(universe, session_date)
        candidates = self._thematic_filter(candidates, session_date)
        candidates = self._breakout_filter(candidates)
        if not candidates:
            return orders

        # Sort by strongest short-term relative strength.
        sorted_syms = sorted(candidates.items(), key=lambda item: item[1]["rs_1m"], reverse=True)
        industry_exposure = self._industry_exposure()
        available_slots = self.config.max_positions - len(self.positions)

        for symbol, row in sorted_syms[:available_slots]:
            if symbol in self.positions:
                continue
            stop_price = self._initial_stop(row)
            shares = self._position_size(equity, row["close"], stop_price)
            if shares <= 0:
                continue
            industry = row.get("industry", "UNKNOWN")
            projected = industry_exposure.get(industry, 0.0) + shares * row["close"]
            if projected > self.config.max_industry_fraction * equity:
                continue
            orders[symbol] = {
                "action": "buy",
                "shares": shares,
                "price": row["close"],
                "stop": stop_price,
            }
            industry_exposure[industry] = projected

        return orders

    def apply_fills(self, fills: Dict[str, Dict[str, float]], session: Optional[DateLike] = None) -> None:
        session_date = _to_date(session) if session is not None else None
        for symbol, details in fills.items():
            action = details.get("action")
            shares = int(details.get("shares", 0))
            price = float(details.get("price", 0.0))
            stop = float(details.get("stop", 0.0))
            partial = bool(details.get("partial", symbol.endswith("_scaleout")))
            new_stop = details.get("new_stop")

            base_symbol = symbol[:-9] if symbol.endswith("_scaleout") else symbol

            if action == "buy" and shares > 0:
                if session_date is None:
                    raise ValueError("apply_fills requires session date for new positions")
                history = self.histories.get(base_symbol)
                industry = "UNKNOWN"
                if history is not None:
                    idx = history.index_for(session_date)
                    if idx is not None:
                        industry = history.industry[idx]
                self.positions[base_symbol] = Position(
                    symbol=base_symbol,
                    entry_date=session_date,
                    entry_price=price,
                    shares=shares,
                    stop=stop,
                    industry=industry,
                )
            elif action == "sell" and base_symbol in self.positions:
                position = self.positions[base_symbol]
                if partial:
                    position.shares = max(position.shares - shares, 0)
                    if new_stop is not None:
                        position.stop = float(new_stop)
                    position.scaled_out = True
                    if position.shares == 0:
                        del self.positions[base_symbol]
                else:
                    del self.positions[base_symbol]

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def _generate_exit_signals(self, session: date) -> Dict[str, Dict[str, float]]:
        if not self.positions:
            return {}

        orders: Dict[str, Dict[str, float]] = {}
        for symbol, position in list(self.positions.items()):
            history = self.histories.get(symbol)
            if history is None:
                continue
            idx = history.index_for(session)
            if idx is None:
                continue
            row = history.row(idx)
            price = row["close"]

            # Hard stop
            if price <= position.stop:
                orders[symbol] = {"action": "sell", "shares": position.shares, "price": price}
                continue

            # Relative strength decay
            bm_ret_21, _, _ = self._benchmark_returns(session)
            ret_21 = row.get("ret_21")
            if ret_21 is not None and bm_ret_21 not in (None, 0.0):
                rs_today = ret_21 / bm_ret_21
                if rs_today < 1.0:
                    position.rs_weak_days += 1
                else:
                    position.rs_weak_days = 0
            if position.rs_weak_days >= self.config.rs_weakness_days:
                orders[symbol] = {"action": "sell", "shares": position.shares, "price": price}
                continue

            # Partial scale-out
            if (
                not position.scaled_out
                and price >= position.entry_price * (1 + self.config.partial_take_profit)
            ):
                reduce_shares = position.shares // 3
                if reduce_shares > 0:
                    new_stop = max(position.entry_price, row.get("sma_50") or position.entry_price)
                    orders[f"{symbol}_scaleout"] = {
                        "action": "sell",
                        "shares": reduce_shares,
                        "price": price,
                        "partial": True,
                        "new_stop": new_stop,
                    }
                    continue

            # Breakdown below the 50-day average with confirmation
            if row.get("sma_50") is not None and price < row["sma_50"]:
                prev_idx = idx - 1
                if prev_idx >= 0 and history.row(prev_idx).get("sma_50") is not None:
                    prev_close = history.close[prev_idx]
                    prev_sma = history.sma_50[prev_idx]
                    if prev_close < prev_sma:
                        orders[symbol] = {
                            "action": "sell",
                            "shares": position.shares,
                            "price": price,
                        }
                        continue
                if row.get("avg_volume_20") is not None and row["volume"] > row["avg_volume_20"]:
                    orders[symbol] = {
                        "action": "sell",
                        "shares": position.shares,
                        "price": price,
                    }
                    continue

            position.last_updated = session

        # Remove any fully closed positions
        for symbol, order in orders.items():
            if not symbol.endswith("_scaleout") and order.get("action") == "sell":
                self.positions.pop(symbol, None)

        return orders


__all__ = [
    "DailyBar",
    "TimeSeries",
    "SymbolHistory",
    "StrategyConfig",
    "Position",
    "JLawSystematicStrategy",
]


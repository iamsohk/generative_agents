from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_strategies import (
    DailyBar,
    JLawSystematicStrategy,
    StrategyConfig,
    TimeSeries,
)


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [float(start + step * idx) for idx in range(count)]


def make_price_history():
    start = date(2023, 1, 2)
    closes = linspace(20.0, 120.0, 260)
    volumes = [1_000_000.0 for _ in range(260)]
    volumes[-1] = 1_600_000.0  # ensure breakout volume condition is met
    bars = []
    for idx, close in enumerate(closes):
        session = start + timedelta(days=idx)
        bars.append(
            DailyBar(
                date=session,
                open=float(close),
                high=float(close * 1.01),
                low=float(close * 0.99),
                close=float(close),
                volume=float(volumes[idx]),
                market_cap=5_000_000_000.0,
                industry="SOXX",
                exchange="NASDAQ",
                dollar_volume=float(close * volumes[idx]),
            )
        )
    return {"TEST": bars}


def make_benchmark_series():
    start = date(2023, 1, 2)
    prices = linspace(100.0, 125.0, 260)
    return TimeSeries([(start + timedelta(days=i), float(price)) for i, price in enumerate(prices)])


def make_industry_proxy():
    start = date(2023, 1, 2)
    prices = linspace(50.0, 120.0, 260)
    return {"SOXX": TimeSeries([(start + timedelta(days=i), float(price)) for i, price in enumerate(prices)])}


def test_strategy_generates_entry_signal():
    price_history = make_price_history()
    benchmark = make_benchmark_series()
    industry_proxy = make_industry_proxy()
    cfg = StrategyConfig()
    strategy = JLawSystematicStrategy(price_history, benchmark, industry_proxy, config=cfg)

    last_date = price_history["TEST"][-1].date
    orders = strategy.run_daily(last_date, equity=1_000_000)

    assert "TEST" in orders
    assert orders["TEST"]["action"] == "buy"
    assert orders["TEST"]["shares"] > 0


def test_stop_is_tighter_of_atr_and_pct():
    price_history = make_price_history()
    benchmark = make_benchmark_series()
    industry_proxy = make_industry_proxy()
    strategy = JLawSystematicStrategy(price_history, benchmark, industry_proxy)

    last_date = price_history["TEST"][-1].date
    history = strategy.histories["TEST"]
    idx = history.index_for(last_date)
    row = history.row(idx)

    stop_price = strategy._initial_stop(row)

    base_level = max(row["low"], row["sma_20"])
    atr_stop = base_level - strategy.config.stop_atr_multiple * row["atr_14"]
    pct_stop = row["close"] * (1 - strategy.config.fixed_stop_pct)
    assert abs(stop_price - max(atr_stop, pct_stop)) < 1e-9

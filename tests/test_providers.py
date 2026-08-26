"""Tests for stockpred.data providers: prices (yfinance), macro (FRED/ECB),
factors (Fama-French via pandas-datareader).

Offline tests monkeypatch the underlying network-calling functions so the
tidy-shape/retry/failure logic is verified without touching the network.
Network smoke tests (marked ``network``) exercise the real providers once.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockpred.data import factors, macro, prices

# ---------------------------------------------------------------------------
# prices.fetch_prices (offline, monkeypatched yfinance)
# ---------------------------------------------------------------------------


def _fake_yf_download(ticker: str) -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "High", "Low", "Open", "Volume"], [ticker]]
    )
    data = {
        ("Adj Close", ticker): [648.0, 643.0],
        ("Close", ticker): [650.0, 645.0],
        ("High", ticker): [655.0, 648.0],
        ("Low", ticker): [640.0, 638.0],
        ("Open", ticker): [642.0, 646.0],
        ("Volume", ticker): [500000, 510000],
    }
    df = pd.DataFrame(data, index=index)
    df.columns = columns
    return df


def test_fetch_prices_returns_tidy_frame(monkeypatch):
    def fake_download(ticker, start=None, end=None, auto_adjust=False, progress=False):
        return _fake_yf_download(ticker)

    monkeypatch.setattr(prices.yf, "download", fake_download)

    df, failed = prices.fetch_prices(["ASML"], start="2024-01-01", yf_suffix=".AS")

    assert failed == []
    assert list(df.columns) == ["date", "ticker", "close", "adj_close", "volume"]
    assert (df["ticker"] == "ASML").all()  # ticker stored WITHOUT suffix
    assert len(df) == 2
    assert df["close"].tolist() == [650.0, 645.0]
    assert df["adj_close"].tolist() == [648.0, 643.0]


def test_fetch_prices_multiple_tickers_concatenated(monkeypatch):
    def fake_download(ticker, start=None, end=None, auto_adjust=False, progress=False):
        return _fake_yf_download(ticker)

    monkeypatch.setattr(prices.yf, "download", fake_download)

    df, failed = prices.fetch_prices(["ASML", "ABN"], start="2024-01-01", yf_suffix=".AS")

    assert failed == []
    assert set(df["ticker"]) == {"ASML", "ABN"}
    assert len(df) == 4


def test_fetch_prices_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_download(ticker, start=None, end=None, auto_adjust=False, progress=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient failure")
        return _fake_yf_download(ticker)

    monkeypatch.setattr(prices.yf, "download", flaky_download)
    monkeypatch.setattr(prices.time, "sleep", lambda _seconds: None)

    df, failed = prices.fetch_prices(["ASML"], start="2024-01-01", yf_suffix=".AS")

    assert failed == []
    assert calls["n"] == 3
    assert len(df) == 2


def test_fetch_prices_ticker_fails_all_retries_goes_to_failed_list(monkeypatch):
    def always_empty(ticker, start=None, end=None, auto_adjust=False, progress=False):
        return pd.DataFrame()

    monkeypatch.setattr(prices.yf, "download", always_empty)
    monkeypatch.setattr(prices.time, "sleep", lambda _seconds: None)

    df, failed = prices.fetch_prices(["NOTREAL"], start="2024-01-01", yf_suffix=".AS")

    assert failed == ["NOTREAL"]
    assert df.empty
    assert list(df.columns) == ["date", "ticker", "close", "adj_close", "volume"]


def test_fetch_prices_partial_failure(monkeypatch):
    def selective_download(ticker, start=None, end=None, auto_adjust=False, progress=False):
        if ticker == "BAD.AS":
            return pd.DataFrame()
        return _fake_yf_download(ticker)

    monkeypatch.setattr(prices.yf, "download", selective_download)
    monkeypatch.setattr(prices.time, "sleep", lambda _seconds: None)

    df, failed = prices.fetch_prices(["ASML", "BAD"], start="2024-01-01", yf_suffix=".AS")

    assert failed == ["BAD"]
    assert set(df["ticker"]) == {"ASML"}


# ---------------------------------------------------------------------------
# macro.fetch_fred (offline, monkeypatched requests.get)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_fetch_fred_returns_tidy_frame(monkeypatch):
    csv_text = "DATE,INDPRO\n2024-01-01,100.5\n2024-02-01,101.2\n"

    def fake_get(url, headers=None, timeout=None):
        assert "User-Agent" in headers
        assert "INDPRO" in url
        return _FakeResponse(csv_text)

    monkeypatch.setattr(macro.requests, "get", fake_get)

    df = macro.fetch_fred(["INDPRO"])

    assert list(df.columns) == ["date", "series_id", "value"]
    assert len(df) == 2
    assert (df["series_id"] == "INDPRO").all()
    assert df["value"].tolist() == [100.5, 101.2]


def test_fetch_fred_missing_values_dropped(monkeypatch):
    csv_text = "DATE,VIXCLS\n2024-01-01,15.2\n2024-01-02,.\n"

    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(csv_text)

    monkeypatch.setattr(macro.requests, "get", fake_get)

    df = macro.fetch_fred(["VIXCLS"])
    assert len(df) == 1
    assert df["value"].tolist() == [15.2]


def test_fetch_fred_partial_failure_collected(monkeypatch):
    csv_text = "DATE,INDPRO\n2024-01-01,100.5\n"

    def fake_get(url, headers=None, timeout=None):
        if "BADSERIES" in url:
            raise ConnectionError("network down")
        return _FakeResponse(csv_text)

    monkeypatch.setattr(macro.requests, "get", fake_get)

    df = macro.fetch_fred(["INDPRO", "BADSERIES"])
    assert set(df["series_id"]) == {"INDPRO"}


def test_fetch_fred_all_fail_returns_empty_with_correct_columns(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        raise ConnectionError("network down")

    monkeypatch.setattr(macro.requests, "get", fake_get)

    df = macro.fetch_fred(["INDPRO"])
    assert df.empty
    assert list(df.columns) == ["date", "series_id", "value"]


# ---------------------------------------------------------------------------
# macro.fetch_ecb (offline, monkeypatched ecbdata.get_series)
# ---------------------------------------------------------------------------


def test_fetch_ecb_returns_tidy_frame(monkeypatch):
    def fake_get_series(series_key):
        return pd.DataFrame(
            {
                "TIME_PERIOD": ["2024-01", "2024-02"],
                "OBS_VALUE": [8.6, 8.5],
            }
        )

    monkeypatch.setattr(macro.ecbdata, "get_series", fake_get_series)

    df = macro.fetch_ecb(["ICP.M.U2.N.000000.4.ANR"])
    assert list(df.columns) == ["date", "series_id", "value"]
    assert len(df) == 2


def test_fetch_ecb_partial_failure_skipped(monkeypatch):
    def fake_get_series(series_key):
        if series_key == "BAD.KEY":
            raise ValueError("unknown key")
        return pd.DataFrame({"TIME_PERIOD": ["2024-01"], "OBS_VALUE": [1.0]})

    monkeypatch.setattr(macro.ecbdata, "get_series", fake_get_series)

    df = macro.fetch_ecb(["GOOD.KEY", "BAD.KEY"])
    assert set(df["series_id"]) == {"GOOD.KEY"}


def test_fetch_ecb_total_failure_returns_empty_with_correct_columns(monkeypatch):
    def fake_get_series(series_key):
        raise ValueError("unknown key")

    monkeypatch.setattr(macro.ecbdata, "get_series", fake_get_series)

    df = macro.fetch_ecb(["BAD.KEY"])
    assert df.empty
    assert list(df.columns) == ["date", "series_id", "value"]


# ---------------------------------------------------------------------------
# factors.fetch_famafrench (offline, monkeypatched pandas_datareader)
# ---------------------------------------------------------------------------


def test_fetch_famafrench_returns_tidy_decimal_frame(monkeypatch):
    period_index = pd.PeriodIndex(["2024-01", "2024-02"], freq="M", name="Date")
    three_factors = pd.DataFrame(
        {"Mkt-RF": [1.0, -2.0], "SMB": [0.5, 0.1], "HML": [-0.2, 0.3], "RF": [0.4, 0.4]},
        index=period_index,
    )
    mom_factor = pd.DataFrame({"WML": [2.0, 1.0]}, index=period_index)

    def fake_reader(name, source, start=None):
        assert source == "famafrench"
        if name == "Europe_3_Factors":
            return {0: three_factors}
        if name == "Europe_Mom_Factor":
            return {0: mom_factor}
        raise AssertionError(f"unexpected dataset {name}")

    monkeypatch.setattr(factors.web, "DataReader", fake_reader)

    df = factors.fetch_famafrench(start="2024-01-01")

    assert list(df.columns) == ["date", "mkt_rf", "smb", "hml", "mom", "rf"]
    assert len(df) == 2
    # decimals, not percent
    assert df["mkt_rf"].tolist() == pytest.approx([0.01, -0.02])
    assert df["mom"].tolist() == pytest.approx([0.02, 0.01])
    # month-end dates
    assert df["date"].tolist() == list(pd.to_datetime(["2024-01-31", "2024-02-29"]))


def test_fetch_famafrench_handles_mom_column_name(monkeypatch):
    period_index = pd.PeriodIndex(["2024-01"], freq="M", name="Date")
    three_factors = pd.DataFrame(
        {"Mkt-RF": [1.0], "SMB": [0.5], "HML": [-0.2], "RF": [0.4]}, index=period_index
    )
    mom_factor = pd.DataFrame({"Mom": [2.0]}, index=period_index)

    def fake_reader(name, source, start=None):
        if name == "Europe_3_Factors":
            return {0: three_factors}
        return {0: mom_factor}

    monkeypatch.setattr(factors.web, "DataReader", fake_reader)

    df = factors.fetch_famafrench(start="2024-01-01")
    assert df["mom"].tolist() == pytest.approx([0.02])


# ---------------------------------------------------------------------------
# Network smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_fetch_prices_one_aex_ticker_network():
    df, failed = prices.fetch_prices(["ASML"], start="2024-01-01", end="2024-02-01")
    assert failed == []
    assert not df.empty
    assert set(df.columns) == {"date", "ticker", "close", "adj_close", "volume"}


@pytest.mark.network
def test_fetch_fred_one_series_network():
    df = macro.fetch_fred(["INDPRO"])
    assert not df.empty
    assert set(df.columns) == {"date", "series_id", "value"}


@pytest.mark.network
def test_fetch_famafrench_network():
    df = factors.fetch_famafrench(start="2023-01-01")
    assert not df.empty
    assert list(df.columns) == ["date", "mkt_rf", "smb", "hml", "mom", "rf"]


@pytest.mark.network
def test_fetch_prices_full_aex_universe_resolution_network():
    """Reports which AEX-25 tickers fail to resolve via yfinance.

    Not a strict pass/fail gate on exact count; the goal is intelligence
    about which tickers are unreachable, printed for the report.
    """
    from stockpred.config import load_config

    config = load_config()
    df, failed = prices.fetch_prices(
        config.tickers, start="2024-01-01", end="2024-02-01", yf_suffix=config.yf_suffix
    )
    resolved = len(config.tickers) - len(failed)
    print(f"\nAEX resolution: {resolved}/{len(config.tickers)} resolved; failed={failed}")
    assert resolved >= 1  # sanity: at least some resolve

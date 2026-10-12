"""Economic indicators from FRED (keyless CSV endpoint)."""
from __future__ import annotations

import io

import pandas as pd
import requests

from pipeline.common import log, raw_dir, save_json

SERIES = {
    "UNRATE": "Unemployment rate (%)",
    "CPIAUCSL": "CPI, all items (index)",
    "UMCSENT": "UMich consumer sentiment",
    "A229RX0": "Real disposable personal income per capita",
    "GASREGW": "Regular gasoline, $/gal (weekly)",
    "PAYEMS": "Nonfarm payrolls (thousands)",
    "GDPC1": "Real GDP (quarterly)",
}


def fetch() -> dict:
    out = {}
    for sid, label in SERIES.items():
        try:
            # FRED's CSV endpoint stalls on custom user agents and on the cosd parameter, so use a plain request.
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            df = df[df["date"] >= "2019-01-01"]
            out[sid] = {"label": label, "dates": df["date"].tolist(), "values": df["value"].round(3).tolist()}
        except Exception as e:
            log.warning("fred %s failed: %s", sid, e)
    save_json(raw_dir("fred") / "fred.json", out)
    return out


def summarize(fr: dict) -> dict:
    """Headline indicators used by the fundamentals model (all changes vs. one year earlier)."""
    def series(sid):
        s = fr.get(sid)
        if not s:
            return None
        return pd.Series(s["values"], index=pd.to_datetime(s["dates"]))

    out = {}
    u = series("UNRATE")
    if u is not None:
        out["unemployment"] = float(u.iloc[-1])
        out["unemployment_change_12m"] = float(u.iloc[-1] - u.asof(u.index[-1] - pd.DateOffset(months=12)))
        out["unemployment_asof"] = u.index[-1].strftime("%Y-%m")
    c = series("CPIAUCSL")
    if c is not None:
        out["inflation_yoy"] = float(100 * (c.iloc[-1] / c.asof(c.index[-1] - pd.DateOffset(months=12)) - 1))
        out["inflation_asof"] = c.index[-1].strftime("%Y-%m")
    s = series("UMCSENT")
    if s is not None:
        out["consumer_sentiment"] = float(s.iloc[-1])
        out["sentiment_asof"] = s.index[-1].strftime("%Y-%m")
    rdi = series("A229RX0")
    if rdi is not None:
        out["rdi_growth_yoy"] = float(100 * (rdi.iloc[-1] / rdi.asof(rdi.index[-1] - pd.DateOffset(months=12)) - 1))
        out["rdi_asof"] = rdi.index[-1].strftime("%Y-%m")
    g = series("GASREGW")
    if g is not None:
        out["gas_price"] = float(g.iloc[-1])
        out["gas_change_12m"] = float(g.iloc[-1] - g.asof(g.index[-1] - pd.DateOffset(months=12)))
    gdp = series("GDPC1")
    if gdp is not None and len(gdp) > 2:
        out["gdp_growth_q_annualized"] = float(100 * ((gdp.iloc[-1] / gdp.iloc[-2]) ** 4 - 1))
        out["gdp_asof"] = gdp.index[-1].strftime("%Y-%m")
    p = series("PAYEMS")
    if p is not None:
        out["payrolls_3m_avg_change"] = float((p.iloc[-1] - p.iloc[-4]) / 3)
    return out

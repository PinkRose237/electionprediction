"""News monitoring: Google News RSS headlines per race and GDELT coverage tone.

Headlines are shown on the site and scanned for poll releases and candidate events. GDELT tone is
reported as a descriptive "news climate" indicator; it is deliberately *not* fed into the vote-margin
model, because media tone has no demonstrated out-of-sample predictive value for election margins.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET

from pipeline.common import Http, log, raw_dir, save_json

_http = Http(min_delay=1.0)


def google_news(query: str, max_items: int = 8, days: int = 10) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        r = _http.get(url, timeout=30, tries=3)
    except RuntimeError as e:
        log.warning("news: %s", e)
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    items = []
    for it in root.iter("item"):
        title = html.unescape(it.findtext("title") or "")
        src = it.find("source")
        pub = it.findtext("pubDate")
        try:
            when = email.utils.parsedate_to_datetime(pub).date().isoformat() if pub else None
        except (TypeError, ValueError):
            when = None
        items.append({"title": re.sub(r"\s+-\s+[^-]+$", "", title).strip(), "source": src.text if src is not None else "",
                      "url": it.findtext("link"), "date": when})
    items.sort(key=lambda x: x["date"] or "", reverse=True)
    return items[:max_items]


POLL_PAT = re.compile(r"\bpoll(s|ing)?\b|\bsurvey\b|\bleads?\b|\btrails?\b|\bpoints?\b", re.I)


def race_news(races: list[dict]) -> dict:
    """races: [{race_id, query}] -> {race_id: [headline...]}"""
    out = {}
    for r in races:
        out[r["race_id"]] = google_news(r["query"])
    out["NATIONAL"] = google_news("midterm elections 2026 House Senate", max_items=12, days=5)
    out["NATIONAL_POLLS"] = google_news("generic ballot poll Democrats Republicans", max_items=10, days=7)
    save_json(raw_dir("news") / "headlines.json", out)
    n = sum(len(v) for v in out.values())
    log.info("news: %d headlines for %d queries", n, len(out))
    return out


def gdelt_tone(queries: dict[str, str], timespan: str = "14d") -> dict:
    """Average GDELT document tone for each query (slow: GDELT allows ~1 request / 5 s)."""
    out = {}
    h = Http(min_delay=6.0)
    for key, q in queries.items():
        try:
            r = h.get("https://api.gdeltproject.org/api/v2/doc/doc",
                      params={"query": q, "mode": "timelinetone", "format": "json", "timespan": timespan}, tries=2)
            js = r.json()
            pts = [p["value"] for s in js.get("timeline", []) for p in s.get("data", [])]
            if pts:
                out[key] = {"tone": sum(pts) / len(pts), "n_days": len(pts)}
        except Exception as e:
            log.info("gdelt %s skipped: %s", key, e)
    save_json(raw_dir("news") / "gdelt_tone.json", out)
    return out

"""Prediction markets: Polymarket, Kalshi, PredictIt, Manifold.

Every market is reduced to a vig-free probability that the Democratic side (or a Democratic-backed
independent) wins, keyed by our race ids: 'GA-SEN', 'OH-SEN-SP', 'AZ-GOV', 'PA-10', and the special
keys 'HOUSE-CONTROL', 'SENATE-CONTROL', plus a national popular-vote ladder.
"""
from __future__ import annotations

import json
import re

from pipeline.common import Http, NAME_TO_ABBR, STATES, log, raw_dir, save_json

_http = Http(min_delay=0.25)
PM = "https://gamma-api.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mid(bid, ask, last):
    bid, ask, last = _f(bid), _f(ask), _f(last)
    if bid is not None and ask is not None and 0 < ask and ask - bid <= 0.10 and ask >= bid:
        return (bid + ask) / 2
    if last is not None and 0 < last < 1:
        return last
    if bid is not None and ask is not None and ask > 0:
        return (bid + ask) / 2
    return None


def _side(label: str) -> str | None:
    s = label.lower()
    if re.search(r"\(d\)|democrat", s):
        return "D"
    if re.search(r"\(r\)|republican", s):
        return "R"
    if re.search(r"\(i\)|independent", s):
        return "I"
    return None


# ---------------- Polymarket ----------------
def _pm_search(q: str) -> list[dict]:
    out = []
    for page in range(1, 6):
        r = _http.get(f"{PM}/public-search", params={"q": q, "limit_per_type": 50, "events_status": "active", "page": page}).json()
        evs = r.get("events", []) or []
        out += evs
        if not r.get("pagination", {}).get("hasMore"):
            break
    return out


def _pm_event(slug: str) -> dict | None:
    r = _http.get(f"{PM}/events", params={"slug": slug}).json()
    return r[0] if r else None


def _pm_binary_prob(markets: list[dict]) -> tuple[float | None, float]:
    """For a 'winner' event with one Yes/No market per candidate: P(D side) normalized."""
    probs = {"D": 0.0, "R": 0.0, "I": 0.0}
    vol = 0.0
    for m in markets:
        label = m.get("groupItemTitle") or m.get("question") or ""
        side = _side(label)
        if not side:
            continue
        p = _mid(m.get("bestBid"), m.get("bestAsk"), m.get("lastTradePrice"))
        if p is None:
            try:
                p = float(json.loads(m.get("outcomePrices") or "[]")[0])
            except (ValueError, IndexError, TypeError):
                p = None
        if p is None:
            continue
        probs[side] += p
        vol += _f(m.get("volume")) or 0
    tot = sum(probs.values())
    if tot <= 0:
        return None, vol
    return probs, vol


def polymarket() -> dict:
    out: dict[str, dict] = {}
    seen = set()
    queries = ["senate election winner", "governor winner 2026", "governor election winner",
               "which party will win the house", "which party will win the senate",
               "house election margin of victory", "house popular vote margin"]
    events = []
    for q in queries:
        try:
            events += _pm_search(q)
        except RuntimeError as e:
            log.warning("polymarket search failed: %s", e)
    for e in events:
        slug = e.get("slug", "")
        if slug in seen or e.get("closed"):
            continue
        seen.add(slug)
        title = e.get("title", "")
        rid = None
        m = re.match(r"^([A-Za-z ]+?) Senate Election Winner", title)
        if m and m.group(1).strip() in NAME_TO_ABBR:
            st = NAME_TO_ABBR[m.group(1).strip()]
            rid = f"{st}-SEN-SP" if st in ("OH", "FL") else f"{st}-SEN"
        m2 = re.match(r"^([A-Za-z ]+?) Governor Election Winner", title)
        if m2 and m2.group(1).strip() in NAME_TO_ABBR:
            rid = f"{NAME_TO_ABBR[m2.group(1).strip()]}-GOV"
        if "county" in title.lower() or "first round" in title.lower():
            rid = None
        if title.startswith("Which party will win the House in 2026"):
            rid = "HOUSE-CONTROL"
        if title.startswith("Which party will win the Senate in 2026"):
            rid = "SENATE-CONTROL"
        m3 = re.match(r"^([A-Z]{2})-(\d{2}|AL) House Election Margin of Victory", title)
        if m3:
            rid = f"{m3.group(1)}-{m3.group(2)}"
            mk = e.get("markets", [])
            pd_, pr_ = 0.0, 0.0
            for x in mk:
                lab = (x.get("groupItemTitle") or "").lower()
                p = _mid(x.get("bestBid"), x.get("bestAsk"), x.get("lastTradePrice"))
                if p is None:
                    continue
                if lab.startswith(("democrat", "d ")) or lab.startswith("dem"):
                    pd_ += p
                elif lab.startswith(("republican", "r ")) or lab.startswith("rep"):
                    pr_ += p
            if pd_ + pr_ > 0.5:
                out[rid] = {"p_dem": pd_ / (pd_ + pr_), "volume": sum(_f(x.get("volume")) or 0 for x in mk),
                            "url": f"https://polymarket.com/event/{slug}", "kind": "margin-buckets"}
            continue
        if "popular vote margin" in title.lower() and "house" in title.lower():
            ladder = []
            for x in e.get("markets", []):
                p = _mid(x.get("bestBid"), x.get("bestAsk"), x.get("lastTradePrice"))
                lab = x.get("groupItemTitle") or ""
                mm = re.match(r"(Democrats|Republicans) (\d+)(?:-(\d+))?%(\+)?", lab)
                if p is None or not mm:
                    continue
                lo, hi = float(mm.group(2)), float(mm.group(3)) if mm.group(3) else None
                sign = 1 if mm.group(1) == "Democrats" else -1
                if hi is None:
                    lo_m, hi_m = (lo, lo + 4) if sign > 0 else (-(lo + 4), -lo)
                else:
                    lo_m, hi_m = (lo, hi) if sign > 0 else (-hi, -lo)
                ladder.append({"lo": lo_m, "hi": hi_m, "p": p})
            if ladder:
                tot = sum(b["p"] for b in ladder)
                mean = sum((b["lo"] + b["hi"]) / 2 * b["p"] for b in ladder) / tot
                out["HOUSE-POPVOTE"] = {"buckets": ladder, "mean_margin": mean,
                                        "url": f"https://polymarket.com/event/{slug}", "kind": "popvote"}
            continue
        if not rid:
            continue
        probs, vol = _pm_binary_prob(e.get("markets", []))
        if not probs:
            # control markets use 'Democratic'/'Republican' party names
            continue
        if rid.endswith(("CONTROL",)) or True:
            d = probs["D"]
            r_ = probs["R"]
            if d + r_ <= 0 and probs["I"] <= 0:
                continue
            out[rid] = {"p_dem": d / (d + r_) if d + r_ > 0 else 0.0, "volume": vol,
                        "url": f"https://polymarket.com/event/{slug}", "raw": probs}
    log.info("polymarket: %d markets mapped", len(out))
    return out


# ---------------- Kalshi ----------------
def kalshi() -> dict:
    out: dict[str, dict] = {}
    try:
        series = _http.get(f"{KALSHI}/series", params={"category": "Elections"}).json().get("series", [])
    except RuntimeError as e:
        log.warning("kalshi series failed: %s", e)
        return out
    wanted = []
    for s in series:
        t = s.get("ticker", "")
        if t in ("CONTROLH", "CONTROLS"):
            wanted.append((t, "HOUSE-CONTROL" if t == "CONTROLH" else "SENATE-CONTROL"))
            continue
        m = re.match(r"^(?:SENATE|SENATEPARTY)([A-Z]{2})(S|SP|SPECIAL)?$", t)
        if m and m.group(1) in STATES:
            st = m.group(1)
            wanted.append((t, f"{st}-SEN-SP" if (m.group(2) or st in ("OH", "FL") and False) else f"{st}-SEN"))
            continue
        m = re.match(r"^GOVPARTY([A-Z]{2})$", t)
        if m and m.group(1) in STATES:
            wanted.append((t, f"{m.group(1)}-GOV"))
            continue
        m = re.match(r"^(?:KX)?HOUSE([A-Z]{2})(\d{1,2}|AL)$", t)
        if m and m.group(1) in STATES:
            n = m.group(2)
            wanted.append((t, f"{m.group(1)}-{'AL' if n == 'AL' else f'{int(n):02d}'}"))
    if any(t == "KXHOUSEPOPVOTEMARGIND" for t in (s.get("ticker") for s in series)):
        wanted.append(("KXHOUSEPOPVOTEMARGIND", "HOUSE-POPVOTE"))
    for ticker, rid in wanted:
        try:
            ms = _http.get(f"{KALSHI}/markets", params={"series_ticker": ticker, "status": "open", "limit": 100}).json().get("markets", [])
        except RuntimeError:
            continue
        ms = [m for m in ms if "-26" in m.get("ticker", "") or "-2026" in m.get("ticker", "") or "26NOV" in m.get("ticker", "")]
        if not ms:
            continue
        if rid == "HOUSE-POPVOTE":
            ladder = []
            for m in ms:
                mm = re.search(r"-AL(\d+)$", m["ticker"])
                p = _mid(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("last_price_dollars"))
                if mm and p is not None:
                    ladder.append({"threshold": float(mm.group(1)), "p_at_least": p})
            out[rid] = {"ladder": sorted(ladder, key=lambda x: x["threshold"]), "url": f"https://kalshi.com/markets/{ticker.lower()}",
                        "kind": "popvote"}
            continue
        if ms and ms[0].get("event_ticker", "").startswith(ticker) and ("special" in (ms[0].get("title") or "").lower()):
            rid = rid if rid.endswith("-SP") else rid + "-SP"
        pd_, pr_, pi_, vol = 0.0, 0.0, 0.0, 0.0
        for m in ms:
            side = m["ticker"].rsplit("-", 1)[-1]
            p = _mid(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("last_price_dollars"))
            if p is None:
                continue
            vol += _f(m.get("volume_fp") or m.get("volume")) or 0
            if side == "D":
                pd_ += p
            elif side == "R":
                pr_ += p
            elif side in ("I", "IND"):
                pi_ += p
        if pd_ + pr_ + pi_ > 0 and pd_ + pr_ > 0:
            out[rid] = {"p_dem": pd_ / (pd_ + pr_), "volume": vol, "url": f"https://kalshi.com/markets/{ticker.lower()}",
                        "raw": {"D": pd_, "R": pr_, "I": pi_}}
    log.info("kalshi: %d markets mapped", len(out))
    return out


# ---------------- PredictIt ----------------
_ORD = r"(\d+)(?:st|nd|rd|th)"


def predictit() -> dict:
    out: dict[str, dict] = {}
    try:
        data = _http.get("https://www.predictit.org/api/marketdata/all/").json()
    except RuntimeError as e:
        log.warning("predictit failed: %s", e)
        return out
    for mk in data.get("markets", []):
        name = mk.get("name", "")
        if "2026" not in name:
            continue
        rid = None
        if re.search(r"party will win the House in the 2026", name):
            rid = "HOUSE-CONTROL"
        elif re.search(r"control the Senate after the 2026", name):
            rid = "SENATE-CONTROL"
        else:
            m = re.search(r"2026 US Senate (special )?election in ([A-Za-z ]+?)\?", name)
            if m and m.group(2).strip() in NAME_TO_ABBR:
                st = NAME_TO_ABBR[m.group(2).strip()]
                rid = f"{st}-SEN-SP" if m.group(1) or (st in ("OH", "FL") and "special" in name.lower()) else f"{st}-SEN"
            m = re.search(r"governor of ([A-Za-z ]+?)\?", name) or re.search(r"2026 ([A-Za-z ]+?) gubernatorial", name)
            if m and m.group(1).strip() in NAME_TO_ABBR:
                rid = f"{NAME_TO_ABBR[m.group(1).strip()]}-GOV"
            m = re.search(r"in ([A-Za-z ]+?)'s " + _ORD + r" District", name)
            if m and m.group(1).strip() in NAME_TO_ABBR:
                rid = f"{NAME_TO_ABBR[m.group(1).strip()]}-{int(m.group(2)):02d}"
            m = re.search(r"in ([A-Za-z ]+?)'s at-large", name, re.I)
            if m and m.group(1).strip() in NAME_TO_ABBR:
                rid = f"{NAME_TO_ABBR[m.group(1).strip()]}-AL"
        if not rid:
            continue
        pd_, pr_, pi_ = 0.0, 0.0, 0.0
        for c in mk.get("contracts", []):
            p = _mid(c.get("bestBuyNoCost") and 1 - c["bestBuyNoCost"], c.get("bestBuyYesCost"), c.get("lastTradePrice"))
            side = _side(c.get("name", "")) or _side(c.get("shortName", ""))
            if p is None or not side:
                continue
            if side == "D":
                pd_ += p
            elif side == "R":
                pr_ += p
            else:
                pi_ += p
        if pd_ + pr_ > 0 or pi_ > 0:
            out[rid] = {"p_dem": pd_ / (pd_ + pr_) if pd_ + pr_ > 0 else 0.0, "url": mk.get("url"), "name": name,
                        "raw": {"D": pd_, "R": pr_, "I": pi_}}
    log.info("predictit: %d markets mapped", len(out))
    return out


# ---------------- Manifold ----------------
def manifold() -> dict:
    out = {}
    for rid, q in (("HOUSE-CONTROL", "which party will win the house 2026"), ("SENATE-CONTROL", "which party will win the senate 2026")):
        try:
            res = _http.get("https://api.manifold.markets/v0/search-markets",
                            params={"term": q, "limit": 10, "filter": "open", "sort": "liquidity"}).json()
        except RuntimeError:
            continue
        for m in res:
            qn = m.get("question", "").lower()
            if "2026" in qn and ("house" in qn if "HOUSE" in rid else "senate" in qn) and m.get("outcomeType") == "BINARY" \
                    and ("democrat" in qn or "dem" in qn):
                out[rid] = {"p_dem": m.get("probability"), "url": m.get("url"), "name": m.get("question")}
                break
    log.info("manifold: %d markets mapped", len(out))
    return out


def fetch_all() -> dict:
    res = {}
    for name, fn in (("Polymarket", polymarket), ("Kalshi", kalshi), ("PredictIt", predictit), ("Manifold", manifold)):
        try:
            res[name] = fn()
        except Exception as e:  # keep the pipeline alive if one exchange changes its API
            log.warning("%s failed: %s", name, e)
            res[name] = {}
    save_json(raw_dir("markets") / "markets.json", res)
    return res


if __name__ == "__main__":
    r = fetch_all()
    for k, v in r.items():
        print(k, len(v), {x: round(y["p_dem"], 3) for x, y in list(v.items())[:12] if "p_dem" in y})

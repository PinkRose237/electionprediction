"""Race-level model: builds every Senate, governor, and House race and blends four components.

  fundamentals : national environment + partisan lean (2 x Cook PVI) + incumbency + candidate
                 experience + scandals + fundraising (FEC) + independent-candidate adjustments
  polls        : adjusted poll average (see polls.py) with an error sd calibrated on 1998-2022
  ratings      : consensus of up to 10 forecasters/raters (Cook, Sabato, Inside Elections, Split
                 Ticket, DDHQ, The Economist, Silver Bulletin, FiftyPlusOne, RCP, Fox News)
  markets      : vig-free prices from Polymarket, Kalshi, and PredictIt

Components are combined by precision weighting; ratings and markets get a correlation discount
because both partly re-use the same polls.
"""
from __future__ import annotations

import math
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

from pipeline.common import DIVISION, STATES, clean, last_name, log, norm_name
from pipeline.fetch.wikipedia import RATING_MARGIN
from pipeline.model import polls as P

ND = NormalDist()

INCUMBENCY = {"SEN": 3.0, "GOV": 4.0, "HOUSE": 2.5}
APPOINTED_INCUMBENCY = 1.0
QUALITY_PER_POINT = {"SEN": 1.5, "GOV": 1.5, "HOUSE": 1.0}
SCANDAL_PER_LEVEL = 3.0
MONEY_COEF = {"SEN": 1.0, "GOV": 0.0, "HOUSE": 1.0}
MONEY_CAP = 4.0
IND_BONUS = 5.0           # Democratic-backed independents (e.g., Osborn 2024 ran ~13 pts ahead of Harris)
FUND_SD = {"SEN": 7.5, "GOV": 12.0, "HOUSE": 6.0}
# National environment (D-R, presidential-equivalent) in the year of an incumbent's last race, used to
# measure how far the incumbent ran ahead of or behind their state's partisanship.
NATIONAL_BY_YEAR = {2014: -5.7, 2016: 2.1, 2018: 8.6, 2020: 4.5, 2021: 4.5, 2022: -2.7, 2023: 0.0, 2024: -1.5, 2025: 0.0}
PERSONAL_VOTE_CARRY = 0.4    # share of past over-performance that carries into the next race
PERSONAL_VOTE_CAP = 60.0
RATING_SD = {"SEN": 7.0, "GOV": 7.5, "HOUSE": 6.5}
SAFE_RATING_SD = 14.0
MARKET_SD = {"SEN": 7.0, "GOV": 7.5, "HOUSE": 7.0}
MARKET_SCALE = 6.0        # converts a market probability to a margin: margin = scale * z(p)
CORR_DISCOUNT = {"fundamentals": 1.0, "polls": 1.0, "ratings": 0.6, "markets": 0.6}
MIN_RACE_SD = {"SEN": 3.0, "GOV": 3.5, "HOUSE": 3.5}
REDISTRICTED_INC_FACTOR = 0.6   # incumbents running on substantially redrawn 2025-26 maps


def rating_consensus(ratings: dict[str, str]) -> dict | None:
    vals = [RATING_MARGIN[r] for r in ratings.values() if r in RATING_MARGIN]
    if not vals:
        return None
    m = float(np.mean(vals))
    all_safe = all(abs(v) >= 25 for v in vals)
    label = rating_label(m)
    return {"margin": m, "sd": None, "n": len(vals), "all_safe": all_safe, "label": label}


def rating_label(m: float) -> str:
    a = abs(m)
    side = "D" if m > 0 else "R"
    if a < 1.5:
        return "Tossup"
    if a < 4.5:
        return f"Tilt {side}"
    if a < 9:
        return f"Lean {side}"
    if a < 18:
        return f"Likely {side}"
    return f"Safe {side}"


def prob_to_margin(p: float, scale: float = MARKET_SCALE) -> float:
    p = min(max(p, 0.03), 0.97)
    return scale * ND.inv_cdf(p)


def market_component(rid: str, markets: dict, kind: str, d_is_ind: bool = False) -> dict | None:
    vals, srcs = [], {}
    for ex, mk in (markets or {}).items():
        m = dict((mk or {}).get(rid) or {})
        raw = m.get("raw")
        if d_is_ind and raw and (raw.get("D", 0) + raw.get("I", 0) + raw.get("R", 0)) > 0:
            m["p_dem"] = (raw.get("D", 0) + raw.get("I", 0)) / (raw.get("D", 0) + raw.get("I", 0) + raw.get("R", 0))
        if m and m.get("p_dem") is not None and 0 < m["p_dem"] < 1:
            # thin markets get less say: weight by log volume when available
            vol = m.get("volume") or 0
            w = 1.0 + (math.log10(vol) - 4 if vol and vol > 1e4 else 0) * 0.25
            vals.append((m["p_dem"], max(w, 0.5)))
            srcs[ex] = round(m["p_dem"], 3)
    if not vals:
        return None
    p = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    return {"margin": prob_to_margin(p), "sd": MARKET_SD[kind], "p_dem": p, "sources": srcs}


def combine(components: dict[str, dict], kind: str) -> tuple[float, float, dict]:
    prec, num = {}, 0.0
    for k, c in components.items():
        if not c or c.get("margin") is None or not c.get("sd"):
            continue
        prec[k] = CORR_DISCOUNT.get(k, 0.6) / c["sd"] ** 2
    tot = sum(prec.values())
    mean = sum(prec[k] * components[k]["margin"] for k in prec) / tot
    sd = max(1.15 / math.sqrt(tot), MIN_RACE_SD[kind])
    weights = {k: round(v / tot, 3) for k, v in prec.items()}
    return mean, sd, weights


# ---------------- candidate selection ----------------
def pick_nominees(cands: list[dict], incumbent: str, inc_party: str, override: dict | None = None) -> tuple[dict | None, dict | None, list]:
    """Choose the Democratic-side and Republican-side nominees from a Wikipedia candidate list."""
    override = override or {}
    by = {"D": [], "R": [], "I": []}
    for c in cands:
        if c["party"] in by:
            by[c["party"]].append(c)
    inc_n = norm_name(incumbent or "")

    def choose(lst, name_hint):
        if not lst:
            return None
        if name_hint:
            for c in lst:   # exact full name first (Alaska lists two Republicans named Dan Sullivan)
                if norm_name(c["name"]) == norm_name(name_hint):
                    return c
            for c in lst:
                if last_name(c["name"]) == last_name(name_hint):
                    return c
        for c in lst:
            if inc_n and (norm_name(c["name"]) == inc_n or (last_name(c["name"]) == last_name(incumbent)
                                                            and norm_name(c["name"]).split()[0][:3] == inc_n.split()[0][:3])):
                return c
        return lst[0]

    d = choose(by["D"], override.get("dem"))
    r = choose(by["R"], override.get("rep"))
    if override.get("dem_is_independent"):
        d = choose(by["I"], override.get("dem")) or d
    others = [c for c in cands if c is not d and c is not r]
    return d, r, others


# ---------------- per-race fundamentals ----------------
def fundamentals(kind: str, N: float, pvi: float | None, inc_party: str | None, inc_running: bool,
                 appointed: bool, d_info: dict, r_info: dict, d_is_ind: bool, redistricted: bool) -> dict:
    lean = 2.0 * (pvi or 0.0)
    parts = {"national": N, "lean": lean}
    inc = 0.0
    if inc_running and inc_party in ("D", "R", "I"):
        size = APPOINTED_INCUMBENCY if appointed else INCUMBENCY[kind]
        if redistricted:
            size *= REDISTRICTED_INC_FACTOR
        inc = size if inc_party in ("D", "I") else -size
    parts["incumbency"] = inc
    pv = 0.0
    hist = d_info.get("_inc_history") or r_info.get("_inc_history")
    if hist and inc_running and not appointed and inc_party in ("D", "R"):
        yr, share, party = hist
        # winner's share -> approximate margin, allowing ~3 points for minor candidates; a plurality winner
        # still won, so the margin is at least half a point
        last_margin = max(2 * share - 97, 0.5) * (1 if party == "D" else -1)
        expected = 2.0 * (pvi or 0.0) + NATIONAL_BY_YEAR.get(yr, 0.0)
        over = max(-PERSONAL_VOTE_CAP, min(PERSONAL_VOTE_CAP, last_margin - expected))
        pv = PERSONAL_VOTE_CARRY * over
        # the generic incumbency bonus is part of past over-performance; don't count it twice
        pv -= inc * PERSONAL_VOTE_CARRY
    parts["personal_vote"] = pv
    qd, qr = d_info.get("experience"), r_info.get("experience")
    q = 0.0
    if qd is not None and qr is not None:
        # incumbents' advantage is already in the incumbency term, so cap experience at 2 for them
        qd2 = min(qd, 2) if d_info.get("is_incumbent") else qd
        qr2 = min(qr, 2) if r_info.get("is_incumbent") else qr
        q = QUALITY_PER_POINT[kind] * (qd2 - qr2)
    parts["candidate_quality"] = q
    sc = -SCANDAL_PER_LEVEL * (d_info.get("scandal") or 0) + SCANDAL_PER_LEVEL * (r_info.get("scandal") or 0)
    parts["scandals"] = sc
    money = 0.0
    if MONEY_COEF[kind] and d_info.get("receipts") is not None and r_info.get("receipts") is not None:
        money = MONEY_COEF[kind] * math.log((d_info["receipts"] + 1e5) / (r_info["receipts"] + 1e5))
        money = max(-MONEY_CAP, min(MONEY_CAP, money))
    parts["fundraising"] = money
    parts["independent_bonus"] = IND_BONUS if d_is_ind else 0.0
    margin = sum(parts.values())
    sd = FUND_SD[kind] + (2.5 if d_is_ind else 0.0)
    return {"margin": margin, "sd": sd, "parts": {k: round(v, 2) for k, v in parts.items()}}


def race_record(kind: str, rid: str, state: str, N: float, pvi, ratings: dict, markets: dict,
                pa: P.PollAverage | None, d_info: dict, r_info: dict, inc_party, inc_running, appointed,
                d_is_ind, redistricted=False, extra: dict | None = None) -> dict:
    comps = {}
    f = fundamentals(kind, N, pvi, inc_party, inc_running, appointed, d_info, r_info, d_is_ind, redistricted)
    comps["fundamentals"] = f
    if pa is not None and pa.margin is not None and pa.n_polls > 0:
        psd = P.poll_sd(kind, pa.n_eff, pa.undecided)
        if rid in P.RCV_RACES:      # first-round polls converted to an estimated final round
            psd = math.sqrt(psd ** 2 + 4.0 ** 2)
        comps["polls"] = {"margin": pa.margin, "sd": psd, "n": pa.n_polls,
                          "n_eff": round(pa.n_eff, 2), "last": pa.last_poll, "dem": pa.dem, "rep": pa.rep}
    rc = rating_consensus(ratings or {})
    if rc:
        rc["sd"] = SAFE_RATING_SD if rc["all_safe"] else RATING_SD[kind]
        comps["ratings"] = rc
    mc = market_component(rid, markets, kind, d_is_ind)
    if mc:
        comps["markets"] = mc
    mean, sd, weights = combine(comps, kind)
    rec = {"race_id": rid, "office": kind, "state": state, "division": DIVISION.get(state, "X"),
           "pvi": pvi, "lean": 2.0 * (pvi or 0.0), "mean": mean, "sd_race": sd, "weights": weights,
           "components": comps, "ratings": ratings, "rating_consensus": rc["label"] if rc else None,
           "dem": d_info, "rep": r_info, "inc_party": inc_party, "inc_running": inc_running,
           "dem_is_independent": d_is_ind}
    if extra:
        rec.update(extra)
    return rec


# ---------------- builders ----------------
def _cand_info(c: dict | None, research: dict | None, side: str, incumbent: str, fec_row: dict | None) -> dict:
    if not c:
        return {}
    info = {"name": c["name"], "party": c["party"]}
    info["is_incumbent"] = bool(incumbent) and last_name(incumbent) == last_name(c["name"])
    if research:
        e = research.get(f"{side}_experience")
        s = research.get(f"{side}_scandal")
        info["experience"] = _num(e)
        info["scandal"] = _num(s) or 0
        info["experience_detail"] = research.get(f"{side}_experience_detail") or None
    if info.get("experience") is None:
        info["experience"] = 3 if info["is_incumbent"] else None
    if fec_row:
        info["receipts"] = fec_row["receipts"]
        info["cash"] = fec_row["cash"]
        info["fec_through"] = fec_row["through"]
    elif research:
        for k_src, k_dst in ((f"{side}_receipts", "receipts"), (f"{side}_cash_on_hand", "cash")):
            v = _num(research.get(k_src))
            if v is not None:
                info[k_dst] = v
    return info


def _num(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            return None
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def appointed_incumbent(status: str, inc_text: str) -> bool:
    s = (status or "").lower() + " " + (inc_text or "").lower()
    return "appointee" in s or "appointed" in s or "interim" in s


def incumbent_running(status: str, cands: list[dict], incumbent: str) -> bool:
    s = (status or "").lower()
    if any(w in s for w in ("retiring", "lost", "term-limited", "term limited", "ineligible", "resigned", "died", "running for", "to run for")):
        return False
    if any(w in s for w in ("renominated", "running", "advanced", "nominated", "seeking")):
        return True
    return any(last_name(c["name"]) == last_name(incumbent or "") for c in cands)

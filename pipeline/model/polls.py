"""Poll processing and averaging.

Every poll is converted to a D-minus-R margin, then adjusted for:
  * partisan sponsorship   (538 archive 1998-2022: D-sponsored polls overstate D by ~4.7 pts, R by ~4.0)
  * pollster house effects (estimated from the 2026 generic ballot, shrunk toward zero)
  * sample population      (RV / adult samples shifted to a likely-voter basis)
  * national trend          (older state polls shifted by the change in the national environment since)
and weighted by recency, sample size, pollster quality, and pollster crowding.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.common import STATIC, clean, log, norm_name

RCV_RACES = {"AK-SEN", "AK-GOV", "AK-AL"}   # top-four primary + ranked-choice general
PARTISAN_ADJ = 4.5          # points toward the non-sponsoring party
PARTISAN_WEIGHT = 0.5       # partisan polls count half
PARTISAN_ADJ_KNOWN_FIRM = 2.0  # tagged polls from firms with an estimated house effect
HOUSE_EFFECT_TRANSFER = 0.5  # share of a pollster's national house effect applied to its state polls
DEFAULT_N = 600


# ---------------- pollster ratings (538 archive) ----------------
_RATINGS: dict[str, float] | None = None


def _pollster_key(name: str) -> str:
    s = norm_name(name)
    s = re.sub(r"\b(the|polling|poll|polls|research|group|inc|llc|strategies|university|college|institute|"
               r"associates|analytics|insights|company|co|survey|surveys|center|for|of|and|public|opinion|news)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pollster_grade(name: str) -> float | None:
    """538 numeric grade (0.5-3.0) if the pollster appears in the archive."""
    global _RATINGS
    if _RATINGS is None:
        _RATINGS = {}
        try:
            pr = pd.read_csv(STATIC / "fte" / "pollster-ratings-combined.csv")
            for _, r in pr.iterrows():
                if pd.notna(r.get("numeric_grade")):
                    for part in re.split(r"/", str(r["pollster"])):
                        k = _pollster_key(part)
                        if k and k not in _RATINGS:
                            _RATINGS[k] = float(r["numeric_grade"])
                    _RATINGS[_pollster_key(r["pollster"])] = float(r["numeric_grade"])
        except FileNotFoundError:
            pass
    k = _pollster_key(name)
    if not k:
        return None
    if k in _RATINGS:
        return _RATINGS[k]
    # partial match on first two tokens (e.g., "Emerson" -> "Emerson College")
    head = " ".join(k.split()[:2])
    for kk, v in _RATINGS.items():
        if kk.startswith(head) or head.startswith(kk) and len(kk) > 3:
            return v
    for part in re.split(r"/| for | and ", name):
        kp = _pollster_key(part)
        if kp in _RATINGS:
            return _RATINGS[kp]
    return None


_GENERIC_TOKENS = {"poll", "polls", "polling", "research", "strategies", "group", "university", "college", "news",
                   "insights", "analytics", "survey", "public", "opinion", "institute", "center", "partners", "media"}


def _name_tokens(name: str) -> set[str]:
    n = norm_name(name)
    toks = {t for t in n.split() if len(t) >= 4 and t not in _GENERIC_TOKENS}
    toks.add(n.replace(" ", "")[:12])
    return toks


def quality_weight(name: str) -> float:
    g = pollster_grade(name)
    if g is None:
        return 0.65
    return 0.35 + 0.22 * g   # grade 3.0 -> 1.01, 2.0 -> 0.79, 1.0 -> 0.57, 0.5 -> 0.46


# ---------------- cleaning ----------------
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts mixed-source poll rows, returns one row per poll with canonical columns."""
    if df.empty:
        return df
    df = df.copy()
    for c in ("dem", "rep", "n"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df.get("start"), errors="coerce").fillna(df["end"])
    df = df.dropna(subset=["dem", "rep", "end"])
    df = df[(df["dem"] + df["rep"] >= 50) & (df["dem"] + df["rep"] <= 101)]
    df["pollster"] = df["pollster"].map(lambda s: re.sub(r"\s+", " ", clean(s)))
    df["pkey"] = df["pollster"].map(_pollster_key)
    df["pop"] = df.get("pop", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.upper().str.strip()
    df.loc[~df["pop"].isin(["LV", "RV", "A"]), "pop"] = "RV"
    sp = df.get("sponsor_party", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.upper().str[:1]
    df["sponsor_party"] = sp.where(sp.isin(["D", "R"]), "")
    df["margin_raw"] = df["dem"] - df["rep"]
    df["variant"] = df.get("variant", False)
    df["variant"] = df["variant"].fillna(False).astype(bool)
    # ---- de-duplicate across sources (Wikipedia, VoteHub, Silver Bulletin, 270toWin, research) ----
    # The same survey arrives with different pollster spellings, field-end vs. release dates, and several
    # versions (LV/RV, with/without leaners, ranked-choice rounds). Rules, for polls of the same race whose
    # pollster names overlap:
    #   end dates within 1 day                              -> versions of one survey
    #   within 3 days and both shares within 3.5 points     -> same poll from two sources
    #   within 10 days and both shares within 1 point       -> same poll reported with its release date
    # The kept version: likely voters first; for ranked-choice races the final round (largest D+R);
    # otherwise the headline (non-footnoted) version from the best-documented source.
    src_pref = {"Wikipedia": 0, "VoteHub": 1, "Silver Bulletin": 1, "270toWin": 2, "Research": 3}
    rcv = df["race_id"].isin(RCV_RACES)
    df["_p"] = df["pop"].map({"LV": 0, "RV": 1, "A": 2})
    df["_s"] = df.get("source", pd.Series("", index=df.index)).map(lambda s: src_pref.get(s, 3))
    df["_v"] = np.where(rcv, 0, df["variant"].astype(int))
    df["_t"] = np.where(rcv, -(df["dem"] + df["rep"]), 0)
    nans = df["answers"].map(lambda a: len(a) if isinstance(a, list) else 0) if "answers" in df.columns else 0
    df["_k"] = np.where(rcv, 0, -nans)   # full-field version first: it matches the actual ballot
    df["_n"] = -df["n"].fillna(0)
    df = df.sort_values(["_p", "_t", "_k", "_v", "_s", "_n"])
    keep, kept = [], {}
    for idx, r in df.iterrows():
        toks = _name_tokens(r["pollster"])
        dup = False
        for k in kept.get(r["race_id"], []):
            if not (toks & k["toks"]):
                continue
            dd = abs((r["end"] - k["end"]).days)
            close = max(abs(r["dem"] - k["dem"]), abs(r["rep"] - k["rep"]))
            if dd <= 1 or (dd <= 3 and close <= 3.5) or (dd <= 10 and close <= 1.0):
                dup = True
                k["sources"].add(r.get("source", ""))
                break
        if dup:
            continue
        kept.setdefault(r["race_id"], []).append({"end": r["end"], "toks": toks, "dem": r["dem"], "rep": r["rep"],
                                                  "idx": idx, "sources": {r.get("source", "")}})
        keep.append(idx)
    srcs = {k["idx"]: ", ".join(sorted(x for x in k["sources"] if x)) for lst in kept.values() for k in lst}
    df = df.loc[keep].drop(columns=["_p", "_v", "_s", "_n", "_t", "_k"])
    df["sources"] = [srcs.get(i, "") for i in df.index]
    df["start"] = df["start"].where(df["start"] < df["end"], df["end"] - pd.Timedelta(days=2))
    return df.reset_index(drop=True)


# ---------------- house effects and LV adjustment (from generic ballot) ----------------
@dataclass
class Adjusters:
    house: dict[str, float]
    lv_shift: dict[str, float]   # added to RV/A margins to convert to LV basis
    gb_series: pd.Series | None = None  # daily national average (for trend adjustment)

    def trend_shift(self, poll_end: pd.Timestamp, asof: pd.Timestamp) -> float:
        s = self.gb_series
        if s is None or s.empty:
            return 0.0
        a = s.asof(asof) if asof >= s.index[0] else s.iloc[0]
        b = s.asof(poll_end) if poll_end >= s.index[0] else s.iloc[0]
        if pd.isna(a) or pd.isna(b):
            return 0.0
        return float(a - b)


def estimate_lv_shift(gb: pd.DataFrame) -> dict[str, float]:
    """Paired comparisons: same pollster & field dates publishing both LV and RV (or A) numbers."""
    out = {"LV": 0.0, "RV": 0.0, "A": 0.0}
    if gb.empty:
        return out
    g = gb.copy()
    g["margin"] = g["dem"] - g["rep"]
    diffs = {"RV": [], "A": []}
    for (pk, end), grp in g.groupby(["pkey", "end"]):
        if "LV" in set(grp["pop"]):
            lv = grp[grp["pop"] == "LV"]["margin"].mean()
            for p in ("RV", "A"):
                if p in set(grp["pop"]):
                    diffs[p].append(lv - grp[grp["pop"] == p]["margin"].mean())
    for p in ("RV", "A"):
        if diffs[p]:
            k = len(diffs[p])
            # shrink toward a prior of 0 (no LV gap) with strength 4 pairs
            out[p] = float(np.mean(diffs[p]) * k / (k + 4))
    log.info("LV shift estimates: %s (pairs RV=%d A=%d)", {k: round(v, 2) for k, v in out.items()}, len(diffs["RV"]), len(diffs["A"]))
    return out


def estimate_house_effects(gb: pd.DataFrame, lv_shift: dict[str, float], asof: pd.Timestamp) -> dict[str, float]:
    """House effect = pollster's mean residual from the all-pollster trend, shrunk with k/(k+4)."""
    if gb.empty:
        return {}
    g = gb.copy()
    g["m"] = g["dem"] - g["rep"] + g["pop"].map(lv_shift).fillna(0)
    g = g[g["sponsor_party"] == ""]
    g = g.sort_values("end")
    # crude trend: centered 21-day rolling median of all polls
    trend = []
    for _, r in g.iterrows():
        win = g[(g["end"] >= r["end"] - pd.Timedelta(days=14)) & (g["end"] <= r["end"] + pd.Timedelta(days=14))]
        trend.append(win["m"].median())
    g["resid"] = g["m"] - np.array(trend)
    he = {}
    for pk, grp in g.groupby("pkey"):
        k = len(grp)
        he[pk] = float(grp["resid"].mean() * k / (k + 4))
    return he


# ---------------- averaging ----------------
@dataclass
class PollAverage:
    margin: float | None
    dem: float | None
    rep: float | None
    n_polls: int
    n_eff: float
    undecided: float | None
    last_poll: str | None
    polls: list[dict]


def average(polls: pd.DataFrame, asof: dt.date, adj: Adjusters | None = None, half_life: float = 18.0,
            trend: bool = True, max_age_days: int = 120, he_transfer: float = HOUSE_EFFECT_TRANSFER) -> PollAverage:
    if polls is None or polls.empty:
        return PollAverage(None, None, None, 0, 0.0, None, None, [])
    asof_ts = pd.Timestamp(asof)
    p = polls[(polls["end"] <= asof_ts) & (polls["end"] >= asof_ts - pd.Timedelta(days=max_age_days))].copy()
    if p.empty:
        return PollAverage(None, None, None, 0, 0.0, None, None, [])
    adj = adj or Adjusters({}, {"LV": 0, "RV": 0, "A": 0})
    m = p["dem"] - p["rep"]
    m = m + p["pop"].map(adj.lv_shift).fillna(0.0)
    m = m - p["pkey"].map(adj.house).fillna(0.0) * he_transfer
    # Partisan tags: a full sponsor adjustment, except for firms whose lean is already measured as a house
    # effect (Wikipedia also tags R/D-aligned firms that are not polling for a campaign)
    known = p["pkey"].isin(set(adj.house)) if adj.house else pd.Series(False, index=p.index)
    amt = np.where(known, PARTISAN_ADJ_KNOWN_FIRM, PARTISAN_ADJ)
    m = m - np.where(p["sponsor_party"] == "D", amt, 0.0) + np.where(p["sponsor_party"] == "R", amt, 0.0)
    if trend:
        m = m + p["end"].map(lambda e: adj.trend_shift(e, asof_ts) * 0.8)
    mid = p["start"] + (p["end"] - p["start"]) / 2
    age = (asof_ts - mid).dt.days.clip(lower=0)
    w = 0.5 ** (age / half_life)
    w = w * np.sqrt(p["n"].fillna(DEFAULT_N).clip(150, 2500) / DEFAULT_N)
    w = w * p["pollster"].map(quality_weight)
    w = w * np.where(p["sponsor_party"] != "", np.where(known, 0.75, PARTISAN_WEIGHT), 1.0)
    # crowding: a pollster's polls share weight (sqrt of count)
    cnt = p.groupby("pkey")["pkey"].transform("count")
    w = w / np.sqrt(cnt)
    p["adj_margin"] = m
    p["weight"] = w
    tot = w.sum()
    if tot <= 0:
        return PollAverage(None, None, None, 0, 0.0, None, None, [])
    marg = float((w * m).sum() / tot)
    n_eff = float(tot ** 2 / (w ** 2).sum())
    # n_eff is "effective independent polls" only up to the recency discount; scale by mean recency
    dem = float((w * p["dem"]).sum() / tot)
    rep = float((w * p["rep"]).sum() / tot)
    und = max(0.0, 100 - dem - rep - float((w * p.get("other", pd.Series(0, index=p.index)).fillna(0)).sum() / tot))
    p = p.sort_values("end", ascending=False)
    recs = []
    for _, r in p.iterrows():
        recs.append({"pollster": r["pollster"], "end": r["end"].date().isoformat(),
                     "start": r["start"].date().isoformat() if pd.notna(r["start"]) else None,
                     "n": None if pd.isna(r["n"]) else int(r["n"]), "pop": r["pop"],
                     "dem": float(r["dem"]), "rep": float(r["rep"]),
                     "sponsor": r["sponsor_party"] or None, "adj": round(float(r["adj_margin"]), 1),
                     "weight": round(float(r["weight"] / tot), 3), "grade": pollster_grade(r["pollster"]),
                     "url": r.get("url") or r.get("source_url")})
    return PollAverage(marg, dem, rep, len(p), n_eff, und, p["end"].max().date().isoformat(), recs)


def daily_series(polls: pd.DataFrame, start: dt.date, end: dt.date, adj: Adjusters, half_life: float = 12.0) -> pd.Series:
    """Daily national average (no trend adjustment) for charts and for trend-adjusting state polls."""
    out = {}
    d = start
    while d <= end:
        a = average(polls, d, adj, half_life=half_life, trend=False, max_age_days=60, he_transfer=1.0)
        if a.margin is not None:
            out[pd.Timestamp(d)] = a.margin
        d += dt.timedelta(days=1)
    s = pd.Series(out).sort_index()
    return s.rolling(3, min_periods=1, center=True).mean() if len(s) else s


def poll_sd(kind: str, n_eff: float, undecided: float | None) -> float:
    """Race-specific (non-national) error sd of a poll average, calibrated on the 538 archive of
    competitive races 1998-2022 (residual after removing each cycle's common error)."""
    base, per = {"SEN": (3.3, 6.5), "GOV": (4.5, 7.0), "HOUSE": (4.5, 6.0)}[kind]
    n = max(n_eff, 0.35)
    var = base ** 2 + per ** 2 / n
    if undecided:
        var += (0.15 * max(0.0, undecided - 4)) ** 2
    return math.sqrt(var)

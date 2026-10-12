"""Automated poll feeds (all public, no keys):

  * Silver Bulletin's published generic-ballot poll sheet (Google Sheets CSV, updated daily)
  * VoteHub's open polls API (generic ballot, approval, Senate, governor, House)
  * 270toWin state poll pages for Senate and governor races

Wikipedia race-page polls are parsed separately in wikipedia.py. FiftyPlusOne is deliberately not
scraped: its API rejects non-browser clients, and we respect that.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests
from lxml import html as lhtml

from pipeline.common import NAME_TO_ABBR, STATES, Http, clean, last_name, log, norm_name, raw_dir, save_json

SB_GENERIC = ("https://docs.google.com/spreadsheets/d/e/2PACX-1vRsvXNCZ0ubJr8D_yNcU5q6C0_HBa35K7oDK03KpO7Ca43"
              "UwdXaIdvVLWoXEmHHph0EREz5430Hm5yZ/pub?output=csv")
VOTEHUB = "https://api.votehub.com/polls"
T270 = "https://www.270towin.com/2026-{kind}-polls/{slug}"

_BROWSERISH = {"User-Agent": "Mozilla/5.0 (compatible; MidtermForecastModel/1.0; personal research)"}


def _date(s):
    d = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(d) else d.date().isoformat()


# ---------------- Silver Bulletin generic ballot ----------------
def silver_bulletin_generic() -> list[dict]:
    r = requests.get(SB_GENERIC, headers=_BROWSERISH, timeout=60)
    r.raise_for_status()
    (raw_dir("pollfeeds") / "silver_bulletin_generic.csv").write_text(r.text)
    df = pd.read_csv(io.StringIO(r.text))
    if "subgroup" in df.columns:
        df = df[df["subgroup"].astype(str).str.lower().isin(["all polls", "all"])] if (df["subgroup"].astype(str).str.lower() == "all polls").any() else df
    out = []
    for _, x in df.iterrows():
        partisan = str(x.get("partisan") or "").strip().upper()[:1]
        out.append({"race_id": "US-GB", "pollster": clean(x.get("pollster")), "sponsor": clean(x.get("sponsors")),
                    "sponsor_party": partisan if partisan in ("D", "R") else "",
                    "start": _date(x.get("startdate")), "end": _date(x.get("enddate")),
                    "n": x.get("samplesize"), "pop": str(x.get("population") or "").upper(),
                    "dem": x.get("dem"), "rep": x.get("rep"), "url": x.get("url"), "source": "Silver Bulletin"})
    log.info("silver bulletin generic sheet: %d rows", len(out))
    return out


# ---------------- VoteHub ----------------
def votehub_raw() -> list[dict]:
    r = requests.get(VOTEHUB, headers=_BROWSERISH, timeout=90)
    r.raise_for_status()
    data = r.json()
    save_json(raw_dir("pollfeeds") / "votehub_polls.json", data)
    log.info("votehub: %d polls", len(data))
    return data


def _vh_base(p: dict) -> dict:
    pop = (p.get("population") or "").upper()
    part = (p.get("partisan") or "").upper()
    return {"pollster": p.get("pollster"), "sponsor": "; ".join(p.get("sponsors") or []),
            "sponsor_party": "D" if part.startswith("D") else "R" if part.startswith("R") else "",
            "internal": bool(p.get("internal")), "start": p.get("start_date"), "end": p.get("end_date"),
            "n": p.get("sample_size"), "pop": pop, "url": p.get("url"), "source": "VoteHub"}


def votehub_generic(data: list[dict]) -> list[dict]:
    out = []
    for p in data:
        if p.get("poll_type") != "generic-ballot":
            continue
        ans = {a["choice"].lower(): a["pct"] for a in p.get("answers", [])}
        d = next((v for k, v in ans.items() if k.startswith("dem")), None)
        r = next((v for k, v in ans.items() if k.startswith("rep")), None)
        if d is None or r is None:
            continue
        out.append({**_vh_base(p), "race_id": "US-GB", "dem": d, "rep": r})
    return out


def votehub_approval(data: list[dict]) -> list[dict]:
    out = []
    for p in data:
        if p.get("poll_type") != "approval":
            continue
        subj = (p.get("subject") or "").lower()
        if "trump" not in subj:
            continue
        ans = {a["choice"].lower(): a["pct"] for a in p.get("answers", [])}
        a, d = ans.get("approve"), ans.get("disapprove")
        if a is None or d is None:
            continue
        out.append({**_vh_base(p), "approve": a, "disapprove": d})
    return out


def _match(answers: list[tuple[str, float]], dem: str, rep: str) -> tuple[float, float, float, list] | None:
    """Pick the nominees' shares from a poll's answer list. When several answers share a nominee's surname
    (Alaska has two "Dan Sullivan"s on the ballot), an exact full-name match wins, else the larger share."""
    def toks(name):
        return norm_name(re.sub(r"\(.*?\)", "", name)).split()
    full_d, full_r = norm_name(dem), norm_name(rep)
    dl, rl = last_name(dem), last_name(rep)
    rows = [(i, n, p) for i, (n, p) in enumerate(answers) if p is not None]

    def best(full, last, exclude=None):
        exact = [x for x in rows if norm_name(re.sub(r"\(.*?\)", "", x[1])) == full and x[0] != exclude]
        if exact:
            return exact[0]
        hits = [x for x in rows if last and last in toks(x[1]) and x[0] != exclude]
        return max(hits, key=lambda x: x[2]) if hits else None

    d = best(full_d, dl)
    r = best(full_r, rl, exclude=d[0] if d else None)
    if not d or not r:
        return None
    other = sum(p for i, n, p in rows if i not in (d[0], r[0]))
    d, r = (d[1], d[2]), (r[1], r[2])
    return float(d[1]), float(r[1]), other, [[n, None, p] for _, n, p in rows]


def votehub_races(data: list[dict], nominees: dict[str, tuple[str, str]]) -> list[dict]:
    """nominees: {race_id: (dem_side_name, rep_name)} for SEN/GOV and House districts."""
    out = []
    for p in data:
        t = p.get("poll_type")
        subj = p.get("subject") or ""
        rid = None
        if t in ("us-senator", "governor"):
            m = re.match(r"^2026 (.+?)( special)?$", subj, re.I)
            if not m or m.group(1) not in NAME_TO_ABBR:
                continue  # primaries are labeled e.g. "2026 Texas Democratic"
            st = NAME_TO_ABBR[m.group(1)]
            if t == "governor":
                rid = f"{st}-GOV"
            else:
                rid = f"{st}-SEN-SP" if st in ("OH", "FL") else f"{st}-SEN"
        elif t == "us-representative":
            seat = p.get("seat_name") or subj
            m = re.search(r"([A-Z]{2})[- ]?(\d{1,2}|AL)\b", seat or "")
            if m and m.group(1) in STATES:
                n = m.group(2)
                rid = f"{m.group(1)}-{'AL' if n == 'AL' else f'{int(n):02d}'}"
        if not rid or rid not in nominees:
            continue
        dem, rep = nominees[rid]
        mt = _match([(a["choice"], a["pct"]) for a in p.get("answers", [])], dem, rep)
        if not mt:
            continue
        out.append({**_vh_base(p), "race_id": rid, "dem": mt[0], "rep": mt[1], "other": mt[2], "answers": mt[3]})
    log.info("votehub: %d race polls matched to nominees", len(out))
    return out


# ---------------- 270toWin ----------------
def t270_races(nominees: dict[str, tuple[str, str]]) -> list[dict]:
    h = Http(min_delay=1.5)
    out = []
    pages = {}
    for rid in nominees:
        if not re.match(r"^[A-Z]{2}-(SEN|SEN-SP|GOV)$", rid):
            continue
        st = rid[:2]
        kind = "governor" if rid.endswith("GOV") else "senate"
        slug = STATES[st].lower().replace(" ", "-")
        pages.setdefault((kind, slug), []).append(rid)
    for (kind, slug), rids in pages.items():
        url = T270.format(kind=kind, slug=slug)
        cache = raw_dir("pollfeeds") / f"270_{kind}_{slug}.html"
        try:
            if cache.exists():
                text = cache.read_text()
            else:
                text = h.get(url, timeout=30, tries=2).text
                cache.write_text(text)
        except Exception as e:
            log.info("270toWin %s/%s: %s", kind, slug, e)
            continue
        doc = lhtml.fromstring(text)
        for tbl in doc.xpath("//table[@id='polls' or contains(@class,'polls')]"):
            heads = tbl.xpath(".//th[contains(@class,'can_name')]")
            cand_ids = [th.get("candidate_id") for th in heads]
            cand_names = [clean(th.text_content()).rstrip("*") for th in heads]
            for tr in tbl.xpath(".//tr[contains(@class,'poll_row')]"):
                src = tr.xpath(".//td[contains(@class,'poll_src')]")
                date = tr.xpath(".//td[contains(@class,'poll_date')]")
                samp = tr.xpath(".//td[contains(@class,'poll_sample')]")
                cells = tr.xpath(".//td[contains(@class,'poll_data')]")
                if not src or not date or len(cells) != len(cand_names):
                    continue
                link = src[0].xpath(".//a/@href")
                pollster = clean(src[0].text_content())
                vals = []
                for c in cells:
                    m = re.search(r"(\d+(?:\.\d+)?)\s*%", c.text_content())
                    vals.append(float(m.group(1)) if m else None)
                s = clean(samp[0].text_content()) if samp else ""
                mn = re.search(r"([\d,]+)\s*(LV|RV|A)\b", s)
                for rid in rids:
                    dem, rep = nominees[rid]
                    mt = _match(list(zip(cand_names, vals)), dem, rep)
                    if not mt:
                        continue
                    sp = re.search(r"\((D|R)\)", pollster)
                    out.append({"race_id": rid, "pollster": re.sub(r"\s*\((D|R)\)\s*", " ", pollster).strip(),
                                "sponsor_party": sp.group(1) if sp else "", "start": None,
                                "end": _date(clean(date[0].text_content())),
                                "n": int(mn.group(1).replace(",", "")) if mn else None,
                                "pop": mn.group(2) if mn else "", "dem": mt[0], "rep": mt[1], "other": mt[2], "answers": mt[3],
                                "url": link[0] if link else url, "source": "270toWin", "source_url": url})
    log.info("270toWin: %d race polls matched", len(out))
    return out

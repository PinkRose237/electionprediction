"""Wikipedia scrapers: race rosters, forecaster ratings, district PVIs, and poll tables.

Uses the cached REST endpoint (/api/rest_v1/page/html/) which is far less rate-limited than
action=parse. Each page is fetched once per day and cached under data/raw/<date>/wikipedia/.
"""
from __future__ import annotations

import io
import re
import urllib.parse
import warnings

import pandas as pd
from lxml import html as lhtml

from pipeline.common import (Http, NAME_TO_ABBR, STATES, clean, last_name, log, norm_name,
                             parse_date_range, pct, raw_dir)

warnings.filterwarnings("ignore")
_http = Http(min_delay=1.0)

SENATE_PAGE = "2026 United States Senate elections"
GOV_PAGE = "2026 United States gubernatorial elections"
HOUSE_PAGE = "2026 United States House of Representatives elections"
HOUSE_RATINGS_PAGE = "2026 United States House of Representatives election ratings"

# Forecaster column labels on Wikipedia -> canonical short names
FORECASTERS = {
    "cook": "Cook", "ie": "Inside Elections", "inside": "Inside Elections", "sabato": "Sabato",
    "ddhq": "DDHQ", "econ": "Economist", "fpo": "FiftyPlusOne", "fox": "Fox News", "rcp": "RCP",
    "silver": "Silver Bulletin", "st": "Split Ticket", "cnalysis": "CNalysis", "cnn": "CNN",
    "rttwh": "Race to the WH", "race": "Race to the WH", "elections daily": "Elections Daily",
    "ed": "Elections Daily", "splitticket": "Split Ticket", "kalshi": "Kalshi", "polymarket": "Polymarket",
    "lean tossup": "Lean Tossup", "nyt": "NYT", "jhk": "JHK", "votehub": "VoteHub", "sis": "Strength In Numbers",
}


def page_html(title: str) -> str:
    cache = raw_dir("wikipedia") / (re.sub(r"[^A-Za-z0-9]+", "_", title) + ".html")
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_text()
    url = "https://en.wikipedia.org/api/rest_v1/page/html/" + urllib.parse.quote(title.replace(" ", "_"), safe="")
    r = _http.get(url)
    cache.write_text(r.text)
    return r.text


def page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def tables_with_sections(title: str):
    """Yield (section_path list[str], DataFrame, table_element) for every wikitable on the page."""
    doc = lhtml.fromstring(page_html(title))
    for tbl in doc.xpath("//table[contains(concat(' ', normalize-space(@class), ' '), ' wikitable ')]"):
        path = []
        for sec in tbl.iterancestors("section"):
            hs = sec.xpath("./h1|./h2|./h3|./h4|./h5|./div[contains(@class,'mw-heading')]/*[self::h2 or self::h3 or self::h4 or self::h5]")
            if hs:
                path.append(clean(hs[0].text_content()))
        path.reverse()
        # drop nested tables before parsing (they confuse read_html)
        for inner in tbl.xpath(".//table"):
            inner.getparent().remove(inner)
        try:
            df = pd.read_html(io.StringIO(lhtml.tostring(tbl, encoding="unicode")))[0]
        except ValueError:
            continue
        yield path, df, tbl


def _flat_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            parts = [clean(x) for x in c if not str(x).startswith("Unnamed")]
            # de-duplicate repeated header levels
            dedup = []
            for p in parts:
                if p and p not in dedup:
                    dedup.append(p)
            cols.append(" | ".join(dedup))
        else:
            cols.append(clean(c))
    return cols


def parse_pvi(s) -> float | None:
    """Cook PVI string -> signed number (D positive). 'R+5' -> -5, 'EVEN' -> 0."""
    s = clean(s).upper()
    if not s:
        return None
    if s.startswith("EVEN"):
        return 0.0
    m = re.match(r"([DR])\s*\+\s*(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    v = float(m.group(2))
    return v if m.group(1) == "D" else -v


def normalize_rating(s: str) -> str | None:
    """Map forecaster rating text to a canonical scale."""
    s = clean(s).lower()
    if not s or s in ("nan", "—", "–", "-"):
        return None
    s = s.replace("(flip)", "").replace("toss-up", "tossup").replace("toss up", "tossup").strip()
    side = "D" if re.search(r"\bd\b|dem", s) else ("R" if re.search(r"\br\b|rep", s) else ("I" if re.search(r"\bi\b|ind", s) else None))
    if "tossup" in s and "tilt" not in s and "lean" not in s:
        return "Tossup"
    for lvl, words in (("Safe", ("safe", "solid")), ("Likely", ("likely",)), ("Lean", ("lean",)), ("Tilt", ("tilt",))):
        if any(w in s for w in words) and side:
            return f"{lvl} {side}"
    return None


# Rating -> expected D-minus-R margin (points). Calibrated loosely on historical outcomes of
# rated races (Cook/Sabato/IE): Safe ≈ 25+, Likely ≈ 12, Lean ≈ 6, Tilt ≈ 3.
RATING_MARGIN = {"Safe D": 25.0, "Likely D": 12.0, "Lean D": 6.0, "Tilt D": 3.0, "Tossup": 0.0,
                 "Tilt R": -3.0, "Lean R": -6.0, "Likely R": -12.0, "Safe R": -25.0,
                 # independents backed by Democrats are coded on the D side
                 "Safe I": 25.0, "Likely I": 12.0, "Lean I": 6.0, "Tilt I": 3.0}


def _forecaster_name(col: str) -> str | None:
    c = col.split("|")[-1].strip()
    head = re.sub(r"\s+\w{3,9}\.? \d{1,2},? \d{4}.*$", "", c).strip().lower()
    head = re.sub(r"\(.*?\)", "", head).strip().rstrip(".")
    return FORECASTERS.get(head, c.split()[0] if c else None)


def _parse_predictions_table(df: pd.DataFrame, label_col_hint="state"):
    cols = _flat_cols(df)
    df = df.copy()
    df.columns = cols
    rating_cols = [c for c in cols if c.lower().startswith("ratings") or re.search(r"\b20\d\d\b", c)]
    out = []
    first = cols[0]
    pvi_col = next((c for c in cols if "pvi" in c.lower()), None)
    for _, row in df.iterrows():
        label = clean(row[first])
        if not label or label.lower() in ("state", "district", "overall"):
            continue
        rec = {"label": label, "pvi": parse_pvi(row[pvi_col]) if pvi_col else None, "ratings": {}}
        inc_col = next((c for c in cols if "senator" in c.lower() or "governor" in c.lower() or "incumbent" in c.lower()), None)
        if inc_col:
            rec["incumbent_text"] = clean(row[inc_col])
        last_col = next((c for c in cols if "last election" in c.lower()), None)
        if last_col:
            m = re.match(r"\s*(\d+(?:\.\d+)?)\s*%\s*([DRI])", clean(row[last_col]))
            if m:
                rec["last_share"], rec["last_party"] = float(m.group(1)), m.group(2)
        for c in rating_cols:
            name = _forecaster_name(c)
            r = normalize_rating(row[c])
            if name and r:
                rec["ratings"][name] = r
        out.append(rec)
    return out


def statewide_overview(kind: str) -> dict:
    """Return {race_id: {...}} for Senate ('SEN') or governor ('GOV') races."""
    title = SENATE_PAGE if kind == "SEN" else GOV_PAGE
    races: dict[str, dict] = {}
    preds, rosters = [], []
    for path, df, _ in tables_with_sections(title):
        cols = " ".join(_flat_cols(df)).lower()
        if path and path[-1].startswith("Predictions") or ("ratings" in cols and ("cook" in cols or "sabato" in cols)):
            preds.append(df)
        if "candidates" in cols and ("status" in cols or "results" in cols) and "territor" not in cols:
            rosters.append(df)
    for df in preds:
        for rec in _parse_predictions_table(df):
            rid = _race_id(rec["label"], kind)
            if not rid:
                continue
            r = races.setdefault(rid, {"race_id": rid})
            r.setdefault("ratings", {}).update(rec["ratings"])
            if rec.get("pvi") is not None:
                r["pvi"] = rec["pvi"]
            if rec.get("incumbent_text"):
                r["incumbent_text"] = rec["incumbent_text"]
            if rec.get("last_share"):
                r["last_share"], r["last_party"] = rec["last_share"], rec["last_party"]
    for df in rosters:
        cols = _flat_cols(df)
        df = df.copy()
        df.columns = cols
        ccol = next(c for c in cols if c.lower().startswith("candidates"))
        scol = next((c for c in cols if c.lower().startswith(("status", "results"))), None)
        icol = next((c for c in cols if c.lower().endswith(("senator", "governor"))), None)
        pcol = next((c for c in cols if c.lower().endswith("party")), None)
        hcol = next((c for c in cols if "electoral history" in c.lower() or "first elected" in c.lower()), None)
        pvicol = next((c for c in cols if "pvi" in c.lower()), None)
        for _, row in df.iterrows():
            rid = _race_id(clean(row[cols[0]]), kind)
            if not rid:
                continue
            r = races.setdefault(rid, {"race_id": rid})
            r["candidates"] = parse_candidates(row[ccol])
            if scol:
                r["status"] = clean(row[scol])
            if hcol:
                yrs = [int(y) for y in re.findall(r"\b(19\d\d|20[0-2]\d)\b", clean(row[hcol]))]
                if yrs:
                    r["first_year"], r["last_year"] = min(yrs), max(yrs)
                r["history_text"] = clean(row[hcol])
            if icol:
                r["incumbent"] = clean(row[icol])
            if pcol:
                r["incumbent_party"] = party_letter(row[pcol])
            if pvicol and r.get("pvi") is None:
                r["pvi"] = parse_pvi(row[pvicol])
    for rid, r in races.items():
        r["state"] = rid[:2]
        r["office"] = kind
        r["source_url"] = page_url(title)
    return races


def _race_id(label: str, kind: str) -> str | None:
    lab = clean(label)
    special = "special" in lab.lower() or "class 3" in lab.lower()
    name = re.sub(r"\(.*?\)", "", lab).strip()
    abbr = NAME_TO_ABBR.get(name)
    if not abbr:
        return None
    if kind == "SEN":
        return f"{abbr}-SEN-SP" if special and abbr in ("OH", "FL") else f"{abbr}-SEN"
    return f"{abbr}-GOV"


def party_letter(s) -> str:
    s = clean(s).lower()
    if s.startswith("dem") or s in ("d", "dfl") or "democratic" in s:
        return "D"
    if s.startswith("rep") or s == "r":
        return "R"
    if s.startswith("ind") or s == "i":
        return "I"
    if s.startswith("lib"):
        return "L"
    if s.startswith("green"):
        return "G"
    return s[:12].upper() or ""


def parse_candidates(cell) -> list[dict]:
    """'▌Mike Collins (Republican)[12] ▌Jon Ossoff (Democratic)[12]' -> [{name, party}]"""
    s = clean(cell)
    out = []
    for chunk in re.split(r"▌|▌|•", s):
        chunk = chunk.strip()
        m = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
        if m:
            out.append({"name": clean(m.group(1)), "party": party_letter(m.group(2)), "party_full": clean(m.group(2))})
    return out


# ---------------- House ----------------
def house_districts() -> pd.DataFrame:
    rows = []
    for path, df, _ in tables_with_sections(HOUSE_PAGE):
        if not path:
            continue
        state = path[0] if path[0] in NAME_TO_ABBR else (path[-1] if path[-1] in NAME_TO_ABBR else None)
        if not state:
            continue
        cols = _flat_cols(df)
        low = [c.lower() for c in cols]
        if not any("pvi" in c for c in low) or not any("candidates" in c for c in low):
            continue
        df = df.copy()
        df.columns = cols
        loc = cols[0]
        pvi = next(c for c in cols if "pvi" in c.lower())
        mem = next((c for c in cols if c.lower().endswith("member")), None)
        par = next((c for c in cols if c.lower().endswith("party")), None)
        fe = next((c for c in cols if "first elected" in c.lower()), None)
        st = next((c for c in cols if c.lower().endswith("status")), None)
        cand = next(c for c in cols if c.lower().startswith("candidates"))
        abbr = NAME_TO_ABBR[state]
        for _, row in df.iterrows():
            label = clean(row[loc])
            m = re.search(r"(\d+)$", label)
            if "at-large" in label.lower() or "at large" in label.lower():
                num = 0
            elif m:
                num = int(m.group(1))
            else:
                continue
            did = f"{abbr}-{'AL' if num == 0 else f'{num:02d}'}"
            rows.append({
                "district": did, "state": abbr, "num": num,
                "pvi": parse_pvi(row[pvi]),
                "incumbent": clean(row[mem]) if mem else "",
                "incumbent_party": party_letter(row[par]) if par else "",
                "first_elected": clean(row[fe]) if fe else "",
                "status": clean(row[st]) if st else "",
                "candidates": parse_candidates(row[cand]),
            })
    df = pd.DataFrame(rows).drop_duplicates("district", keep="first")
    log.info("wikipedia: parsed %d House districts", len(df))
    return df


def house_ratings() -> dict[str, dict[str, str]]:
    """{district: {forecaster: rating}} from the House ratings article."""
    out: dict[str, dict[str, str]] = {}
    for path, df, _ in tables_with_sections(HOUSE_RATINGS_PAGE):
        cols = _flat_cols(df)
        low = " ".join(cols).lower()
        if not ("cook" in low or "sabato" in low or "ie" in low.split()):
            continue
        df = df.copy()
        df.columns = cols
        for _, row in df.iterrows():
            label = clean(row[cols[0]])
            did = district_id_from_label(label)
            if not did:
                continue
            rec = out.setdefault(did, {})
            for c in cols[1:]:
                name = _forecaster_name(c)
                r = normalize_rating(row[c])
                if name and r and name not in ("District", "PVI", "Incumbent"):
                    rec[name] = r
    log.info("wikipedia: House ratings for %d districts", len(out))
    return out


def district_id_from_label(label: str) -> str | None:
    label = clean(label)
    m = re.match(r"([A-Z]{2})[- ]?(\d{1,2}|AL)$", label.upper())
    if m:
        st, n = m.groups()
        return f"{st}-{'AL' if n == 'AL' else f'{int(n):02d}'}"
    m = re.match(r"([A-Za-z .]+?)\s+(\d{1,2}|at-large|at large)", label, re.I)
    if m and clean(m.group(1)) in NAME_TO_ABBR:
        st = NAME_TO_ABBR[clean(m.group(1))]
        n = m.group(2).lower()
        return f"{st}-{'AL' if 'large' in n else f'{int(n):02d}'}"
    return None


# ---------------- Polls ----------------
def race_page_title(race_id: str) -> str:
    st = STATES[race_id[:2]]
    if race_id.endswith("-SEN-SP"):
        return f"2026 United States Senate special election in {st}"
    if race_id.endswith("-SEN"):
        return f"2026 United States Senate election in {st}"
    if race_id.endswith("-GOV"):
        return f"2026 {st} gubernatorial election"
    num = race_id.split("-")[1]
    if num == "AL":
        return f"2026 United States House of Representatives election in {st}"
    return f"2026 {st}'s {ordinal(int(num))} congressional district election"


def ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


_CAND_COL = re.compile(r"^(.*?)\s*\(([A-Z]{1,3})\)\s*$")


def _sample(s) -> tuple[int | None, str | None]:
    s = clean(s)
    m = re.search(r"([\d,]+)", s)
    n = int(m.group(1).replace(",", "")) if m else None
    pop = None
    for p in ("LV", "RV", "A"):
        if re.search(rf"\({p}\)|\b{p}\b", s):
            pop = p
            break
    return n, pop


def race_polls(race_id: str, dem_name: str, rep_name: str, title: str | None = None,
               other_name: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return (individual polls, aggregator averages) for the general-election matchup on a race page."""
    title = title or race_page_title(race_id)
    try:
        tabs = list(tables_with_sections(title))
    except RuntimeError as e:
        log.info("wikipedia polls: %s (%s)", title, e)
        return [], []
    return parse_poll_tables(tabs, race_id, dem_name or other_name or "", rep_name or "", page_url(title))


def _pick_col(df: pd.DataFrame, cand_cols: dict, name: str, parties: tuple) -> str | None:
    """Column for a nominee: exact full-name match first, else last-name match within the party, and when
    several columns share the surname (e.g., two "Dan Sullivan"s in Alaska) the one with the most support."""
    if not name:
        return None
    full, last = norm_name(name), last_name(name)
    exact = [c for c, (n, p) in cand_cols.items() if n == full]
    if exact:
        return exact[0]
    cands = [c for c, (n, p) in cand_cols.items() if last and last in n.split() and (p in parties or not p)]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    return max(cands, key=lambda c: pd.to_numeric(df[c].map(pct), errors="coerce").mean(skipna=True) or 0)


def parse_poll_tables(tabs, race_id: str, dem_name: str, rep_name: str, src_url: str) -> tuple[list[dict], list[dict]]:
    d_last = last_name(dem_name)
    r_last = last_name(rep_name)
    polls, aggs = [], []
    for path, df, _ in tabs:
        p = " / ".join(path).lower()
        if "primary" in p and "general" not in p:
            continue
        cols = _flat_cols(df)
        if not cols:
            continue
        low0 = cols[0].lower()
        cand_cols = {}
        for c in cols:
            m = _CAND_COL.match(c.split("|")[-1].strip())
            if m:
                cand_cols[c] = (norm_name(m.group(1)), m.group(2))
        df = df.copy()
        df.columns = cols
        dcol = _pick_col(df, cand_cols, dem_name, ("D", "I", "DFL"))
        rcol = _pick_col(df, cand_cols, rep_name, ("R",))
        if not dcol or not rcol or dcol == rcol:
            continue
        if "aggregation" in low0:
            for _, row in df.iterrows():
                src = clean(row[cols[0]])
                if not src or src.lower() == "average":
                    continue
                d, r = pct(row[dcol]), pct(row[rcol])
                if d is None or r is None:
                    continue
                upd = next((c for c in cols if "updated" in c.lower()), None)
                _, upd_d = parse_date_range(row[upd]) if upd else (None, None)
                aggs.append({"race_id": race_id, "aggregator": src, "dem": d, "rep": r,
                             "as_of": upd_d.isoformat() if upd_d else None, "source_url": src_url})
            continue
        if not low0.startswith("poll source"):
            continue
        dcolname = next((c for c in cols if c.lower().startswith("date")), None)
        scol = next((c for c in cols if c.lower().startswith("sample")), None)
        mcol = next((c for c in cols if "margin of error" in c.lower()), None)
        others = [c for c in cand_cols if c not in (dcol, rcol)]
        ocol = next((c for c in cols if c.split("|")[-1].strip().lower().startswith("other")), None)
        ucol = next((c for c in cols if c.split("|")[-1].strip().lower().startswith("undecided")), None)
        for _, row in df.iterrows():
            src_raw = str(row[cols[0]])
            src = clean(src_raw)
            if not src or src.lower().startswith(("poll source", "hypothetical")):
                continue
            start, end = parse_date_range(row[dcolname]) if dcolname else (None, None)
            if not end:
                continue
            n, popn = _sample(row[scol]) if scol else (None, None)
            d, r = pct(row[dcol]), pct(row[rcol])
            if d is None or r is None or d + r < 50 or d + r > 101:
                continue
            spons = None
            m = re.search(r"\((D|R)\)", src)
            if m:
                spons = m.group(1)
            pollster = re.sub(r"\s*\((D|R|I|L)\)\s*", " ", src).strip()
            answers = [[cand_cols[c][0], cand_cols[c][1], pct(row[c])] for c in cand_cols]
            polls.append({
                "race_id": race_id, "pollster": pollster, "sponsor_party": spons, "answers": answers,
                "start": start.isoformat() if start else None, "end": end.isoformat(),
                "n": n, "pop": popn, "dem": d, "rep": r,
                "other": sum(filter(None, [pct(row[c]) for c in others])) if others else pct(row[ocol]) if ocol else None,
                "undecided": pct(row[ucol]) if ucol else None,
                "moe": pct(row[mcol]) if mcol else None,
                "variant": bool(re.search(r"\[[a-z]\]", str(row[dcol]))),
                "n_cands": 2 + len(others),
                "source": "Wikipedia", "source_url": src_url,
            })
    return polls, aggs


def house_state_pages(state_abbr: str) -> list[str]:
    st = STATES[state_abbr]
    return [f"2026 United States House of Representatives elections in {st}",
            f"2026 United States House of Representatives elections in {st} (districts 1–26)",
            f"2026 United States House of Representatives elections in {st} (districts 27–52)"]


def house_district_polls(state_abbr: str, nominees: dict[str, tuple[str, str]]) -> list[dict]:
    """Polls for every district in a state from its state-level House election page(s).
    nominees: {district_id: (dem_name, rep_name)} for that state."""
    out = []
    pages = house_state_pages(state_abbr)
    if state_abbr != "CA":
        pages = pages[:1]
    else:
        pages = pages[1:]
    for title in pages:
        try:
            tabs = list(tables_with_sections(title))
        except RuntimeError as e:
            log.info("wikipedia state House page missing: %s (%s)", title, type(e).__name__)
            continue
        by_dist: dict[str, list] = {}
        for path, df, el in tabs:
            if not path:
                continue
            m = re.match(r"District (\d+)", path[0]) or (re.match(r"At-large", path[0], re.I) and re.match(r"(.*)", "0"))
            if not m:
                continue
            num = int(m.group(1))
            did = f"{state_abbr}-{'AL' if num == 0 else f'{num:02d}'}"
            by_dist.setdefault(did, []).append((path, df, el))
        for did, tabs_d in by_dist.items():
            if did not in nominees:
                continue
            dem, rep = nominees[did]
            polls, _ = parse_poll_tables(tabs_d, did, dem, rep, page_url(title))
            out += polls
    return out


def generic_ballot_aggregates() -> list[dict]:
    out = []
    for path, df, _ in tables_with_sections(HOUSE_PAGE):
        cols = _flat_cols(df)
        if not cols or "aggregation" not in cols[0].lower():
            continue
        df = df.copy()
        df.columns = cols
        dcol = next((c for c in cols if c.lower().startswith("democrat")), None)
        rcol = next((c for c in cols if c.lower().startswith("republican")), None)
        ucol = next((c for c in cols if "updated" in c.lower()), None)
        if not dcol or not rcol:
            continue
        for _, row in df.iterrows():
            src = clean(row[cols[0]])
            if not src or src.lower() == "average":
                continue
            d, r = pct(row[dcol]), pct(row[rcol])
            if d is None or r is None:
                continue
            _, upd = parse_date_range(row[ucol]) if ucol else (None, None)
            out.append({"aggregator": src, "dem": d, "rep": r, "as_of": upd.isoformat() if upd else None,
                        "source_url": page_url(HOUSE_PAGE)})
    return out


# ---------------- Presidential approval ----------------
APPROVAL_PAGE = "Opinion polling on the second Trump presidency"


def approval() -> dict:
    """Aggregator averages and individual national approval polls (last ~120 days)."""
    aggs, polls = [], []
    for path, df, _ in tables_with_sections(APPROVAL_PAGE):
        cols = _flat_cols(df)
        if not cols:
            continue
        df = df.copy()
        df.columns = cols
        if path and path[0] != "Nationwide job approval ratings":
            continue
        path = path[1:]
        if path[:2] == ["Aggregate polls", "Approval"] and cols[0].lower().startswith("aggregator"):
            for _, row in df.iterrows():
                a, d = pct(row.get("Approve")), pct(row.get("Disapprove"))
                if a is None or d is None:
                    continue
                _, upd = parse_date_range(row.get("Updated", ""))
                aggs.append({"aggregator": clean(row[cols[0]]), "approve": a, "disapprove": d,
                             "as_of": upd.isoformat() if upd else None})
        elif len(path) >= 2 and path[0] in ("2026", "2025") and cols[0].lower().startswith("poll source") and "Approve" in cols:
            for _, row in df.iterrows():
                a, d = pct(row.get("Approve")), pct(row.get("Disapprove"))
                st, en = parse_date_range(row.get("Date", ""), default_year=int(path[0]))
                if a is None or d is None or not en:
                    continue
                n, popn = _sample(row.get("Sample size", ""))
                polls.append({"pollster": clean(row[cols[0]]), "start": st.isoformat() if st else None,
                              "end": en.isoformat(), "n": n, "pop": popn, "approve": a, "disapprove": d})
    return {"aggregates": aggs, "polls": polls, "source_url": page_url(APPROVAL_PAGE)}

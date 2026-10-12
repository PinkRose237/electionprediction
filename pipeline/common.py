"""Shared helpers: paths, HTTP session with retries, caching, parsing utilities."""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
STATIC = DATA / "static"
RESEARCH = DATA / "research"
PROCESSED = DATA / "processed"
HISTORY = DATA / "history"
SITE = ROOT / "site"

ELECTION_DAY = dt.date(2026, 11, 3)
TODAY = dt.date.today()

USER_AGENT = "MidtermForecastModel/1.0 (+https://github.com/PinkRose237/electionprediction; python-requests)"

log = logging.getLogger("forecast")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

for d in (RAW, STATIC, RESEARCH, PROCESSED, HISTORY):
    d.mkdir(parents=True, exist_ok=True)

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
NAME_TO_ABBR = {v: k for k, v in STATES.items()}

# Census regions/divisions are used to build correlated polling errors between states.
DIVISION = {
    "CT": "NE", "ME": "NE", "MA": "NE", "NH": "NE", "RI": "NE", "VT": "NE",
    "NJ": "MA", "NY": "MA", "PA": "MA",
    "IL": "ENC", "IN": "ENC", "MI": "ENC", "OH": "ENC", "WI": "ENC",
    "IA": "WNC", "KS": "WNC", "MN": "WNC", "MO": "WNC", "NE": "WNC", "ND": "WNC", "SD": "WNC",
    "DE": "SA", "DC": "SA", "FL": "SA", "GA": "SA", "MD": "SA", "NC": "SA", "SC": "SA", "VA": "SA", "WV": "SA",
    "AL": "ESC", "KY": "ESC", "MS": "ESC", "TN": "ESC",
    "AR": "WSC", "LA": "WSC", "OK": "WSC", "TX": "WSC",
    "AZ": "MT", "CO": "MT", "ID": "MT", "MT": "MT", "NV": "MT", "NM": "MT", "UT": "MT", "WY": "MT",
    "AK": "PAC", "CA": "PAC", "HI": "PAC", "OR": "PAC", "WA": "PAC",
}


class NotFound(RuntimeError):
    """Client error (4xx other than 429): retrying will not help."""


class Http:
    """requests.Session wrapper with polite retries/backoff and a per-host minimum delay."""

    def __init__(self, min_delay: float = 0.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.min_delay = min_delay
        self._last: dict[str, float] = {}

    def get(self, url: str, *, params=None, timeout=60, tries=5, delay: float | None = None, headers=None) -> requests.Response:
        host = re.sub(r"^https?://([^/]+).*$", r"\1", url)
        wait = self.min_delay if delay is None else delay
        last_err = None
        for attempt in range(tries):
            since = time.time() - self._last.get(host, 0)
            if since < wait:
                time.sleep(wait - since)
            try:
                r = self.s.get(url, params=params, timeout=timeout, headers=headers)
                self._last[host] = time.time()
                if r.status_code == 200:
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    back = min(60, 4 * (attempt + 1))
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        back = min(90, int(ra))
                    log.warning("HTTP %s from %s; backing off %ss", r.status_code, host, back)
                    time.sleep(back)
                    continue
                if 400 <= r.status_code < 500:
                    raise NotFound(f"HTTP {r.status_code}: {url}")
                r.raise_for_status()
            except NotFound:
                raise
            except requests.RequestException as e:  # network hiccup
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GET failed after {tries} tries: {url} ({last_err})")


def raw_dir(source: str, day: dt.date | None = None) -> Path:
    p = RAW / (day or TODAY).isoformat() / source
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(obj):
    """Recursively replace NaN/inf (which browsers' JSON.parse rejects) with None."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = json_safe(json.loads(json.dumps(obj, default=str)))
    path.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False, allow_nan=False))


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


# ---------- parsing helpers ----------
_FOOT = re.compile(r"\[[^\]]*\]")


def clean(s) -> str:
    """Strip Wikipedia footnote markers and whitespace."""
    if s is None:
        return ""
    s = str(s)
    if s == "nan":
        return ""
    s = _FOOT.sub("", s)
    s = s.replace("\xa0", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def pct(s) -> float | None:
    """Parse '51%' / '51.5' / '—' into a float or None."""
    s = clean(s)
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m or s.startswith(("–", "—")) and len(s) <= 2:
        return None
    return float(m.group())


def norm_name(s: str) -> str:
    s = clean(s).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\.?\b", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def last_name(s: str) -> str:
    parts = norm_name(s).split()
    return parts[-1] if parts else ""


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MON_ABBR = {k[:3]: v for k, v in MONTHS.items()}
MON_ABBR["sept"] = 9


def parse_date_range(s: str, default_year: int = 2026) -> tuple[dt.date | None, dt.date | None]:
    """Parse Wikipedia-style date ranges: 'October 2–3, 2026', 'Sep 28 – Oct 1, 2026', 'September 15, 2026'."""
    s = clean(s).replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.lower().replace(".", "")
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", s)]
    year = years[-1] if years else default_year
    toks = re.findall(r"[a-z]+|\d+|-", s)
    dates = []
    cur_m = None
    cur_y = year
    for t in toks:
        if t.isalpha():
            key = t[:4] if t.startswith("sept") else t[:3]
            if key in MON_ABBR:
                cur_m = MON_ABBR[key]
        elif t.isdigit():
            n = int(t)
            if n > 1900:
                # assign the year to any dates captured so far that lack it
                cur_y = n
                continue
            if cur_m and 1 <= n <= 31:
                dates.append([cur_m, n])
    if not dates:
        return None, None
    try:
        start = dt.date(year, dates[0][0], dates[0][1])
        end = dt.date(year, dates[-1][0], dates[-1][1])
    except ValueError:
        return None, None
    if start > end:  # range crossing a year boundary
        start = dt.date(year - 1, start.month, start.day)
    return start, end


def days_until_election(day: dt.date | None = None) -> int:
    return (ELECTION_DAY - (day or TODAY)).days

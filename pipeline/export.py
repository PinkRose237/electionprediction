"""Writes the site: a compact forecast dataset, day-over-day changes, and the running history.

  data/history/topline.csv        one row per run-day (chamber odds, seats, national margin)
  data/history/race_probs.csv     one row per run-day x competitive race (P(D), mean margin)
  site/data/forecast.json         compact dataset the page reads
  site/index.html                 template with the dataset inlined (works from file:// and as an Artifact)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pipeline.common import HISTORY, PROCESSED, RESEARCH, SITE, json_safe, log, save_json

TEMPLATE = SITE / "template.html"


def _r(x, nd=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def model_rating(p: float) -> str:
    side = "D" if p >= 0.5 else "R"
    q = max(p, 1 - p)
    if q < 0.60:
        return "Toss-up"
    if q < 0.75:
        return f"Tilt {side}"
    if q < 0.90:
        return f"Lean {side}"
    if q < 0.97:
        return f"Likely {side}"
    return f"Safe {side}"


def _cand(c: dict | None) -> dict | None:
    if not c:
        return None
    out = {"name": c.get("name"), "party": c.get("party"), "inc": bool(c.get("is_incumbent"))}
    for k in ("experience", "scandal", "receipts", "cash", "fec_through", "experience_detail"):
        v = c.get(k)
        if v is not None and v == v:
            out[k] = v if not isinstance(v, float) else round(v, 1)
    return out


def slim_race(r: dict, news: dict) -> dict:
    sim = r.get("sim") or {}
    comps = {}
    for k, c in (r.get("components") or {}).items():
        d = {"margin": _r(c.get("margin")), "sd": _r(c.get("sd")), "w": (r.get("weights") or {}).get(k)}
        if k == "fundamentals":
            d["parts"] = c.get("parts")
        elif k == "polls":
            d.update({"n": c.get("n"), "n_eff": c.get("n_eff"), "last": c.get("last"), "dem": _r(c.get("dem")), "rep": _r(c.get("rep"))})
        elif k == "ratings":
            d.update({"label": c.get("label"), "n": c.get("n")})
        elif k == "markets":
            d.update({"p": _r(c.get("p_dem"), 3), "sources": c.get("sources")})
        comps[k] = d
    competitive = bool(r.get("competitive")) or r["office"] != "HOUSE"
    out = {"id": r["race_id"], "office": r["office"], "state": r["state"], "name": r.get("name"),
           "pvi": r.get("pvi"), "p": _r(r.get("p_dem"), 4), "mean": _r(sim.get("mean", r.get("mean"))),
           "p10": _r(sim.get("p10")), "p90": _r(sim.get("p90")), "p025": _r(sim.get("p025")), "p975": _r(sim.get("p975")),
           "rating": model_rating(r.get("p_dem", 0.5)), "dem": _cand(r.get("dem")), "rep": _cand(r.get("rep")),
           "inc": r.get("incumbent") or None, "incp": r.get("inc_party"), "incrun": bool(r.get("inc_running")),
           "fixed": r.get("fixed_winner"), "ind": bool(r.get("dem_is_independent"))}
    if not competitive and not r.get("ratings"):
        return out
    out.update({"status": r.get("status"), "comps": comps, "ratings": r.get("ratings") or {},
                "consensus": r.get("rating_consensus"), "notes": r.get("notes"), "rules": r.get("special_rules"),
                "wiki": r.get("wiki"), "appointed": bool(r.get("appointed")),
                "others": [o.get("name") + f" ({o.get('party')})" for o in (r.get("others") or [])][:3]})
    polls = []
    for p in (r.get("poll_list") or [])[: (14 if r["office"] != "HOUSE" else 8)]:
        polls.append({k: p.get(k) for k in ("pollster", "end", "n", "pop", "dem", "rep", "sponsor", "adj", "weight", "grade", "url")})
    out["polls"] = polls
    out["aggs"] = [{"src": a.get("aggregator"), "dem": a.get("dem"), "rep": a.get("rep"), "asof": a.get("as_of")}
                   for a in (r.get("aggregator_averages") or [])]
    if news.get(r["race_id"]):
        out["news"] = news[r["race_id"]][:5]
    return out


def chamber(ch: dict) -> dict:
    keys = ("total_seats", "d_mean", "r_mean", "i_mean", "d_median", "d_p10", "d_p90", "d_p025", "d_p975",
            "p_d_control", "p_r_control", "p_other", "tipping_point", "seats_by_national_vote")
    out = {k: ch.get(k) for k in keys if k in ch}
    out["hist"] = ch.get("d_hist")
    return out


def append_history(out: dict) -> dict:
    HISTORY.mkdir(parents=True, exist_ok=True)
    tl = HISTORY / "topline.csv"
    row = {"date": out["asof"], "house_p_d": _r(out["house"]["p_d_control"], 4), "house_d_seats": _r(out["house"]["d_mean"], 1),
           "senate_p_d": _r(out["senate"]["p_d_control"], 4), "senate_p_r": _r(out["senate"]["p_r_control"], 4),
           "senate_d_seats": _r(out["senate"]["d_mean"], 2), "gov_d": _r(out["governors"]["d_mean"], 2),
           "national": _r(out["national"]["margin"], 2), "national_sd": _r(out["national"]["sd"], 2)}
    rows = []
    if tl.exists():
        rows = [r for r in csv.DictReader(tl.open()) if r["date"] != row["date"]]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["date"])
    with tl.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)
    rp = HISTORY / "race_probs.csv"
    prev = []
    if rp.exists():
        prev = [r for r in csv.DictReader(rp.open()) if r["date"] != out["asof"]]
    for r in out["races"]:
        if r.get("fixed_winner"):
            continue
        p = r.get("p_dem", 0)
        if r["office"] != "HOUSE" or 0.01 < p < 0.99 or r.get("competitive"):
            prev.append({"date": out["asof"], "race_id": r["race_id"], "p_dem": f"{p:.4f}",
                         "mean": f"{(r.get('sim') or {}).get('mean', r.get('mean', 0)):.2f}"})
    with rp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "race_id", "p_dem", "mean"])
        w.writeheader()
        w.writerows(sorted(prev, key=lambda r: (r["date"], r["race_id"])))
    # compact history for the page
    topline = [{k: (float(v) if k != "date" and v not in ("", "None") else (None if v in ("", "None") else v)) for k, v in r.items()}
               for r in rows]
    races: dict[str, list] = {}
    for r in prev:
        races.setdefault(r["race_id"], []).append([r["date"], round(float(r["p_dem"]), 3)])
    return {"topline": topline, "races": races}


def changes(out: dict, races_slim: list[dict]) -> dict:
    """Compare today's run with the previous run day, using the committed history files
    (data/history/topline.csv and race_probs.csv), so it works on a fresh checkout too."""
    tl_path, rp_path = HISTORY / "topline.csv", HISTORY / "race_probs.csv"
    if not tl_path.exists():
        return {"since": None, "races": [], "topline": {}}
    prev_days = sorted({r["date"] for r in csv.DictReader(tl_path.open()) if r["date"] < out["asof"]})
    if not prev_days:
        return {"since": None, "races": [], "topline": {}}
    since = prev_days[-1]
    prev_tl = next(r for r in csv.DictReader(tl_path.open()) if r["date"] == since)
    pmap = {}
    if rp_path.exists():
        pmap = {r["race_id"]: float(r["p_dem"]) for r in csv.DictReader(rp_path.open()) if r["date"] == since}
    moved = []
    for r in races_slim:
        p0 = pmap.get(r["id"])
        if p0 is None or r.get("p") is None:
            continue
        d = r["p"] - p0
        if abs(d) >= 0.03:
            moved.append({"id": r["id"], "name": r["name"], "office": r["office"], "from": round(p0, 3), "to": r["p"], "delta": round(d, 3)})
    moved.sort(key=lambda x: -abs(x["delta"]))

    def f(k):
        try:
            return float(prev_tl[k])
        except (KeyError, TypeError, ValueError):
            return None
    tl = {"house": {"from": f("house_p_d"), "to": out["house"].get("p_d_control")},
          "senate": {"from": f("senate_p_d"), "to": out["senate"].get("p_d_control")},
          "national": {"from": f("national"), "to": out["national"].get("margin")}}
    return {"since": since, "races": moved[:25], "topline": tl}


def _research_csv(name: str) -> list[dict]:
    import pandas as pd
    p = RESEARCH / name
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return json_safe(df.where(df.notna(), None).to_dict("records"))


def site_data(out: dict, hist: dict) -> dict:
    nat = out["national"]
    news = out.get("news") or {}
    races = [slim_race(r, news) for r in out["races"]]
    comps = {}
    for k, c in (nat.get("components") or {}).items():
        if not c:
            continue
        d = {kk: vv for kk, vv in c.items() if kk not in ("detail",)}
        if k == "markets":
            d["detail"] = c.get("detail")
        d["w"] = (nat.get("weights") or {}).get(k)
        comps[k] = d
    fm = nat.get("fundamentals_model") or {}
    natout = {"margin": _r(nat["margin"], 2), "sd": _r(nat["sd"], 2), "struct_sd": _r(nat.get("struct_sd"), 2),
              "drift_sd": _r(nat.get("drift_sd"), 2), "sim": nat.get("sim"), "components": comps,
              "approval": nat.get("approval"), "aggregators": nat.get("aggregators"), "series": nat.get("series"),
              "polls": [{k: p.get(k) for k in ("pollster", "end", "n", "pop", "dem", "rep", "sponsor", "adj", "weight", "grade", "url")}
                        for p in (nat.get("generic_polls") or [])[:60]],
              "chart_polls": nat.get("chart_polls") or [],
              "n_polls_total": nat.get("n_generic_polls_total"), "lv_shift": nat.get("lv_shift"),
              "house_effects": nat.get("house_effects"), "backtest": nat.get("backtest"),
              "fundamentals": {"level": {k: fm.get("level", {}).get(k) for k in ("intercept", "slope", "era", "loo_rmse", "n")},
                               "swing": {k: (fm.get("swing") or {}).get(k) for k in ("intercept", "slope", "loo_rmse", "n")}}}
    data = {"meta": {"generated_at": out["generated_at"], "asof": out["asof"], "days_out": out["days_out"],
                     "n_sims": out["n_sims"], "n_polls": out.get("n_polls"), "fetch_status": out.get("fetch_status"),
                     "election_day": out["election_day"]},
            "national": natout, "economy": out.get("economy"), "house": chamber(out["house"]),
            "senate": chamber(out["senate"]), "governors": chamber(out["governors"]), "joint": out.get("joint"),
            "races": races, "national_news": {k: news.get(k, [])[:8] for k in ("NATIONAL", "NATIONAL_POLLS")},
            "markets_control": out.get("markets_control"), "other_forecasts": _research_csv("other_forecasts.csv"),
            "history": hist,
            "research": out.get("research")}
    return data


def write_site(out: dict) -> None:
    races_slim = [slim_race(r, out.get("news") or {}) for r in out["races"]]
    delta = changes(out, races_slim)
    hist = append_history(out)
    data = site_data(out, hist)
    data["changes"] = delta
    save_json(SITE / "data" / "forecast.json", data)
    if TEMPLATE.exists():
        safe = json_safe(json.loads(json.dumps(data, default=str)))
        payload = json.dumps(safe, separators=(",", ":"), default=str, allow_nan=False).replace("</", "<\\/")
        html = TEMPLATE.read_text().replace("/*__FORECAST_DATA__*/null", payload)
        (SITE / "index.html").write_text(html)
        log.info("site written: %s (%.0f KB)", SITE / "index.html", len(html) / 1024)
    else:
        log.warning("no site template at %s; wrote data only", TEMPLATE)


if __name__ == "__main__":   # rebuild the site from the latest processed forecast without re-running the model
    write_site(json.loads((PROCESSED / "forecast.json").read_text()))

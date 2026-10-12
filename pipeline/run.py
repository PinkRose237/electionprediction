"""Daily pipeline: fetch every source -> build races -> simulate -> export site data.

Usage:  .venv/bin/python -m pipeline.run [--sims 40000] [--no-news] [--offline]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import traceback

import numpy as np
import pandas as pd

from pipeline.common import (DATA, PROCESSED, RESEARCH, STATES, STATIC, TODAY, days_until_election, last_name,
                             load_json, log, save_json)
from pipeline.fetch import fec as FEC
from pipeline.fetch import fred as FRED
from pipeline.fetch import markets as MK
from pipeline.fetch import news as NEWS
from pipeline.fetch import pollfeeds as PF
from pipeline.fetch import wikipedia as W
from pipeline.model import national as NAT
from pipeline.model import polls as P
from pipeline.model import races as R
from pipeline.model import simulate as SIM

MANUAL = DATA / "manual"
MANUAL.mkdir(exist_ok=True)

# Seats not up in 2026. Senate: 53 R / 47 D caucus (incl. King & Sanders) before the election;
# 22 R-held and 13 D-held seats are on the ballot.
SENATE_NOT_UP = {"D": 34, "R": 31, "I": 0}
GOV_NOT_UP = {"D": 6, "R": 8}     # DE KY NJ NC VA WA (D); IN LA MS MO MT ND UT WV (R)
# States whose congressional maps were redrawn for 2026 (incumbency advantage discounted).
REDISTRICTED_STATES = {"TX", "CA", "NC", "OH", "UT", "FL", "TN", "LA", "AL"}  # MO blocked; VA voided
# Races where the Democratic side is an independent Democrats are not opposing.
DEM_BACKED_INDEPENDENTS = {"NE-SEN": "Dan Osborn"}
OVERRIDES = load_json(STATIC / "overrides.json", {})
# Ranked-choice final round from first-round polls: how eliminated candidates' votes transfer
# (share to the Democratic-side finalist, share to the Republican finalist; the rest exhaust).
RCV_TRANSFER = {"R": (0.10, 0.70), "D": (0.70, 0.10), "O": (0.35, 0.25)}


def read_csv(name: str, bases=(RESEARCH, MANUAL)) -> pd.DataFrame:
    frames = []
    for base in bases:
        p = base / name
        if p.exists() and p.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                log.warning("could not read %s: %s", p, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def research_rows(df: pd.DataFrame, key: str) -> dict:
    if df.empty or key not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        k = str(r[key]).strip().upper()
        m = re.match(r"^([A-Z]{2})-(\d{1,2})$", k)
        if m:
            k = f"{m.group(1)}-{int(m.group(2)):02d}"
        out[k] = {c: (None if (isinstance(v, float) and math.isnan(v)) else v) for c, v in r.items()}
    return out


def research_polls(df: pd.DataFrame, id_col: str) -> list[dict]:
    out = []
    if df.empty or id_col not in df.columns:
        return out
    for _, r in df.iterrows():
        rid = str(r.get(id_col, "")).strip().upper()
        m = re.match(r"^([A-Z]{2})-(\d{1,2})$", rid)
        if m:
            rid = f"{m.group(1)}-{int(m.group(2)):02d}"
        if not rid:
            continue
        out.append({"race_id": rid, "pollster": r.get("pollster"), "sponsor_party": r.get("sponsor_party"),
                    "start": r.get("start_date"), "end": r.get("end_date"), "n": r.get("sample_size"),
                    "pop": r.get("population"), "dem": r.get("dem_pct"), "rep": r.get("rep_pct"),
                    "other": r.get("other_pct"), "url": r.get("url"), "source": "Research"})
    return out


def step(status: dict, name: str, fn, default):
    try:
        v = fn()
        status[name] = "ok"
        return v
    except Exception as e:
        log.error("%s failed: %s", name, e)
        traceback.print_exc()
        status[name] = f"failed: {e}"
        return default


# ---------------------------------------------------------------- nominee resolution
def resolve_statewide(rid: str, ov: dict, res: dict) -> dict:
    cands = ov.get("candidates", [])
    override = {"dem": res.get("dem_nominee"), "rep": res.get("rep_nominee")}
    manual = OVERRIDES.get(rid, {})
    if manual.get("dem_is_independent"):
        override = {"dem": manual["dem"], "rep": manual.get("rep") or override["rep"], "dem_is_independent": True}
        d, r, others = R.pick_nominees(cands, ov.get("incumbent", ""), ov.get("incumbent_party", ""), override)
        return _resolved(rid, ov, res, d, r, others, True)
    if manual.get("rep"):
        override["rep"] = manual["rep"]
    if manual.get("dem"):
        override["dem"] = manual["dem"]
    d_is_ind = rid in DEM_BACKED_INDEPENDENTS
    if d_is_ind:
        override = {"dem": DEM_BACKED_INDEPENDENTS[rid], "rep": override["rep"], "dem_is_independent": True}
    elif str(res.get("other_backed_by_dems", "")).strip() in ("1", "1.0", "True") and not res.get("dem_nominee"):
        d_is_ind = True
        override = {"dem": res.get("other_major_candidate"), "rep": override["rep"], "dem_is_independent": True}
    d, r, others = R.pick_nominees(cands, ov.get("incumbent", ""), ov.get("incumbent_party", ""), override)
    if d is None and r is not None:
        # no Democrat on the ballot: the main independent (research file, else the only one listed) is the opposition
        inds = [c for c in cands if c["party"] == "I"]
        hint = res.get("other_major_candidate")
        pick = next((c for c in inds if hint and last_name(c["name"]) == last_name(hint)), inds[0] if len(inds) == 1 else None)
        if pick:
            d, d_is_ind = dict(pick), True
            others = [c for c in others if c is not pick and c.get("name") != pick["name"]]
    return _resolved(rid, ov, res, d, r, others, d_is_ind)


def _resolved(rid, ov, res, d, r, others, d_is_ind) -> dict:
    cands = ov.get("candidates", [])
    inc = ov.get("incumbent", "")
    hist = None
    if ov.get("last_share") and ov.get("last_party"):
        if rid.endswith("-GOV"):
            yr = 2024 if rid[:2] in ("VT", "NH") else 2022
            if (ov.get("first_year") or 0) > yr:      # took office later (succession): not their own result
                yr = None
        else:
            yr = ov.get("last_year")
        if yr:
            hist = (yr, ov["last_share"], ov["last_party"])
    return {"d": d, "r": r, "others": others, "d_is_ind": d_is_ind, "inc": inc, "hist": hist,
            "inc_party": ov.get("incumbent_party") or res.get("incumbent_party"),
            "inc_run": R.incumbent_running(ov.get("status", ""), cands, inc),
            "appointed": R.appointed_incumbent(ov.get("status", ""), ov.get("incumbent_text", ""))}


def resolve_house(row, res: dict, rated: bool) -> dict:
    cands = row["candidates"] or []
    inc = row["incumbent"] if "new seat" not in str(row["incumbent"]).lower() else ""
    inc_party = row["incumbent_party"] if row["incumbent_party"] in ("D", "R") else None
    d, r, others = R.pick_nominees(cands, inc, inc_party or "", {"dem": res.get("dem_nominee"), "rep": res.get("rep_nominee")})
    inc_run = bool(inc) and R.incumbent_running(row["status"], cands, inc)
    # a serious independent (e.g., an incumbent who left their party) stands in for a missing nominee
    inds = [c for c in cands if c["party"] == "I"]
    serious = [c for c in inds if last_name(c["name"]) == last_name(inc or "~")] or (inds if rated else [])
    if serious and (r is None) != (d is None):
        if r is None:
            r = dict(serious[0])
        else:
            d = dict(serious[0])
    fixed = None
    if not r and d:
        fixed = "D"
    elif not d and r:
        fixed = "R"
    elif not d and not r:
        fixed = "D" if (row["pvi"] or 0) > 0 else "R"
    return {"d": d, "r": r, "others": others, "inc": inc, "inc_party": inc_party, "inc_run": inc_run, "fixed": fixed}


# ---------------------------------------------------------------- polls
def rcv_final_round(row: dict, party_of: dict) -> dict:
    """Approximate the ranked-choice final round from a first-round poll (Alaska): votes for eliminated
    candidates transfer by party per RCV_TRANSFER. Final-round and head-to-head polls pass through."""
    ans = row.get("answers") or []
    d, r = row.get("dem"), row.get("rep")
    if d is None or r is None or len(ans) <= 2:
        return row
    rest = {"R": 0.0, "D": 0.0, "O": 0.0}
    used_d = used_r = False
    for name, party, share in ans:
        if share is None:
            continue
        if not used_d and abs(share - d) < 1e-9:
            used_d = True
            continue
        if not used_r and abs(share - r) < 1e-9:
            used_r = True
            continue
        p = party or party_of.get(last_name(name)) or "O"
        rest["R" if p == "R" else "D" if p in ("D", "DFL") else "O"] += share
    out = dict(row)
    out["dem"] = d + sum(rest[k] * RCV_TRANSFER[k][0] for k in rest)
    out["rep"] = r + sum(rest[k] * RCV_TRANSFER[k][1] for k in rest)
    out["rcv_adjusted"] = True
    return out


def gather_polls(status, statewide_noms: dict, house_noms: dict, vh_raw: list, offline=False,
                 party_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    party_maps = party_maps or {}
    rows, aggs = [], {}
    for rid, (dem, rep) in statewide_noms.items():
        wp, ag = W.race_polls(rid, dem, rep)
        rows += wp
        aggs[rid] = ag
    by_state: dict[str, dict] = {}
    for did, nm in house_noms.items():
        by_state.setdefault(did[:2], {})[did] = nm
    for st, noms in sorted(by_state.items()):
        rows += step(status, f"wiki_house_polls_{st}", lambda: W.house_district_polls(st, noms), [])
    allnoms = {**statewide_noms, **house_noms}
    if vh_raw:
        rows += PF.votehub_races(vh_raw, allnoms)
    if not offline:
        rows += step(status, "270towin", lambda: PF.t270_races(statewide_noms), [])
    rows += research_polls(read_csv("polls_statewide_extra.csv"), "race_id")
    rows += research_polls(read_csv("house_polls.csv"), "district")
    rows = [rcv_final_round(r, party_maps.get(r["race_id"], {})) if r.get("race_id") in P.RCV_RACES else r for r in rows]
    df = P.standardize(pd.DataFrame(rows)) if rows else pd.DataFrame()
    log.info("race polls: %d raw rows -> %d unique polls across %d races", len(rows), len(df),
             df["race_id"].nunique() if not df.empty else 0)
    return df, aggs


# ---------------------------------------------------------------- national
def build_national(status, mk: dict, approval: dict, vh_raw: list, asof: dt.date):
    gb_rows = step(status, "silver_bulletin_generic", PF.silver_bulletin_generic, [])
    gb_rows += PF.votehub_generic(vh_raw) if vh_raw else []
    for _, r in read_csv("generic_ballot_polls.csv").iterrows():
        gb_rows.append({"race_id": "US-GB", "pollster": r.get("pollster"), "sponsor_party": r.get("sponsor_party"),
                        "start": r.get("start_date"), "end": r.get("end_date"), "n": r.get("sample_size"),
                        "pop": r.get("population"), "dem": r.get("dem_pct"), "rep": r.get("rep_pct"),
                        "url": r.get("url"), "source": "Research"})
    gb = P.standardize(pd.DataFrame(gb_rows)) if gb_rows else pd.DataFrame()
    gb_recent = gb[gb["end"] >= pd.Timestamp(asof) - pd.Timedelta(days=240)] if not gb.empty else gb
    lv = P.estimate_lv_shift(gb_recent) if not gb.empty else {"LV": 0.0, "RV": 0.0, "A": 0.0}
    he = P.estimate_house_effects(gb_recent, lv, pd.Timestamp(asof)) if not gb.empty else {}
    adj = P.Adjusters(house=he, lv_shift=lv)
    series = P.daily_series(gb, dt.date(2026, 3, 1), asof, adj) if not gb.empty else pd.Series(dtype=float)
    adj.gb_series = series
    own = P.average(gb, asof, adj, half_life=10, trend=False, max_age_days=45, he_transfer=1.0) if not gb.empty else None

    aggs = []
    for a in step(status, "wiki_generic_aggregates", W.generic_ballot_aggregates, []):
        aggs.append({"source": a["aggregator"], "margin": a["dem"] - a["rep"], "dem": a["dem"], "rep": a["rep"],
                     "as_of": a["as_of"], "url": a["source_url"]})
    agr = read_csv("aggregator_national.csv")
    if not agr.empty and "metric" in agr.columns:
        for _, r in agr[agr["metric"].astype(str).str.contains("generic", case=False)].iterrows():
            try:
                aggs.append({"source": str(r["source"]), "margin": float(r["margin"]), "dem": r.get("dem_or_approve"),
                             "rep": r.get("rep_or_disapprove"), "as_of": r.get("as_of"), "url": r.get("url")})
            except (TypeError, ValueError):
                continue
    agg_df = pd.DataFrame(aggs)
    if not agg_df.empty:
        agg_df["key"] = agg_df["source"].str.lower().str.replace(r"[^a-z]", "", regex=True).str[:10]
        agg_df["as_of"] = pd.to_datetime(agg_df["as_of"], errors="coerce")
        agg_df = agg_df.sort_values("as_of").drop_duplicates("key", keep="last")
        agg_df = agg_df[agg_df["as_of"].isna() | (agg_df["as_of"] >= pd.Timestamp(asof) - pd.Timedelta(days=21))]
    agg_median = float(agg_df["margin"].median()) if not agg_df.empty else None

    comps = {}
    poll_vals = [v for v in (own.margin if own and own.margin is not None else None, agg_median) if v is not None]
    if poll_vals:
        raw = float(np.mean(poll_vals))
        comps["polls"] = {"margin": raw + NAT.GB_BIAS_CORRECTION, "sd": NAT.GB_SD, "raw": raw,
                          "own_average": own.margin if own else None, "aggregator_median": agg_median,
                          "n_polls": own.n_polls if own else 0, "bias_correction": NAT.GB_BIAS_CORRECTION}
    mp = NAT.market_popvote(mk)
    mvals = [v.get("median") if v.get("median") is not None else v.get("mean") for v in mp.values()]
    mvals = [v for v in mvals if v is not None]
    if mvals:
        comps["markets"] = {"margin": float(np.mean(mvals)), "sd": NAT.MARKET_SD, "detail": mp}

    appr = [a for a in approval.get("aggregates", []) if a["aggregator"].lower() != "average"]
    approve = float(np.median([a["approve"] for a in appr])) if appr else None
    net = float(np.median([a["approve"] - a["disapprove"] for a in appr])) if appr else None
    hist = read_csv("history_midterms.csv", bases=(STATIC,))
    hist_gv = read_csv("history_house_vote.csv", bases=(STATIC,))
    fund = NAT.fit_fundamentals(hist, hist_gv)
    comps["approval"] = NAT.approval_component(fund, approve)
    comps["specials"] = NAT.specials_component(read_csv("special_elections.csv"), pd.Timestamp(asof))
    days = days_until_election(asof)
    bt = NAT.backtest(hist, hist_gv, fund)
    blend = NAT.blend(comps, days, backtest_sd=bt.get("rmse") if bt else None)

    vh_appr = PF.votehub_approval(vh_raw) if vh_raw else []
    nat = {"asof": asof.isoformat(), "days_out": days, **blend, "components": comps,
           "approval": {"approve": approve, "net": net, "aggregates": approval.get("aggregates", []),
                        "url": approval.get("source_url"), "n_votehub_polls": len(vh_appr)},
           "aggregators": (agg_df.drop(columns=["key"]).assign(as_of=lambda d: d["as_of"].dt.date.astype(str))
                           .to_dict("records") if not agg_df.empty else []),
           "lv_shift": lv, "house_effects": {k: round(v, 2) for k, v in sorted(he.items(), key=lambda kv: -abs(kv[1]))[:30]},
           "own_poll_average": own.margin if own else None, "generic_polls": own.polls[:120] if own else [],
           "chart_polls": ([{"end": e.date().isoformat(), "dem": float(dm), "rep": float(rp), "pollster": pl}
                            for e, dm, rp, pl in gb[gb["end"] >= pd.Timestamp("2026-03-01")][["end", "dem", "rep", "pollster"]].itertuples(index=False)]
                           if not gb.empty else []),
           "n_generic_polls_total": int(len(gb)),
           "series": [{"date": d.date().isoformat(), "margin": round(float(v), 2)} for d, v in series.items()],
           "fundamentals_model": fund, "backtest": bt}
    return nat, adj


# ---------------------------------------------------------------- race builders
def build_statewide(kind, overview, research, resolved, fec_df, mk, adj, N, asof, polls, aggs) -> list[dict]:
    races = []
    for rid, ov in sorted(overview.items()):
        st = rid[:2]
        res = research.get(rid, {})
        rv = resolved[rid]
        d, r = rv["d"], rv["r"]
        fd = FEC.lookup(fec_df, st, "S", d["name"]) if (kind == "SEN" and d) else None
        fr = FEC.lookup(fec_df, st, "S", r["name"]) if (kind == "SEN" and r) else None
        d_info = R._cand_info(d, res, "dem", rv["inc"], fd)
        r_info = R._cand_info(r, res, "rep", rv["inc"], fr)
        if rv.get("hist"):
            for info in (d_info, r_info):
                if info.get("is_incumbent"):
                    info["_inc_history"] = rv["hist"]
        if rv["d_is_ind"] and d:
            d_info["party"] = "I"
            if res.get("other_experience") is not None:
                d_info["experience"] = R._num(res.get("other_experience"))
        fixed = "R" if (not d and r) else ("D" if (d and not r) else None)
        pf = polls[polls["race_id"] == rid] if not polls.empty else polls
        pa = P.average(pf, asof, adj) if (not pf.empty and not fixed) else None
        rec = R.race_record(kind, rid, st, N, ov.get("pvi"), ov.get("ratings", {}), mk, pa, d_info, r_info,
                            rv["inc_party"], rv["inc_run"], rv["appointed"], rv["d_is_ind"],
                            extra={"name": f"{STATES[st]}{' (special)' if rid.endswith('-SP') else ''}",
                                   "incumbent": rv["inc"], "status": ov.get("status"), "appointed": rv["appointed"],
                                   "others": [o for o in rv["others"] if o["party"] not in ("D", "R")][:4],
                                   "poll_list": (pa.polls[:40] if pa else []), "aggregator_averages": aggs.get(rid, []),
                                   "fixed_winner": fixed, "notes": res.get("scandal_notes") or res.get("notes"),
                                   "special_rules": res.get("special_rules"), "wiki": W.page_url(W.race_page_title(rid))})
        if fixed:
            rec["mean"] = 100.0 if fixed == "D" else -100.0
        races.append(rec)
    log.info("%s: %d races built", kind, len(races))
    return races


def build_house(hd, ratings, research, resolved, fec_df, mk, adj, N, asof, polls) -> list[dict]:
    races = []
    for _, row in hd.iterrows():
        did, st = row["district"], row["state"]
        res = research.get(did, {})
        rv = resolved[did]
        d, r = rv["d"], rv["r"]
        num = 0 if row["num"] == 0 else int(row["num"])
        fd = FEC.lookup(fec_df, st, "H", d["name"], num) if d else None
        fr = FEC.lookup(fec_df, st, "H", r["name"], num) if r else None
        d_info = R._cand_info(d, res, "dem", rv["inc"], fd)
        r_info = R._cand_info(r, res, "rep", rv["inc"], fr)
        pa = None
        if not rv["fixed"] and not polls.empty:
            pf = polls[polls["race_id"] == did]
            pa = P.average(pf, asof, adj, half_life=25) if not pf.empty else None
        rec = R.race_record("HOUSE", did, st, N, row["pvi"], ratings.get(did, {}), mk, pa, d_info, r_info,
                            rv["inc_party"], rv["inc_run"], False, False, redistricted=st in REDISTRICTED_STATES,
                            extra={"name": did, "incumbent": rv["inc"], "status": row["status"],
                                   "fixed_winner": rv["fixed"], "poll_list": (pa.polls[:20] if pa else []),
                                   "competitive": did in ratings or did in research, "notes": res.get("notes")})
        if rv["fixed"]:
            rec["mean"] = 100.0 if rv["fixed"] == "D" else -100.0
        races.append(rec)
    log.info("HOUSE: %d districts built", len(races))
    return races


def _jsonable(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=SIM.N_SIMS)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--offline", action="store_true", help="skip slow optional scrapers (270toWin)")
    args = ap.parse_args(argv)
    asof = TODAY
    log.info("=== forecast run for %s (%d days out) ===", asof, days_until_election(asof))
    status: dict = {}

    mk = step(status, "markets", MK.fetch_all, {})
    for ex in mk.values():   # FL and OH have no regular 2026 Senate race
        for st in ("FL", "OH"):
            if f"{st}-SEN" in ex and f"{st}-SEN-SP" not in ex:
                ex[f"{st}-SEN-SP"] = ex.pop(f"{st}-SEN")
    vh_raw = step(status, "votehub", PF.votehub_raw, [])
    approval = step(status, "wiki_approval", W.approval, {"aggregates": [], "polls": []})
    fec_df = step(status, "fec", FEC.fetch, pd.DataFrame(columns=FEC.COLS + ["office", "district", "last", "first", "party"]))
    fred = step(status, "fred", FRED.fetch, {})
    econ = FRED.summarize(fred) if fred else {}

    nat, adj = build_national(status, mk, approval, vh_raw, asof)
    N = nat["margin"]

    sen_ov = step(status, "wiki_senate", lambda: W.statewide_overview("SEN"), {})
    gov_ov = step(status, "wiki_governor", lambda: W.statewide_overview("GOV"), {})
    hd = step(status, "wiki_house", W.house_districts, pd.DataFrame())
    hr = step(status, "wiki_house_ratings", W.house_ratings, {})
    sen_res = research_rows(read_csv("senate_candidates.csv"), "race_id")
    gov_res = research_rows(read_csv("governor_candidates.csv"), "race_id")
    house_res = research_rows(read_csv("house_competitive.csv"), "district")

    resolved = {rid: resolve_statewide(rid, ov, sen_res.get(rid, {})) for rid, ov in sen_ov.items()}
    resolved.update({rid: resolve_statewide(rid, ov, gov_res.get(rid, {})) for rid, ov in gov_ov.items()})
    for _, row in hd.iterrows():
        resolved[row["district"]] = resolve_house(row, house_res.get(row["district"], {}), row["district"] in hr)
    statewide_noms = {rid: (rv["d"]["name"], rv["r"]["name"]) for rid, rv in resolved.items()
                      if "-SEN" in rid or rid.endswith("-GOV") if rv["d"] and rv["r"]}
    house_noms = {rid: (rv["d"]["name"], rv["r"]["name"]) for rid, rv in resolved.items()
                  if re.match(r"^[A-Z]{2}-(\d\d|AL)$", rid) and rv["d"] and rv["r"] and not rv["fixed"]
                  and (rid in hr or rid in house_res)}
    party_maps = {rid: {last_name(c["name"]): c["party"] for c in ov.get("candidates", [])}
                  for rid, ov in list(sen_ov.items()) + list(gov_ov.items())}
    for _, row in hd.iterrows():
        party_maps[row["district"]] = {last_name(c["name"]): c["party"] for c in (row["candidates"] or [])}
    polls, aggs = gather_polls(status, statewide_noms, house_noms, vh_raw, offline=args.offline, party_maps=party_maps)

    sen = build_statewide("SEN", sen_ov, sen_res, resolved, fec_df, mk, adj, N, asof, polls, aggs)
    gov = build_statewide("GOV", gov_ov, gov_res, resolved, fec_df, mk, adj, N, asof, polls, aggs)
    house = build_house(hd, hr, house_res, resolved, fec_df, mk, adj, N, asof, polls) if not hd.empty else []

    races = sen + gov + house
    sims = SIM.run(races, nat, n=args.sims)
    SIM.summarize_races(races, sims)
    house_ch = SIM.chamber(races, sims, "HOUSE", 0, 0, 0, control_d=218, control_r=218)
    sen_ch = SIM.chamber(races, sims, "SEN", SENATE_NOT_UP["D"], SENATE_NOT_UP["R"], SENATE_NOT_UP["I"],
                         control_d=51, control_r=50)
    gov_ch = SIM.chamber(races, sims, "GOV", GOV_NOT_UP["D"], GOV_NOT_UP["R"], 0, control_d=26, control_r=25)
    sen_ch["tipping_point"] = SIM.tipping_point(races, sims, "SEN", SENATE_NOT_UP["D"], 51)
    house_ch["tipping_point"] = SIM.tipping_point(races, sims, "HOUSE", 0, 218)
    natvote = sims["national"]
    nat["sim"] = {"mean": float(natvote.mean()), "p10": float(np.percentile(natvote, 10)),
                  "p90": float(np.percentile(natvote, 90)), "p025": float(np.percentile(natvote, 2.5)),
                  "p975": float(np.percentile(natvote, 97.5)), "p_dem_win_popvote": float((natvote > 0).mean())}
    dw, sd_, sr_ = house_ch["d_wins_draws"], sen_ch["d_wins_draws"], sen_ch["r_wins_draws"]
    curve = []
    for lo in np.arange(-6, 21, 1.0):
        msk = (natvote >= lo) & (natvote < lo + 1)
        if msk.sum() > 80:
            curve.append({"margin": float(lo + 0.5), "d_seats": float(dw[msk].mean()),
                          "p_control": float((dw[msk] >= 218).mean()), "n": int(msk.sum())})
    house_ch["seats_by_national_vote"] = curve
    joint = {"both_d": float(((dw >= 218) & (sd_ >= 51)).mean()),
             "house_d_senate_r": float(((dw >= 218) & (sr_ >= 50)).mean()),
             "house_r_senate_d": float(((dw < 218) & (sd_ >= 51)).mean()),
             "both_r": float(((dw < 218) & (sr_ >= 50)).mean())}
    for ch in (house_ch, sen_ch, gov_ch):
        for k in ("d_wins_draws", "r_wins_draws", "i_wins_draws"):
            ch.pop(k, None)

    news = {}
    if not args.no_news:
        qs = [{"race_id": r["race_id"], "query": f'"{r["dem"]["name"]}" "{r["rep"]["name"]}"'}
              for r in sen + gov if not r.get("fixed_winner") and r.get("dem") and r.get("rep") and 0.02 < r["p_dem"] < 0.98]
        qs += [{"race_id": r["race_id"], "query": f'"{r["dem"]["name"]}" "{r["rep"]["name"]}"'}
               for r in house if not r.get("fixed_winner") and r.get("dem") and r.get("rep") and 0.15 < r["p_dem"] < 0.85]
        news = step(status, "news", lambda: NEWS.race_news(qs), {})

    other = read_csv("other_forecasts.csv")
    out = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "asof": asof.isoformat(),
           "election_day": "2026-11-03", "days_out": days_until_election(asof), "n_sims": args.sims,
           "national": nat, "economy": econ, "house": house_ch, "senate": sen_ch, "governors": gov_ch,
           "joint": joint, "races": races, "news": news, "fetch_status": status,
           "n_polls": {"generic": nat.get("n_generic_polls_total", 0), "races": int(len(polls))},
           "markets_control": {ex: {k: v.get("p_dem") for k, v in m.items() if k in ("HOUSE-CONTROL", "SENATE-CONTROL")}
                               for ex, m in mk.items()},
           "other_forecasts": other.replace({np.nan: None}).to_dict("records") if not other.empty else [],
           "research": {"national_environment": load_json(RESEARCH / "national_environment.json", {}),
                        "special_elections": read_csv("special_elections.csv").replace({np.nan: None}).to_dict("records")}}
    out = json.loads(json.dumps(out, default=_jsonable))
    save_json(PROCESSED / "forecast.json", out)
    save_json(PROCESSED / f"forecast_{asof.isoformat()}.json", out)
    from pipeline import export
    export.write_site(out)
    log.info("House: D control %.1f%% (D seats %.1f) | Senate: D %.1f%% R %.1f%% | Gov D %.1f | national D%+.1f (sd %.2f)",
             100 * house_ch["p_d_control"], house_ch["d_mean"], 100 * sen_ch["p_d_control"], 100 * sen_ch["p_r_control"],
             gov_ch["d_mean"], nat["margin"], nat["sd"])
    return out


if __name__ == "__main__":
    main()

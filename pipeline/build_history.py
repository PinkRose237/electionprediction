"""Builds the historical calibration tables used by the fundamentals model and backtests.

Sources (copied into data/static/history_src/):
  * house_infobox.jsonl     - national House popular vote & seats, 1946-2024 (Wikipedia election infoboxes)
  * midterm_last_polls.json - Gallup's last pre-election presidential approval before each midterm
                              (UCSB American Presidency Project, Gallup series)
  * data/static/fte/raw_polls.csv - FiveThirtyEight's archive of polls in the final 21 days (1998-2022),
                              used to compute each cycle's final generic-ballot polling average
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from pipeline.common import STATIC, log

SRC = STATIC / "history_src"
PRES_PARTY = {"harry-s-truman": "D", "dwight-d-eisenhower": "R", "john-f-kennedy": "D", "lyndon-b-johnson": "D",
              "richard-m-nixon": "R", "gerald-r-ford": "R", "jimmy-carter": "D", "ronald-reagan": "R",
              "george-bush": "R", "william-j-clinton": "D", "george-w-bush": "R", "barack-obama": "D",
              "donald-j-trump": "R", "joseph-r-biden": "D"}


def _pct(s):
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    return float(m.group(1)) if m else None


def house_votes() -> pd.DataFrame:
    rows = []
    for line in open(SRC / "house_infobox.jsonl"):
        rec = json.loads(line)
        yr = int(re.search(r"(\d{4})_United_States_House", rec["file"]).group(1))
        d = r = ds = rs = None
        for b in rec["blocks"]:
            parties = b.get("Party") or []
            for i, p in enumerate(parties):
                pc = (b.get("Percentage") or [None] * len(parties))[i] if b.get("Percentage") else None
                seats = (b.get("Seats won") or [None] * len(parties))[i] if b.get("Seats won") else None
                if p.startswith("Democratic") and d is None:
                    d, ds = _pct(pc), _pct(seats)
                elif p.startswith("Republican") and r is None:
                    r, rs = _pct(pc), _pct(seats)
        rows.append({"year": yr, "house_dem_pct": d, "house_rep_pct": r, "house_dem_seats": ds, "house_rep_seats": rs,
                     "source": rec["source"]})
    df = pd.DataFrame(rows).sort_values("year")
    df["house_popvote_margin"] = df["house_dem_pct"] - df["house_rep_pct"]
    return df


def generic_finals() -> pd.DataFrame:
    """Final generic-ballot average per cycle from 538's archive: mean of likely-voter polls in the final
    14 days (falls back to all polls if fewer than 3 LV polls). D minus R."""
    rp = pd.read_csv(STATIC / "fte" / "raw_polls.csv")
    g = rp[rp["type_simple"] == "House-G-US"].copy()
    sgn = np.where((g.cand1_party == "DEM") & (g.cand2_party == "REP"), 1, np.where((g.cand1_party == "REP") & (g.cand2_party == "DEM"), -1, 0))
    g = g[sgn != 0]
    g["m"] = g["margin_poll"] * sgn[sgn != 0]
    from pipeline.model.polls import PARTISAN_ADJ, PARTISAN_WEIGHT, quality_weight
    out = []
    for cyc, x in g.groupby("cycle"):
        # the same weighting the live model uses: recency, sample size, 538 grade, partisan adjustment, crowding
        part = x["partisan"].fillna("")
        m = x["m"] - np.where(part == "DEM", PARTISAN_ADJ, 0) + np.where(part == "REP", PARTISAN_ADJ, 0)
        w = 0.5 ** (x["time_to_election"] / 10) * np.sqrt(x["samplesize"].fillna(600).clip(150, 2500) / 600)
        w = w * x["pollster"].map(quality_weight) * np.where(part != "", PARTISAN_WEIGHT, 1.0)
        w = w / np.sqrt(x.groupby("pollster")["pollster"].transform("count"))
        out.append({"year": int(cyc), "generic_final_avg": round(float((w * m).sum() / w.sum()), 2), "n_polls": int(len(x))})
    return pd.DataFrame(out)


def build() -> None:
    hv = house_votes()
    ap = json.load(open(SRC / "midterm_last_polls.json"))
    gf = generic_finals()
    mids = []
    for y, v in sorted(ap.items()):
        y = int(y)
        row = hv[hv["year"] == y]
        if row.empty:
            continue
        row = row.iloc[0]
        mids.append({"year": y, "president": v["pres"], "pres_party": PRES_PARTY.get(v["pres"]),
                     "gallup_approval": float(v["approve"]), "gallup_disapproval": float(v["disapprove"]),
                     "house_popvote_dem_pct": row["house_dem_pct"], "house_popvote_rep_pct": row["house_rep_pct"],
                     "house_popvote_margin": round(row["house_popvote_margin"], 2),
                     "house_dem_seats_after": row["house_dem_seats"], "house_rep_seats_after": row["house_rep_seats"],
                     "sources": f"{row['source']}; https://www.presidency.ucsb.edu/statistics/data/presidential-job-approval"})
    m = pd.DataFrame(mids).merge(gf.rename(columns={"generic_final_avg": "generic_ballot_final_avg_margin"}), on="year", how="left")
    m.to_csv(STATIC / "history_midterms.csv", index=False)
    gv = hv.merge(gf, on="year", how="left")
    gv["cycle_type"] = np.where(gv["year"] % 4 == 0, "presidential", "midterm")
    gv = gv.rename(columns={"house_popvote_margin": "house_popvote_margin_actual", "generic_final_avg": "final_generic_avg_margin_other"})
    gv.to_csv(STATIC / "history_house_vote.csv", index=False)
    log.info("history: %d midterms, %d House elections", len(m), len(gv))


if __name__ == "__main__":
    build()
    print(pd.read_csv(STATIC / "history_midterms.csv")[["year", "pres_party", "gallup_approval", "house_popvote_margin", "generic_ballot_final_avg_margin"]].to_string())
    hv = pd.read_csv(STATIC / "history_house_vote.csv")
    print(hv[hv.year >= 1994][["year", "house_popvote_margin_actual", "final_generic_avg_margin_other"]].to_string())

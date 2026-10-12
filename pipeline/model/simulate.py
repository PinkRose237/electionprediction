"""Correlated Monte Carlo simulation of every race.

simulated margin_i = mean_i
                   + national shock        (Student-t, df=5; shared by every race; sd = national sd)
                   + office shock          (shared within Senate / governor / House)
                   + regional shock        (shared within a Census division, all offices)
                   + partisan-lean factor  (errors that scale with how red/blue the place is, as in
                                            2016/2020 when polls missed most in Trump-leaning states)
                   + race-specific error   (Student-t, df=6)
Office-shock sizes are calibrated so the total common error matches the 538 archive (cycle-level sd of
competitive-race poll error: Senate 3.3, governor 3.75).
"""
from __future__ import annotations

import numpy as np

from pipeline.common import log

N_SIMS = 40000
OFFICE_SD = {"SEN": 1.8, "GOV": 2.2, "HOUSE": 1.0}
REGION_SD = 1.5
LEAN_FACTOR_SD = 1.2     # per 15 points of partisan lean
NAT_DF = 5
RACE_DF = 6


def _t(rng, df, size):
    return rng.standard_t(df, size=size) * np.sqrt((df - 2) / df)


def run(races: list[dict], national: dict, n: int = N_SIMS, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    divisions = sorted({r["division"] for r in races})
    div_idx = {d: i for i, d in enumerate(divisions)}
    nat = _t(rng, NAT_DF, n) * national["sd"]
    office = {k: rng.normal(0, s, n) for k, s in OFFICE_SD.items()}
    region = rng.normal(0, REGION_SD, (n, len(divisions)))
    lean_f = rng.normal(0, LEAN_FACTOR_SD, n)
    margins = np.empty((n, len(races)), dtype=np.float32)
    for j, r in enumerate(races):
        if r.get("fixed_winner"):
            margins[:, j] = 100.0 if r["fixed_winner"] == "D" else -100.0
            continue
        # region shock already supplies part of the race-specific variance; avoid double counting
        race_sd = np.sqrt(max(r["sd_race"] ** 2 - REGION_SD ** 2 * 0.5, 4.0))
        eps = _t(rng, RACE_DF, n) * race_sd
        margins[:, j] = (r["mean"] + nat + office[r["office"]] + region[:, div_idx[r["division"]]]
                         + lean_f * (r["lean"] / 15.0) + eps)
    natvote = national["margin"] + nat
    log.info("simulated %d races x %d draws", len(races), n)
    return {"margins": margins, "national": natvote}


def summarize_races(races: list[dict], sims: dict) -> None:
    m = sims["margins"]
    for j, r in enumerate(races):
        col = m[:, j]
        r["p_dem"] = float((col > 0).mean())
        if r.get("fixed_winner"):
            r["sim"] = {"mean": r["mean"], "p10": r["mean"], "p90": r["mean"], "p025": r["mean"], "p975": r["mean"]}
            continue
        q = np.percentile(col, [2.5, 10, 50, 90, 97.5])
        r["sim"] = {"mean": float(col.mean()), "p025": float(q[0]), "p10": float(q[1]), "median": float(q[2]),
                    "p90": float(q[3]), "p975": float(q[4]), "sd": float(col.std())}


def chamber(races: list[dict], sims: dict, office: str, base_d: int, base_r: int, base_i: int = 0,
            control_d: int = 0, control_r: int = 0) -> dict:
    """Seat distribution for a chamber. base_* are seats not up (Senate) or 0 (House)."""
    idx = [j for j, r in enumerate(races) if r["office"] == office]
    m = sims["margins"][:, idx]
    rr = [races[j] for j in idx]
    d_side = m > 0
    is_ind = np.array([bool(r.get("dem_is_independent")) for r in rr])
    d_wins = d_side[:, ~is_ind].sum(axis=1) + base_d
    i_wins = d_side[:, is_ind].sum(axis=1) + base_i
    r_wins = (~d_side).sum(axis=1) + base_r
    total = base_d + base_r + base_i + len(rr)
    out = {"office": office, "total_seats": int(total),
           "d_mean": float(d_wins.mean()), "r_mean": float(r_wins.mean()), "i_mean": float(i_wins.mean()),
           "d_median": float(np.median(d_wins)), "d_p10": float(np.percentile(d_wins, 10)),
           "d_p90": float(np.percentile(d_wins, 90)), "d_p025": float(np.percentile(d_wins, 2.5)),
           "d_p975": float(np.percentile(d_wins, 97.5))}
    vals, counts = np.unique(d_wins, return_counts=True)
    out["d_hist"] = {int(v): float(c / len(d_wins)) for v, c in zip(vals, counts)}
    if control_d:
        out["p_d_control"] = float((d_wins >= control_d).mean())
        out["p_r_control"] = float((r_wins >= control_r).mean())
        out["p_other"] = max(0.0, 1 - out["p_d_control"] - out["p_r_control"])
    out["d_wins_draws"] = d_wins  # removed before export
    out["r_wins_draws"] = r_wins
    out["i_wins_draws"] = i_wins
    return out


def tipping_point(races: list[dict], sims: dict, office: str, base_d: int, need: int, max_draws: int = 10000) -> dict:
    """Race that delivers the decisive seat when races are ordered from most to least Democratic."""
    idx = [j for j, r in enumerate(races) if r["office"] == office and not r.get("fixed_winner")]
    fixed_d = sum(1 for r in races if r["office"] == office and r.get("fixed_winner") == "D")
    m = sims["margins"][:max_draws, idx]
    need_from_contested = need - base_d - fixed_d
    counts = {}
    if need_from_contested <= 0 or need_from_contested > len(idx):
        return {}
    order = np.argsort(-m, axis=1)
    pivot = order[:, need_from_contested - 1]
    for k in pivot:
        rid = races[idx[k]]["race_id"]
        counts[rid] = counts.get(rid, 0) + 1
    tot = len(pivot)
    return {k: v / tot for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12]}

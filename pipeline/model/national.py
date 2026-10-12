"""National environment: expected national House popular-vote margin (D minus R) on Election Day.

Components (each converted to a D-R national House margin with an error sd):
  1. Generic-ballot polls  - our own adjusted average, plus the median of published aggregators,
                             minus a historical bias correction.
  2. Prediction markets    - Kalshi and Polymarket national popular-vote ladders.
  3. Approval fundamentals - regression of the president's party's House-vote swing on presidential
                             approval, fit on every midterm 1946-2022 (leave-one-out validated).
  4. Special elections     - average 2025-26 overperformance vs. the 2024 presidential baseline,
                             scaled by the historical pass-through ratio.
Weights are inverse-variance with a correlation discount, and the final sd is taken from the
backtest of the same procedure rather than from the (overconfident) independence formula.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline.common import log

# 2024 results (two-party-era baselines) used to translate swings into 2026 levels.
PRES_2024_MARGIN = -1.48     # Trump +1.48 national popular vote (certified, D minus R)
HOUSE_2024_MARGIN = -2.6     # Republicans won the national House vote by ~2.6 points (D minus R)

# Historical error of a final generic-ballot polling average vs the actual national House vote
# (538 archive & RCP finals, midterms 1998-2022): mean +2.0 toward Democrats for all-poll averages,
# ~+0.6 for LV-heavy averages. We correct by the conservative middle of that range.
GB_BIAS_CORRECTION = -1.0
GB_SD = 2.6                  # sd of final generic-ballot average error (incl. bias uncertainty)
MARKET_SD = 3.0
SPECIALS_PASS_THROUGH = 0.45  # national swing ~= 0.45 x special-election overperformance (2017-18, 2021-22, 2025)
SPECIALS_SD = 5.0


def time_drift_sd(days: int) -> float:
    """How much the true environment can still move before Election Day (pts).
    Generic-ballot averages moved ~1.5 pts (sd) over the final month in 2006-2022."""
    return 0.30 * math.sqrt(max(days, 0))


def market_popvote(markets: dict) -> dict:
    """Market-implied national House margin from Kalshi's survival ladder and Polymarket's buckets."""
    out = {}
    k = (markets.get("Kalshi") or {}).get("HOUSE-POPVOTE")
    if k and k.get("ladder"):
        lad = sorted(k["ladder"], key=lambda x: x["threshold"])
        ts = [lad[0]["threshold"] - 8.0] + [x["threshold"] for x in lad] + [lad[-1]["threshold"] + 4.0]
        ss = [0.995] + [min(0.995, max(0.005, x["p_at_least"])) for x in lad] + [0.0]
        # enforce monotone survival
        for i in range(1, len(ss)):
            ss[i] = min(ss[i], ss[i - 1])
        mean = ts[0] + sum((ts[i + 1] - ts[i]) * (ss[i] + ss[i + 1]) / 2 for i in range(len(ts) - 1))
        med = None
        for i in range(len(ts) - 1):
            if ss[i] >= 0.5 >= ss[i + 1]:
                med = ts[i] + (ts[i + 1] - ts[i]) * (ss[i] - 0.5) / max(ss[i] - ss[i + 1], 1e-9)
                break
        out["Kalshi"] = {"mean": mean, "median": med, "url": k.get("url")}
    p = (markets.get("Polymarket") or {}).get("HOUSE-POPVOTE")
    if p and p.get("buckets"):
        b = p["buckets"]
        tot = sum(x["p"] for x in b)
        mean = sum((x["lo"] + x["hi"]) / 2 * x["p"] for x in b) / tot
        cum, med = 0.0, None
        for x in sorted(b, key=lambda x: x["lo"]):
            if cum + x["p"] / tot >= 0.5:
                med = x["lo"] + (x["hi"] - x["lo"]) * (0.5 - cum) / (x["p"] / tot)
                break
            cum += x["p"] / tot
        out["Polymarket"] = {"mean": mean, "median": med, "url": p.get("url")}
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _hist_frame(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    for c in ("year", "gallup_approval", "house_popvote_margin", "house_popvote_dem_pct", "house_popvote_rep_pct"):
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    if "house_popvote_margin" not in h.columns or h["house_popvote_margin"].isna().all():
        h["house_popvote_margin"] = h.get("house_popvote_dem_pct") - h.get("house_popvote_rep_pct")
    h["house_popvote_margin"] = h["house_popvote_margin"].fillna(h.get("house_popvote_dem_pct") - h.get("house_popvote_rep_pct"))
    h = h.dropna(subset=["year", "gallup_approval", "house_popvote_margin"])
    h = h[h["pres_party"].isin(["D", "R"])]
    h["year"] = h["year"].astype(int)
    return h.sort_values("year")


def _ols_loo(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    loo = []
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        loo.append(float(y[i] - X[i] @ b))
    return beta, np.array(loo)


def prev_pres_house_margins(hist_gv: pd.DataFrame | None) -> dict[int, float]:
    out = {}
    if hist_gv is None or hist_gv.empty:
        return out
    for _, r in hist_gv.iterrows():
        y, m = _num(r.get("year")), _num(r.get("house_popvote_margin_actual"))
        if y is not None and m is not None:
            out[int(y)] = m
    return out


def fit_fundamentals(hist: pd.DataFrame, hist_gv: pd.DataFrame | None) -> dict | None:
    """Two approval models of the president's party's national House margin, both validated LOO:
      level : margin = a + b*(approval-45) + c*era  (era = +1 for D presidents pre-1994, -1 for R
              presidents pre-1994, capturing the old structural Democratic House majority)
      swing : change in margin vs. the previous presidential-year House vote = a + b*(approval-45)
    """
    if hist is None or hist.empty or "gallup_approval" not in hist.columns:
        return None
    h = _hist_frame(hist)
    if len(h) < 8:
        return None
    sign = np.where(h["pres_party"] == "D", 1.0, -1.0)
    y = sign * h["house_popvote_margin"].to_numpy()
    x = h["gallup_approval"].to_numpy() - 45.0
    era = np.where(h["year"] < 1994, sign, 0.0)
    Xl = np.column_stack([np.ones_like(x), x, era])
    bl, loo_l = _ols_loo(Xl, y)
    out = {"level": {"intercept": float(bl[0]), "slope": float(bl[1]), "era": float(bl[2]),
                     "loo_rmse": float(np.sqrt(np.mean(loo_l ** 2))), "n": int(len(y)),
                     "years": h["year"].tolist(), "approval": h["gallup_approval"].tolist(),
                     "pres_party_margin": y.round(2).tolist(), "loo_resid": loo_l.round(2).tolist()}}
    prev = prev_pres_house_margins(hist_gv)
    rows = [(yr, ap, sg * (m - prev[yr - 2])) for yr, ap, sg, m in
            zip(h["year"], h["gallup_approval"], sign, h["house_popvote_margin"]) if (yr - 2) in prev]
    if len(rows) >= 5:
        yrs, ap, sw = (np.array(v, dtype=float) for v in zip(*rows))
        Xs = np.column_stack([np.ones_like(ap), ap - 45.0])
        bs, loo_s = _ols_loo(Xs, sw)
        out["swing"] = {"intercept": float(bs[0]), "slope": float(bs[1]), "loo_rmse": float(np.sqrt(np.mean(loo_s ** 2))),
                        "n": int(len(sw)), "years": yrs.astype(int).tolist(), "loo_resid": loo_s.round(2).tolist()}
    out["house_2024_margin"] = prev.get(2024, HOUSE_2024_MARGIN)
    return out


def approval_component(model: dict | None, approval: float | None, pres_party: str = "R") -> dict | None:
    if not model or approval is None:
        return None
    sign = 1.0 if pres_party == "D" else -1.0
    preds = {}
    lv = model["level"]
    preds["level"] = (sign * (lv["intercept"] + lv["slope"] * (approval - 45.0)), lv["loo_rmse"])
    if "swing" in model:
        sw = model["swing"]
        swing = sw["intercept"] + sw["slope"] * (approval - 45.0)
        preds["swing"] = (model["house_2024_margin"] + sign * swing, sw["loo_rmse"])
    w = {k: 1 / v[1] ** 2 for k, v in preds.items()}
    margin = sum(w[k] * preds[k][0] for k in preds) / sum(w.values())
    sd = max(min(v[1] for v in preds.values()), 3.0)
    # approval far outside the historical range (Gallup midterm lows ~38-40) -> extrapolation risk
    lo = min(lv["approval"])
    if approval < lo:
        sd = math.sqrt(sd ** 2 + (0.5 * (lo - approval)) ** 2)
    return {"margin": margin, "sd": sd, "approval": approval, "models": {k: round(v[0], 2) for k, v in preds.items()}}


def backtest(hist: pd.DataFrame, hist_gv: pd.DataFrame | None, model: dict | None) -> dict | None:
    """Out-of-sample test of the polls + approval blend on past midterms (1998-2022).
    Each year's bias correction and approval model are estimated without that year."""
    if hist is None or hist.empty or not model:
        return None
    h = _hist_frame(hist)
    gbcol = None
    for c in ("generic_ballot_final_avg_margin", "fte_generic_final"):
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    h["gb"] = h.get("generic_ballot_final_avg_margin")
    if "fte_generic_final" in h.columns:
        h["gb"] = h["gb"].fillna(h["fte_generic_final"])
    if hist_gv is not None and not hist_gv.empty:
        gv = {int(_num(r["year"])): _num(r.get("final_generic_avg_margin_rcp")) or _num(r.get("final_generic_avg_margin_other"))
              for _, r in hist_gv.iterrows() if _num(r.get("year"))}
        h["gb"] = h["gb"].fillna(h["year"].map(gv))
    t = h.dropna(subset=["gb"])
    t = t[t["year"] >= 1994]
    if len(t) < 4:
        return None
    rows = []
    sign_all = np.where(h["pres_party"] == "D", 1.0, -1.0)
    for _, r in t.iterrows():
        yr = int(r["year"])
        others = t[t["year"] != yr]
        bias = float((others["gb"] - others["house_popvote_margin"]).mean()) * 0.5   # shrunk 50%
        polls_est = float(r["gb"]) - bias
        m = h["year"] != yr
        sign = np.where(h["pres_party"][m] == "D", 1.0, -1.0)
        y = sign * h["house_popvote_margin"][m].to_numpy()
        x = h["gallup_approval"][m].to_numpy() - 45.0
        era = np.where(h["year"][m] < 1994, sign, 0.0)
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x, era]), y, rcond=None)
        s = 1.0 if r["pres_party"] == "D" else -1.0
        fund_est = s * (b[0] + b[1] * (r["gallup_approval"] - 45.0))
        wp, wf = 1 / GB_SD ** 2, 0.8 / max(model["level"]["loo_rmse"], 3.0) ** 2
        blend_est = (wp * polls_est + wf * fund_est) / (wp + wf)
        act = float(r["house_popvote_margin"])
        rows.append({"year": yr, "actual": round(act, 2), "polls": round(polls_est, 2), "fundamentals": round(fund_est, 2),
                     "blend": round(blend_est, 2), "err_polls": round(polls_est - act, 2),
                     "err_fund": round(fund_est - act, 2), "err_blend": round(blend_est - act, 2)})
    df = pd.DataFrame(rows)
    res = {"years": rows,
           "rmse": float(np.sqrt((df["err_blend"] ** 2).mean())), "mae": float(df["err_blend"].abs().mean()),
           "rmse_polls": float(np.sqrt((df["err_polls"] ** 2).mean())), "mae_polls": float(df["err_polls"].abs().mean()),
           "rmse_fund": float(np.sqrt((df["err_fund"] ** 2).mean())), "mae_fund": float(df["err_fund"].abs().mean()),
           "n": int(len(df))}
    log.info("national backtest (%d midterms): blend MAE %.2f RMSE %.2f | polls-only MAE %.2f | fundamentals MAE %.2f",
             res["n"], res["mae"], res["rmse"], res["mae_polls"], res["mae_fund"])
    return res


def specials_component(specials: pd.DataFrame | None, asof: pd.Timestamp) -> dict | None:
    if specials is None or specials.empty or "swing" not in specials.columns:
        return None
    s = specials.copy()
    s["swing"] = pd.to_numeric(s["swing"], errors="coerce")
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["swing", "date"])
    s = s[s["district"].astype(str).str.upper() != "AVERAGE"]
    if len(s) < 5:
        return None
    age = (asof - s["date"]).dt.days.clip(lower=0)
    w = 0.5 ** (age / 180.0)   # half-life of six months
    # winsorize extreme swings (tiny-turnout specials)
    sw = s["swing"].clip(s["swing"].quantile(0.05), s["swing"].quantile(0.95))
    avg = float((w * sw).sum() / w.sum())
    margin = PRES_2024_MARGIN + SPECIALS_PASS_THROUGH * avg
    return {"margin": margin, "sd": SPECIALS_SD, "avg_swing": avg, "n": int(len(s))}


def blend(components: dict[str, dict], days_out: int, backtest_sd: float | None = None) -> dict:
    """Inverse-variance blend with a correlation discount for the poll-derived signals."""
    corr_discount = {"polls": 1.0, "markets": 0.5, "approval": 0.8, "specials": 0.8}
    ws, num = {}, 0.0
    for k, c in components.items():
        if c is None or c.get("margin") is None:
            continue
        w = corr_discount.get(k, 0.7) / c["sd"] ** 2
        ws[k] = w
    tot = sum(ws.values())
    mean = sum(ws[k] * components[k]["margin"] for k in ws) / tot
    indep_sd = 1 / math.sqrt(tot)
    # errors share causes (all can miss a late shift together), so never report less than the
    # backtested error of this blend
    struct_sd = max(indep_sd, backtest_sd or 0.0, 1.8)
    sd = math.sqrt(struct_sd ** 2 + time_drift_sd(days_out) ** 2)
    weights = {k: v / tot for k, v in ws.items()}
    log.info("national blend: D%+.2f sd %.2f weights %s", mean, sd, {k: round(v, 2) for k, v in weights.items()})
    return {"margin": mean, "sd": sd, "struct_sd": struct_sd, "drift_sd": time_drift_sd(days_out), "weights": weights}

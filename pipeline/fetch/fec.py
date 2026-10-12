"""FEC bulk 'all candidates' summary (weball26.zip): receipts and cash on hand for every House and
Senate candidate, updated by the FEC roughly daily."""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from pipeline.common import Http, log, norm_name, raw_dir

URL = "https://www.fec.gov/files/bulk-downloads/2026/weball26.zip"
COLS = ["CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS",
        "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB",
        "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY",
        "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION",
        "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PRECENT",
        "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS"]


def fetch() -> pd.DataFrame:
    cache = raw_dir("fec") / "weball26.zip"
    if not cache.exists():
        r = Http().get(URL, timeout=120)
        cache.write_bytes(r.content)
    with zipfile.ZipFile(cache) as z:
        txt = z.read(z.namelist()[0]).decode("latin-1")
    df = pd.read_csv(io.StringIO(txt), sep="|", header=None, names=COLS, dtype=str)
    for c in ("TTL_RECEIPTS", "COH_COP", "TTL_DISB", "TTL_INDIV_CONTRIB", "CAND_CONTRIB", "CAND_LOANS"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["office"] = df["CAND_ID"].str[0]
    df["district"] = pd.to_numeric(df["CAND_OFFICE_DISTRICT"], errors="coerce")
    # FEC names are "LAST, FIRST MIDDLE"
    df["last"] = df["CAND_NAME"].str.split(",").str[0].map(lambda s: norm_name(s or "").split()[-1] if norm_name(s or "") else "")
    df["first"] = df["CAND_NAME"].str.split(",").str[1].fillna("").map(lambda s: norm_name(s).split()[0] if norm_name(s) else "")
    df["party"] = df["CAND_PTY_AFFILIATION"].map(lambda p: "D" if p in ("DEM", "DFL") else "R" if p == "REP" else "I" if p in ("IND", "NPA", "UN", "NNE") else p)
    log.info("fec: %d candidate rows (coverage end %s)", len(df), df["CVG_END_DT"].dropna().max())
    return df


def lookup(df: pd.DataFrame, state: str, office: str, name: str, district: int | None = None) -> dict | None:
    """Find a candidate's money by state/office/name. office 'S' or 'H'."""
    if not name:
        return None
    parts = norm_name(name).split()
    if not parts:
        return None
    last, first = parts[-1], parts[0]
    sub = df[(df["office"] == office) & (df["CAND_OFFICE_ST"] == state)]
    if office == "H" and district is not None:
        sub = sub[(sub["district"] == district) | (sub["district"].isna())]
    cand = sub[sub["last"] == last]
    if len(cand) > 1:
        c2 = cand[cand["first"].str[:3] == first[:3]]
        if len(c2):
            cand = c2
    if cand.empty:
        # hyphenated/compound surnames: try any token match
        cand = sub[sub["CAND_NAME"].str.lower().str.contains(last, na=False)]
    if cand.empty:
        return None
    row = cand.sort_values("TTL_RECEIPTS", ascending=False).iloc[0]
    return {"fec_id": row["CAND_ID"], "receipts": float(row["TTL_RECEIPTS"]), "cash": float(row["COH_COP"]),
            "self_funding": float(row["CAND_CONTRIB"] + row["CAND_LOANS"]), "through": row["CVG_END_DT"]}

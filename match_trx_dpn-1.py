"""
Match customers in trx_all (7,000 rows) against the dpn table (72,000 rows)
and enrich trx_all with namadpn / alasan / alasantambahan when a match is
found.

Requirements:
    pip install pandas pyarrow rapidfuzz --break-system-packages

Matching logic:
  - If ktp OR npwp already matches nomorid exactly -> ID_MATCH straight away
    (no need to check name/dob further). If name+dob also line up on top of
    that, it's a PERFECT_MATCH.
  - Otherwise, fuzzy-match customername vs namadpn using rapidfuzz. Because
    trx (7k) x dpn (72k) is ~500M pairwise comparisons, this uses
    rapidfuzz.process.cdist, which is a vectorized/parallel C++ routine
    built exactly for this "compare every row against every row" case
    (much faster than looping row by row).
  - dob match is used as a supporting signal, not a hard requirement, since
    only name+dob+id together define a PERFECT_MATCH.
  - Case, extra spaces, dots, dashes etc. are stripped BEFORE any
    comparison so they never count as real differences.

  When only the name matches (no exact ktp/npwp == nomorid found anywhere),
  the result is further split into:
    - NAME_DOB_MATCH         : name matches, dob also matches exactly
    - NAME_DOB_SIMILAR       : name matches, dob's day & month match but
                                the year is different (mirip, beda tahun)
    - NAME_ONLY_MATCH_ID_EMPTY    : name matches, dob doesn't, and trx has
                                     no ktp/npwp at all (ktp kosong)
    - NAME_ONLY_MATCH_ID_DIFFERENT: name matches, dob doesn't, and trx's
                                     ktp/npwp has a value but it differs
                                     from the dpn candidate's nomorid

Output:
  - trx_all_enriched.csv: original trx_all columns + dpn_namadpn,
    dpn_alasan, dpn_alasantambahan, dpn_match_type, dpn_match_score
    (the dpn_* columns are blank/NaN when there's no match).
  - customer_dpn_match_result.csv: a compact match-only summary (optional,
    useful for QA).
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

TRX_PATH = "/home/cdsw/parquet_output/trx_all/trx_all/data.parquet"
DPN_PATH = "/home/cdsw/parquet_output/dpn/dpn/data.parquet"
ENRICHED_OUT_PATH = "/home/cdsw/parquet_output/trx_all_enriched.csv"
SUMMARY_OUT_PATH = "/home/cdsw/parquet_output/customer_dpn_match_result.csv"

TRX_COLS = [
    "key1", "flag", "disposition_class", "alert_month", "accountid", "customerid",
    "customername", "customerdate", "ktp", "npwp", "accountopendate",
    "customerbirthdate", "customerjob", "customerjobsector", "customerjoblevel",
    "customerjobnote", "employment_start_date", "monthly_income", "account_age",
    "flag_account_age_lt9", "flag_anomaly_income", "n_sameday_pass_3h_l3m",
    "n_sameday_pass_3h_l6m", "n_sameday_pass_1h_l3m", "n_sameday_pass_1h_l6m",
    "n_transfer_l3m", "n_transfer_l6m", "n_transfer_not_normal_l3m",
    "transfer_not_normal_l3m_pct", "n_transfer_not_normal_l6m",
    "transfer_not_normal_l6m_pct", "n_crypto", "n_in_10x_income",
    "flag_in_10x_income", "n_out_10x_income", "flag_out_10x_income",
    "n_new_beneficiary", "flag_new_beneficiary",
]

DPN_COLS = [
    "alasan", "alasantambahan", "branch", "dob", "iddpn", "inputdate", "jenisid",
    "jenisnasabah", "namadpn", "nomorid", "tglpengajuan", "tglupdate", "usernik",
    "username", "etl_date", "partition_date",
]

NAME_MATCH_THRESHOLD = 90  # rapidfuzz score (0-100) to accept as a name match


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).upper()
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x.strip()


def normalize_id(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\D", "", str(x))


def normalize_date(x):
    if pd.isna(x):
        return ""
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def name_similarity(a, b):
    if not a or not b:
        return 0
    return max(
        fuzz.ratio(a, b),
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b),
        fuzz.partial_ratio(a, b),
    )


def dob_match_status(a_norm, b_norm):
    """Compare two normalized 'YYYY-MM-DD' dob strings.
    EXACT -> full date matches
    SIMILAR_MONTH_DAY -> month & day match, year differs (mirip, beda tahun)
    DIFFERENT -> no match
    EMPTY -> one or both sides missing
    """
    if not a_norm or not b_norm:
        return "EMPTY"
    if a_norm == b_norm:
        return "EXACT"
    date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if date_pat.match(a_norm) and date_pat.match(b_norm):
        ay, am, ad = a_norm.split("-")
        by, bm, bd = b_norm.split("-")
        if am == bm and ad == bd and ay != by:
            return "SIMILAR_MONTH_DAY"
    return "DIFFERENT"


def id_status(ktp_norm, npwp_norm, nomorid_norm):
    """Compare trx ktp/npwp against a specific dpn candidate's nomorid.
    EMPTY -> trx has no ktp and no npwp at all
    MATCH -> ktp or npwp equals nomorid (shouldn't normally reach here, since
             exact id matches are already caught earlier, but kept for safety)
    DIFFERENT -> trx has a ktp/npwp value but it doesn't equal nomorid
    """
    if not ktp_norm and not npwp_norm:
        return "EMPTY"
    if (ktp_norm and ktp_norm == nomorid_norm) or (npwp_norm and npwp_norm == nomorid_norm):
        return "MATCH"
    return "DIFFERENT"


def load_data():
    trx = pd.read_parquet(TRX_PATH, columns=TRX_COLS)
    dpn = pd.read_parquet(DPN_PATH, columns=DPN_COLS)
    return trx, dpn


def prepare(trx, dpn):
    trx = trx.reset_index(drop=True).copy()
    dpn = dpn.reset_index(drop=True).copy()

    trx["_name_norm"] = trx["customername"].apply(normalize_text)
    trx["_dob_norm"] = trx["customerbirthdate"].apply(normalize_date)
    trx["_ktp_norm"] = trx["ktp"].apply(normalize_id)
    trx["_npwp_norm"] = trx["npwp"].apply(normalize_id)

    dpn["_name_norm"] = dpn["namadpn"].apply(normalize_text)
    dpn["_dob_norm"] = dpn["dob"].apply(normalize_date)
    dpn["_id_norm"] = dpn["nomorid"].apply(normalize_id)

    return trx, dpn


def match_customers(trx, dpn):
    n = len(trx)
    match_type = np.array(["NO_MATCH"] * n, dtype=object)
    match_score = np.zeros(n, dtype=float)
    matched_dpn_idx = np.full(n, -1, dtype=int)

    dpn_by_id = {}
    for idx, id_norm in dpn["_id_norm"].items():
        if id_norm:
            dpn_by_id.setdefault(id_norm, idx)

    needs_fuzzy = []
    for i, row in trx.iterrows():
        d_idx = dpn_by_id.get(row["_ktp_norm"]) or dpn_by_id.get(row["_npwp_norm"])
        if d_idx is not None:
            d_row = dpn.loc[d_idx]
            name_score = name_similarity(row["_name_norm"], d_row["_name_norm"])
            dob_match = row["_dob_norm"] != "" and row["_dob_norm"] == d_row["_dob_norm"]
            match_type[i] = (
                "PERFECT_MATCH"
                if (dob_match and name_score >= NAME_MATCH_THRESHOLD)
                else "ID_MATCH"
            )
            match_score[i] = name_score
            matched_dpn_idx[i] = d_idx
        else:
            needs_fuzzy.append(i)

    if needs_fuzzy:
        query_names = trx.loc[needs_fuzzy, "_name_norm"].tolist()
        choice_names = dpn["_name_norm"].tolist()

        score_matrix = process.cdist(
            query_names,
            choice_names,
            scorer=fuzz.token_sort_ratio,
            dtype=np.uint8,
            workers=-1,
        )

        best_pos = score_matrix.argmax(axis=1)
        best_scores = score_matrix[np.arange(len(needs_fuzzy)), best_pos]

        for k, i in enumerate(needs_fuzzy):
            if best_scores[k] >= NAME_MATCH_THRESHOLD:
                d_idx = dpn.index[best_pos[k]]
                d_row = dpn.loc[d_idx]
                final_score = name_similarity(trx.loc[i, "_name_norm"], d_row["_name_norm"])

                dob_stat = dob_match_status(trx.loc[i, "_dob_norm"], d_row["_dob_norm"])
                id_stat = id_status(
                    trx.loc[i, "_ktp_norm"], trx.loc[i, "_npwp_norm"], d_row["_id_norm"]
                )

                if dob_stat == "EXACT":
                    match_type[i] = "NAME_DOB_MATCH"
                elif dob_stat == "SIMILAR_MONTH_DAY":
                    # nama sama, tanggal & bulan lahir sama tapi tahun beda
                    match_type[i] = "NAME_DOB_SIMILAR"
                elif id_stat == "EMPTY":
                    # nama doang sama, ktp kosong
                    match_type[i] = "NAME_ONLY_MATCH_ID_EMPTY"
                else:
                    # nama sama tapi ktp ada isinya dan berbeda
                    match_type[i] = "NAME_ONLY_MATCH_ID_DIFFERENT"

                match_score[i] = final_score
                matched_dpn_idx[i] = d_idx

    return match_type, match_score, matched_dpn_idx


def build_enriched_trx(trx, dpn, match_type, match_score, matched_dpn_idx):
    enriched = trx.drop(columns=["_name_norm", "_dob_norm", "_ktp_norm", "_npwp_norm"]).copy()

    dpn_namadpn = np.full(len(trx), np.nan, dtype=object)
    dpn_alasan = np.full(len(trx), np.nan, dtype=object)
    dpn_alasantambahan = np.full(len(trx), np.nan, dtype=object)

    matched_mask = matched_dpn_idx >= 0
    matched_positions = np.where(matched_mask)[0]
    d_idxs = matched_dpn_idx[matched_positions]

    dpn_namadpn[matched_positions] = dpn.loc[d_idxs, "namadpn"].values
    dpn_alasan[matched_positions] = dpn.loc[d_idxs, "alasan"].values
    dpn_alasantambahan[matched_positions] = dpn.loc[d_idxs, "alasantambahan"].values

    enriched["dpn_namadpn"] = dpn_namadpn
    enriched["dpn_alasan"] = dpn_alasan
    enriched["dpn_alasantambahan"] = dpn_alasantambahan
    enriched["dpn_match_type"] = match_type
    enriched["dpn_match_score"] = match_score

    return enriched


def build_summary(trx, dpn, match_type, match_score, matched_dpn_idx):
    rows = []
    for i, row in trx.iterrows():
        d_idx = matched_dpn_idx[i]
        rec = {
            "customerid": row.get("customerid"),
            "accountid": row.get("accountid"),
            "customername": row.get("customername"),
            "customerbirthdate": row.get("customerbirthdate"),
            "ktp": row.get("ktp"),
            "npwp": row.get("npwp"),
            "match_type": match_type[i],
            "match_score": match_score[i],
        }
        if d_idx >= 0:
            d_row = dpn.loc[d_idx]
            rec.update(
                {
                    "matched_iddpn": d_row.get("iddpn"),
                    "matched_namadpn": d_row.get("namadpn"),
                    "matched_dob": d_row.get("dob"),
                    "matched_nomorid": d_row.get("nomorid"),
                    "matched_alasan": d_row.get("alasan"),
                    "matched_alasantambahan": d_row.get("alasantambahan"),
                    "matched_branch": d_row.get("branch"),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    trx, dpn = load_data()
    trx, dpn = prepare(trx, dpn)

    match_type, match_score, matched_dpn_idx = match_customers(trx, dpn)

    print(pd.Series(match_type).value_counts())

    enriched = build_enriched_trx(trx, dpn, match_type, match_score, matched_dpn_idx)
    enriched.to_csv(ENRICHED_OUT_PATH, index=False)
    print(f"Saved enriched trx_all to {ENRICHED_OUT_PATH}")

    summary = build_summary(trx, dpn, match_type, match_score, matched_dpn_idx)
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"Saved match summary to {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()

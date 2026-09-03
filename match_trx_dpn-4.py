"""
Match customers in trx_all (7,000 rows) against:
  1) the dpn table (72,000 rows)
  2) the screening table

...and enrich trx_all with the matched info from both tables.

Requirements:
    pip install pandas pyarrow rapidfuzz --break-system-packages

=====================================================================
DPN MATCHING (customername vs namadpn, customerbirthdate vs dob,
               ktp/npwp vs nomorid)
=====================================================================
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

=====================================================================
SCREENING MATCHING (customername vs alias OR name, customerbirthdate
                     vs dob, ktp/npwp vs idnumber)
=====================================================================
  Same overall logic as the dpn matching above (id short-circuit, then
  fuzzy fallback with the same DOB/ID sub-categories), except name is
  checked against BOTH the "alias" and "name" columns in screening and
  the better of the two scores/fields wins. The category values
  themselves are the SAME as the dpn ones (PERFECT_MATCH, ID_MATCH,
  NAME_DOB_MATCH, NAME_DOB_SIMILAR, NAME_ONLY_MATCH_ID_EMPTY,
  NAME_ONLY_MATCH_ID_DIFFERENT, NO_MATCH) - the two checks are told apart
  by column name only (dpn_match_type vs screening_match_type), not by a
  prefix on the category value.

Output:
  - trx_all_enriched.csv: original trx_all columns + dpn_* columns (from
    dpn matching) + screening_* columns (from screening matching,
    including screening_comment and every screening column checked, when
    a match is found). Both sets of columns are blank/NaN when there's no
    match for that table.
  - customer_dpn_match_result.csv: a compact match-only summary covering
    both dpn and screening matches, useful for QA.
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

TRX_PATH = "/home/cdsw/parquet_output/trx_all/trx_all/data.parquet"
DPN_PATH = "/home/cdsw/parquet_output/dpn/dpn/data.parquet"
SCREENING_PATH = "/home/cdsw/parquet_output/screening/screening/data.parquet"
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

SCREENING_COLS = [
    "alias", "comment", "dob", "idnumber", "idtype", "name", "nationality",
    "othersid", "othertype", "summaryid", "etl_date", "partition_date",
]

NAME_MATCH_THRESHOLD = 90  # rapidfuzz score (0-100) to accept as a name match
FUZZY_BATCH_SIZE = 200     # trx rows processed per cdist call - keeps memory bounded
                            # even against a 147k-row table (200 x 147000 x 1 byte
                            # per candidate list ~= 29MB per batch, instead of
                            # building one giant matrix that can OOM the process)


def batched_best_match(query_names, choice_lists, batch_size=FUZZY_BATCH_SIZE):
    """
    Find, for each query name, the best fuzzy match across one or more
    candidate lists (e.g. [alias_list, name_list] for screening, or just
    [namadpn_list] for dpn), processing in row-batches so memory never
    scales with the full query_count x choice_count matrix at once.

    Returns three arrays of length len(query_names):
      best_score    - best rapidfuzz token_sort_ratio score (0-100)
      best_pos      - the winning candidate's position within its list
      best_list_idx - which list in choice_lists produced the winner
    """
    q_total = len(query_names)
    best_score = np.zeros(q_total, dtype=np.uint8)
    best_pos = np.zeros(q_total, dtype=np.int64)
    best_list_idx = np.zeros(q_total, dtype=np.int64)

    for start in range(0, q_total, batch_size):
        end = min(start + batch_size, q_total)
        batch = query_names[start:end]

        batch_score = None
        batch_pos = None
        batch_list = None

        for li, choices in enumerate(choice_lists):
            mat = process.cdist(
                batch, choices, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1
            )
            pos = mat.argmax(axis=1)
            scores = mat[np.arange(len(batch)), pos]
            del mat  # free this batch's matrix before moving to the next list

            if batch_score is None:
                batch_score, batch_pos, batch_list = scores, pos, np.full(len(batch), li)
            else:
                better = scores > batch_score
                batch_score = np.where(better, scores, batch_score)
                batch_pos = np.where(better, pos, batch_pos)
                batch_list = np.where(better, li, batch_list)

        best_score[start:end] = batch_score
        best_pos[start:end] = batch_pos
        best_list_idx[start:end] = batch_list

    return best_score, best_pos, best_list_idx


# ---------------------------------------------------------------------------
# Normalization helpers - so case/spacing/punctuation never counts as a diff
# ---------------------------------------------------------------------------
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


def id_status(ktp_norm, npwp_norm, other_id_norm):
    """Compare trx ktp/npwp against a specific candidate's id (nomorid or
    idnumber).
    EMPTY -> trx has no ktp and no npwp at all, OR the candidate's id
             (nomorid / idnumber) itself is empty - either side missing
             means there's nothing real to compare, so it counts as
             "name only match, id kosong" rather than "id different"
    MATCH -> ktp or npwp equals the candidate id (shouldn't normally reach
             here, since exact id matches are already caught earlier, but
             kept for safety)
    DIFFERENT -> both sides have a value, but they don't match
    """
    if not ktp_norm and not npwp_norm:
        return "EMPTY"
    if not other_id_norm:
        return "EMPTY"
    if (ktp_norm and ktp_norm == other_id_norm) or (npwp_norm and npwp_norm == other_id_norm):
        return "MATCH"
    return "DIFFERENT"


# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
def load_data():
    trx = pd.read_parquet(TRX_PATH, columns=TRX_COLS)
    dpn = pd.read_parquet(DPN_PATH, columns=DPN_COLS)
    screening = pd.read_parquet(SCREENING_PATH, columns=SCREENING_COLS)
    return trx, dpn, screening


def prepare(trx, dpn, screening):
    trx = trx.reset_index(drop=True).copy()
    dpn = dpn.reset_index(drop=True).copy()
    screening = screening.reset_index(drop=True).copy()

    trx["_name_norm"] = trx["customername"].apply(normalize_text)
    trx["_dob_norm"] = trx["customerbirthdate"].apply(normalize_date)
    trx["_ktp_norm"] = trx["ktp"].apply(normalize_id)
    trx["_npwp_norm"] = trx["npwp"].apply(normalize_id)

    dpn["_name_norm"] = dpn["namadpn"].apply(normalize_text)
    dpn["_dob_norm"] = dpn["dob"].apply(normalize_date)
    dpn["_id_norm"] = dpn["nomorid"].apply(normalize_id)

    screening["_alias_norm"] = screening["alias"].apply(normalize_text)
    screening["_name_norm"] = screening["name"].apply(normalize_text)
    screening["_dob_norm"] = screening["dob"].apply(normalize_date)
    screening["_id_norm"] = screening["idnumber"].apply(normalize_id)

    return trx, dpn, screening


# ---------------------------------------------------------------------------
# DPN matching
# ---------------------------------------------------------------------------
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

        best_scores, best_pos, _ = batched_best_match(query_names, [choice_names])

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
                    match_type[i] = "NAME_DOB_SIMILAR"
                elif id_stat == "EMPTY":
                    match_type[i] = "NAME_ONLY_MATCH_ID_EMPTY"
                else:
                    match_type[i] = "NAME_ONLY_MATCH_ID_DIFFERENT"

                match_score[i] = final_score
                matched_dpn_idx[i] = d_idx

    return match_type, match_score, matched_dpn_idx


# ---------------------------------------------------------------------------
# Screening matching (name checked against BOTH alias and name)
# ---------------------------------------------------------------------------
def match_screening(trx, screening):
    n = len(trx)
    match_type = np.array(["NO_MATCH"] * n, dtype=object)
    match_score = np.zeros(n, dtype=float)
    matched_idx = np.full(n, -1, dtype=int)
    matched_field = np.array([""] * n, dtype=object)  # "ALIAS" or "NAME"

    screening_by_id = {}
    for idx, id_norm in screening["_id_norm"].items():
        if id_norm:
            screening_by_id.setdefault(id_norm, idx)

    needs_fuzzy = []
    for i, row in trx.iterrows():
        s_idx = screening_by_id.get(row["_ktp_norm"]) or screening_by_id.get(row["_npwp_norm"])
        if s_idx is not None:
            s_row = screening.loc[s_idx]
            alias_score = name_similarity(row["_name_norm"], s_row["_alias_norm"])
            name_score = name_similarity(row["_name_norm"], s_row["_name_norm"])
            best_score = max(alias_score, name_score)
            best_field = "ALIAS" if alias_score >= name_score else "NAME"

            dob_match = row["_dob_norm"] != "" and row["_dob_norm"] == s_row["_dob_norm"]
            match_type[i] = (
                "PERFECT_MATCH"
                if (dob_match and best_score >= NAME_MATCH_THRESHOLD)
                else "ID_MATCH"
            )
            match_score[i] = best_score
            matched_idx[i] = s_idx
            matched_field[i] = best_field
        else:
            needs_fuzzy.append(i)

    if needs_fuzzy:
        query_names = trx.loc[needs_fuzzy, "_name_norm"].tolist()
        alias_choices = screening["_alias_norm"].tolist()
        name_choices = screening["_name_norm"].tolist()

        # choice_lists[0] = alias, choice_lists[1] = name -> best_list_idx tells
        # us which field won for each row
        best_scores, best_pos, best_list_idx = batched_best_match(
            query_names, [alias_choices, name_choices]
        )

        for k, i in enumerate(needs_fuzzy):
            if best_scores[k] >= NAME_MATCH_THRESHOLD:
                pos = best_pos[k]
                s_idx = screening.index[pos]
                s_row = screening.loc[s_idx]

                alias_final = name_similarity(trx.loc[i, "_name_norm"], s_row["_alias_norm"])
                name_final = name_similarity(trx.loc[i, "_name_norm"], s_row["_name_norm"])
                if alias_final >= name_final:
                    final_score, field = alias_final, "ALIAS"
                else:
                    final_score, field = name_final, "NAME"

                dob_stat = dob_match_status(trx.loc[i, "_dob_norm"], s_row["_dob_norm"])
                id_stat = id_status(
                    trx.loc[i, "_ktp_norm"], trx.loc[i, "_npwp_norm"], s_row["_id_norm"]
                )

                if dob_stat == "EXACT":
                    match_type[i] = "NAME_DOB_MATCH"
                elif dob_stat == "SIMILAR_MONTH_DAY":
                    match_type[i] = "NAME_DOB_SIMILAR"
                elif id_stat == "EMPTY":
                    match_type[i] = "NAME_ONLY_MATCH_ID_EMPTY"
                else:
                    match_type[i] = "NAME_ONLY_MATCH_ID_DIFFERENT"

                match_score[i] = final_score
                matched_idx[i] = s_idx
                matched_field[i] = field

    return match_type, match_score, matched_idx, matched_field


# ---------------------------------------------------------------------------
# Build outputs
# ---------------------------------------------------------------------------
def build_enriched_trx(
    trx, dpn, dpn_match_type, dpn_match_score, matched_dpn_idx,
    screening, scr_match_type, scr_match_score, matched_scr_idx, matched_scr_field,
):
    enriched = trx.drop(columns=["_name_norm", "_dob_norm", "_ktp_norm", "_npwp_norm"]).copy()

    # --- dpn columns ---
    dpn_namadpn = np.full(len(trx), np.nan, dtype=object)
    dpn_alasan = np.full(len(trx), np.nan, dtype=object)
    dpn_alasantambahan = np.full(len(trx), np.nan, dtype=object)

    d_matched_pos = np.where(matched_dpn_idx >= 0)[0]
    d_idxs = matched_dpn_idx[d_matched_pos]
    dpn_namadpn[d_matched_pos] = dpn.loc[d_idxs, "namadpn"].values
    dpn_alasan[d_matched_pos] = dpn.loc[d_idxs, "alasan"].values
    dpn_alasantambahan[d_matched_pos] = dpn.loc[d_idxs, "alasantambahan"].values

    enriched["dpn_namadpn"] = dpn_namadpn
    enriched["dpn_alasan"] = dpn_alasan
    enriched["dpn_alasantambahan"] = dpn_alasantambahan
    enriched["dpn_match_type"] = dpn_match_type
    enriched["dpn_match_score"] = dpn_match_score

    # --- screening columns ---
    scr_alias = np.full(len(trx), np.nan, dtype=object)
    scr_name = np.full(len(trx), np.nan, dtype=object)
    scr_comment = np.full(len(trx), np.nan, dtype=object)
    scr_idnumber = np.full(len(trx), np.nan, dtype=object)
    scr_idtype = np.full(len(trx), np.nan, dtype=object)
    scr_dob = np.full(len(trx), np.nan, dtype=object)
    scr_nationality = np.full(len(trx), np.nan, dtype=object)
    scr_othersid = np.full(len(trx), np.nan, dtype=object)
    scr_othertype = np.full(len(trx), np.nan, dtype=object)
    scr_summaryid = np.full(len(trx), np.nan, dtype=object)

    s_matched_pos = np.where(matched_scr_idx >= 0)[0]
    s_idxs = matched_scr_idx[s_matched_pos]
    scr_alias[s_matched_pos] = screening.loc[s_idxs, "alias"].values
    scr_name[s_matched_pos] = screening.loc[s_idxs, "name"].values
    scr_comment[s_matched_pos] = screening.loc[s_idxs, "comment"].values
    scr_idnumber[s_matched_pos] = screening.loc[s_idxs, "idnumber"].values
    scr_idtype[s_matched_pos] = screening.loc[s_idxs, "idtype"].values
    scr_dob[s_matched_pos] = screening.loc[s_idxs, "dob"].values
    scr_nationality[s_matched_pos] = screening.loc[s_idxs, "nationality"].values
    scr_othersid[s_matched_pos] = screening.loc[s_idxs, "othersid"].values
    scr_othertype[s_matched_pos] = screening.loc[s_idxs, "othertype"].values
    scr_summaryid[s_matched_pos] = screening.loc[s_idxs, "summaryid"].values

    enriched["screening_matched_field"] = matched_scr_field
    enriched["screening_alias"] = scr_alias
    enriched["screening_name"] = scr_name
    enriched["screening_comment"] = scr_comment
    enriched["screening_idnumber"] = scr_idnumber
    enriched["screening_idtype"] = scr_idtype
    enriched["screening_dob"] = scr_dob
    enriched["screening_nationality"] = scr_nationality
    enriched["screening_othersid"] = scr_othersid
    enriched["screening_othertype"] = scr_othertype
    enriched["screening_summaryid"] = scr_summaryid
    enriched["screening_match_type"] = scr_match_type
    enriched["screening_match_score"] = scr_match_score

    return enriched


def build_summary(
    trx, dpn, dpn_match_type, dpn_match_score, matched_dpn_idx,
    screening, scr_match_type, scr_match_score, matched_scr_idx, matched_scr_field,
):
    rows = []
    for i, row in trx.iterrows():
        rec = {
            "customerid": row.get("customerid"),
            "accountid": row.get("accountid"),
            "customername": row.get("customername"),
            "customerbirthdate": row.get("customerbirthdate"),
            "ktp": row.get("ktp"),
            "npwp": row.get("npwp"),
            "dpn_match_type": dpn_match_type[i],
            "dpn_match_score": dpn_match_score[i],
            "screening_match_type": scr_match_type[i],
            "screening_match_score": scr_match_score[i],
        }

        d_idx = matched_dpn_idx[i]
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

        s_idx = matched_scr_idx[i]
        if s_idx >= 0:
            s_row = screening.loc[s_idx]
            rec.update(
                {
                    "screening_matched_field": matched_scr_field[i],
                    "matched_alias": s_row.get("alias"),
                    "matched_screening_name": s_row.get("name"),
                    "matched_screening_dob": s_row.get("dob"),
                    "matched_idnumber": s_row.get("idnumber"),
                    "matched_idtype": s_row.get("idtype"),
                    "matched_comment": s_row.get("comment"),
                    "matched_nationality": s_row.get("nationality"),
                    "matched_othersid": s_row.get("othersid"),
                    "matched_othertype": s_row.get("othertype"),
                    "matched_summaryid": s_row.get("summaryid"),
                }
            )

        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    trx, dpn, screening = load_data()
    trx, dpn, screening = prepare(trx, dpn, screening)

    dpn_match_type, dpn_match_score, matched_dpn_idx = match_customers(trx, dpn)
    print("DPN matches:")
    print(pd.Series(dpn_match_type).value_counts())

    scr_match_type, scr_match_score, matched_scr_idx, matched_scr_field = match_screening(
        trx, screening
    )
    print("\nScreening matches:")
    print(pd.Series(scr_match_type).value_counts())

    enriched = build_enriched_trx(
        trx, dpn, dpn_match_type, dpn_match_score, matched_dpn_idx,
        screening, scr_match_type, scr_match_score, matched_scr_idx, matched_scr_field,
    )
    enriched.to_csv(ENRICHED_OUT_PATH, index=False)
    print(f"\nSaved enriched trx_all to {ENRICHED_OUT_PATH}")

    summary = build_summary(
        trx, dpn, dpn_match_type, dpn_match_score, matched_dpn_idx,
        screening, scr_match_type, scr_match_score, matched_scr_idx, matched_scr_field,
    )
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"Saved match summary to {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()

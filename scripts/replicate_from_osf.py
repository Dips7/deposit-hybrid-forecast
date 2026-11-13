"""
Self-Contained Replication and Verification Script for OSF Peer Review Archive.

Target Journal: Applied Stochastic Models in Business and Industry (Wiley)
Manuscript Title: Information-Enriched Forecast Combinations for Nepalese Bank Deposits:
                  Retrospective Evidence and Reliability Limits
Authors: Dipesh Gurung, Deepika Sharma, Rashmi Dahiya (Noida International University)

REPLICATION SCOPE AND OPERATIONAL BOUNDARY:
This script performs rigorous, independent mathematical recalculation and statistical verification
directly from the saved evaluation and forecast records (1,337 scored evaluations across 191 rolling origins).
It verifies cryptographic file integrity, confirms the scheduled evaluation cohort, checks
error-field mathematical identities and finiteness, recalculates all sample performance metrics
for all 7 candidate procedures, verifies tail squared-loss concentration, and recalculates
moving-block bootstrap hypothesis tests (B=10,000, seed=20261029) and step-down Holm adjusted
p-values across block lengths L in {6, 12, 18}.

This script validates saved evaluation records; it does NOT perform raw model training or refitting
from preliminary Nepal Rastra Bank balance sheets.

Dependencies: Python >= 3.9, numpy, pandas (no plotting library required).
Execution:
    python3 scripts/replicate_from_osf.py
    python3 -O scripts/replicate_from_osf.py  (works under optimization; uses explicit exceptions)
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ARCHIVE_ROOT / "data"

EXPECTED_PROCEDURES = ["E_EQ", "E_S", "E_X", "H_OMO", "H_Y", "XGB_OMO", "XGB_Y"]
PRIMARY_COMPARISONS = [
    {"comparison_id": "C1", "name": "H_OMO vs E_EQ", "challenger": "H_OMO", "reference": "E_EQ"},
    {"comparison_id": "C2", "name": "H_OMO vs H_Y", "challenger": "H_OMO", "reference": "H_Y"},
    {"comparison_id": "C3", "name": "H_OMO vs E_X", "challenger": "H_OMO", "reference": "E_X"},
    {"comparison_id": "C4", "name": "H_OMO vs E_S", "challenger": "H_OMO", "reference": "E_S"},
]


def verify_hash_manifest():
    """Verify cryptographic SHA-256 hashes for all files listed in HASH_MANIFEST.json."""
    manifest_path = ARCHIVE_ROOT / "HASH_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing HASH_MANIFEST.json at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"\n[PHASE 0] Verifying cryptographic integrity of {len(manifest)} archive files...")
    this_script_rel = "scripts/replicate_from_osf.py"

    for rel_path, expected_hash in manifest.items():
        file_path = ARCHIVE_ROOT / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Archive file missing: {rel_path} (expected at {file_path})")

        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Cryptographic hash mismatch for {rel_path}:\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )

    print(f"         [OK] All {len(manifest)} archive files verified against SHA-256 manifest.")


def moving_block_bootstrap_paired(
    d_series: np.ndarray,
    block_length: int = 12,
    n_boot: int = 10000,
    seed: int = 20261029
):
    """
    Centered moving block bootstrap for paired loss differential d = L_ref - L_chal.
    Higher d indicates challenger superiority (lower loss).
    Tests two-sided null H0: E[d] == 0 vs H1: E[d] != 0.
    """
    d_arr = np.asarray(d_series, dtype=float)
    n = len(d_arr)
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_length))
    possible_starts = n - block_length + 1

    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.randint(0, possible_starts, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        boot_means[b] = np.mean(d_arr[indices])

    obs_mean = float(np.mean(d_arr))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Center bootstrap distribution at 0 under null hypothesis
    boot_centered = boot_means - obs_mean
    p_val = float(np.mean(np.abs(boot_centered) >= np.abs(obs_mean)))
    p_val = max(p_val, 1.0 / n_boot)

    return obs_mean, (ci_lower, ci_upper), p_val


def run_bootstrap_family(pivot_loss: pd.DataFrame, block_length: int, n_boot: int = 10000, seed: int = 20261029):
    comparisons_out = []
    raw_p_values = []

    for comp in PRIMARY_COMPARISONS:
        ref = comp["reference"]
        chal = comp["challenger"]
        l_ref = pivot_loss[ref].values
        l_chal = pivot_loss[chal].values
        d = l_ref - l_chal

        mean_ref = float(np.mean(l_ref))
        mean_chal = float(np.mean(l_chal))
        effect_mean, (ci_low, ci_high), p_val = moving_block_bootstrap_paired(
            d, block_length=block_length, n_boot=n_boot, seed=seed
        )
        pct_reduction = 100.0 * (effect_mean / mean_ref) if mean_ref > 0 else 0.0

        raw_p_values.append(p_val)
        comparisons_out.append({
            "comparison_id": comp["comparison_id"],
            "name": comp["name"],
            "effect_mean": effect_mean,
            "pct_reduction": pct_reduction,
            "ci_lower_95": ci_low,
            "ci_upper_95": ci_high,
            "raw_p_value": p_val
        })

    # Step-down Holm FWER adjustment
    m = len(comparisons_out)
    order = np.argsort(raw_p_values)
    holm_p = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        multiplier = m - rank
        holm_p[idx] = min(1.0, raw_p_values[idx] * multiplier)

    # Monotonicity enforcement
    for i in range(1, m):
        curr_idx = order[i]
        prev_idx = order[i - 1]
        if holm_p[curr_idx] < holm_p[prev_idx]:
            holm_p[curr_idx] = holm_p[prev_idx]

    for i, c in enumerate(comparisons_out):
        c["holm_adjusted_p"] = float(holm_p[i])

    return comparisons_out


def replicate_all():
    print("=" * 82)
    print("INDEPENDENT MATHEMATICAL REPLICATION AND AUDIT FROM SAVED EVALUATION RECORDS")
    print("=" * 82)
    print("NOTE: This script performs evaluation reproduction and statistical verification")
    print("      from saved forecast records; it does not perform raw model training.")

    # Phase 0: Hash Manifest Verification
    verify_hash_manifest()

    # Phase 1: Load and Validate Evaluation Cohort
    eval_file = DATA_DIR / "evaluation_DEVELOPMENT_FIT_V1.csv"
    if not eval_file.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_file}")

    df_eval = pd.read_csv(eval_file)
    print(f"\n[PHASE 1] Validating evaluation cohort structure and scheduled origins...")
    if len(df_eval) != 1337:
        raise ValueError(f"Expected exactly 1,337 total evaluations, found {len(df_eval)}")

    eligible = df_eval[df_eval["is_eligible_for_evaluation"] == True].copy()
    if len(eligible) != 1337:
        raise ValueError(f"Expected exactly 1,337 eligible evaluations, found {len(eligible)}")

    procedures = sorted(eligible["procedure_id"].unique())
    if procedures != EXPECTED_PROCEDURES:
        raise ValueError(f"Procedure set mismatch: expected {EXPECTED_PROCEDURES}, found {procedures}")

    origins = sorted(eligible["origin_period"].unique())
    if len(origins) != 191:
        raise ValueError(f"Expected exactly 191 distinct origins, found {len(origins)}")
    if origins[0] != "2010-01" or origins[-1] != "2025-11":
        raise ValueError(f"Origin range mismatch: expected 2010-01 to 2025-11, found {origins[0]} to {origins[-1]}")

    targets = sorted(eligible["target_period"].unique())
    if targets[0] != "2010-02" or targets[-1] != "2025-12":
        raise ValueError(f"Target range mismatch: expected 2010-02 to 2025-12, found {targets[0]} to {targets[-1]}")

    cutoffs = sorted(eligible["cutoff_period"].unique())
    if cutoffs[0] != "2009-11" or cutoffs[-1] != "2025-09":
        raise ValueError(f"Cutoff range mismatch: expected 2009-11 to 2025-09, found {cutoffs[0]} to {cutoffs[-1]}")

    duplicates = eligible.duplicated(subset=["origin_period", "procedure_id"]).sum()
    if duplicates != 0:
        raise ValueError(f"Found {duplicates} duplicate (origin_period, procedure_id) pairs in evaluation ledger!")

    print(f"         [OK] Cohort verified: 191 contiguous origins (2010-01 to 2025-11), 7 procedures, 0 duplicates.")

    # Phase 2: Error-Field Mathematical Identities and Finiteness
    print("\n[PHASE 2] Auditing error-field mathematical identities and numerical finiteness...")
    numeric_fields = ["y_actual", "y_hat", "actual_error", "abs_error", "sq_error"]
    for field in numeric_fields:
        if not np.all(np.isfinite(eligible[field].values)):
            raise ValueError(f"Non-finite values detected in field: {field}")

    # Check actual_error == y_actual - y_hat
    err_diff = np.abs(eligible["actual_error"].values - (eligible["y_actual"].values - eligible["y_hat"].values))
    max_err_diff = float(np.max(err_diff))
    if max_err_diff > 1e-6:
        raise ValueError(f"Mathematical identity violation: max |actual_error - (y_actual - y_hat)| = {max_err_diff} > 1e-6")

    # Check abs_error == abs(actual_error)
    abs_diff = np.abs(eligible["abs_error"].values - np.abs(eligible["actual_error"].values))
    max_abs_diff = float(np.max(abs_diff))
    if max_abs_diff > 1e-6:
        raise ValueError(f"Mathematical identity violation: max |abs_error - |actual_error|| = {max_abs_diff} > 1e-6")

    # Check sq_error == actual_error ** 2
    sq_diff = np.abs(eligible["sq_error"].values - (eligible["actual_error"].values ** 2))
    max_sq_diff = float(np.max(sq_diff))
    rel_sq_diff = float(np.max(sq_diff / (eligible["sq_error"].values + 1e-12)))
    if rel_sq_diff > 1e-6 and max_sq_diff > 1e-4:
        raise ValueError(f"Mathematical identity violation: max sq_error mismatch = {max_sq_diff}, rel = {rel_sq_diff}")

    print(f"         [OK] Identities verified: actual_error == y_actual - y_hat, abs_error == |err|, sq_error == err^2.")

    # Phase 3: Recalculate and Validate Aggregate Metrics
    print("\n[PHASE 3] Independent recalculation and validation of all aggregate metrics...")
    metrics_csv = DATA_DIR / "metrics_DEVELOPMENT_FIT_V1.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing saved metrics file: {metrics_csv}")
    df_saved = pd.read_csv(metrics_csv).set_index("procedure_id")

    recalc_metrics = []
    print(f"{'Procedure':<10} | {'Recalc MAE':<12} | {'Saved MAE':<12} | {'Recalc RMSE':<12} | {'Saved RMSE':<12} | {'Status'}")
    print("-" * 82)
    for proc in procedures:
        sub = eligible[eligible["procedure_id"] == proc]
        err = sub["actual_error"].values
        abs_err = sub["abs_error"].values
        fb_rate = float(sub["is_fallback"].mean())
        n_scored = len(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        med_ae = float(np.median(abs_err))
        bias = float(np.mean(err))

        s_row = df_saved.loc[proc]

        # Validate all 5 reported metrics plus n_scored
        if n_scored != int(s_row["n_scored"]):
            raise ValueError(f"n_scored mismatch for {proc}: recalc {n_scored} vs saved {s_row['n_scored']}")
        if abs(mae - s_row["mae"]) > 1e-5:
            raise ValueError(f"MAE mismatch for {proc}: recalc {mae} vs saved {s_row['mae']}")
        if abs(rmse - s_row["rmse"]) > 1e-5:
            raise ValueError(f"RMSE mismatch for {proc}: recalc {rmse} vs saved {s_row['rmse']}")
        if abs(med_ae - s_row["median_ae"]) > 1e-5:
            raise ValueError(f"MedAE mismatch for {proc}: recalc {med_ae} vs saved {s_row['median_ae']}")
        if abs(bias - s_row["bias"]) > 1e-5:
            raise ValueError(f"Bias mismatch for {proc}: recalc {bias} vs saved {s_row['bias']}")
        if abs(fb_rate - s_row["fallback_rate"]) > 1e-5:
            raise ValueError(f"Fallback rate mismatch for {proc}: recalc {fb_rate} vs saved {s_row['fallback_rate']}")

        recalc_metrics.append({
            "procedure_id": proc,
            "n_scored": n_scored,
            "mae": mae,
            "rmse": rmse,
            "median_ae": med_ae,
            "bias": bias,
            "fallback_rate": fb_rate
        })
        print(f"{proc:<10} | {mae:<12.2f} | {s_row['mae']:<12.2f} | {rmse:<12.2f} | {s_row['rmse']:<12.2f} | EXACT MATCH")

    print(f"         [OK] All 7 procedures match across MAE, RMSE, MedAE, Bias, and Fallback Rate.")

    # Phase 4: Recalculate and Validate Tail Loss Concentration
    print("\n[PHASE 4] Independent recalculation and validation of tail squared loss concentration...")
    tail_csv = DATA_DIR / "TAIL_METRICS.csv"
    if not tail_csv.exists():
        raise FileNotFoundError(f"Missing tail metrics file: {tail_csv}")
    df_tail_saved = pd.read_csv(tail_csv).set_index("procedure")

    print(f"{'Procedure':<10} | {'Recalc Top 9 (%)':<18} | {'Saved Top 9 (%)':<18} | {'Recalc Top 10 (%)':<18} | {'Status'}")
    print("-" * 82)
    for proc in procedures:
        sq = eligible[eligible["procedure_id"] == proc]["sq_error"].sort_values(ascending=False).values
        top9_share = float(np.sum(sq[:9]) / np.sum(sq) * 100.0)
        top10_share = float(np.sum(sq[:10]) / np.sum(sq) * 100.0)
        saved_top9 = float(df_tail_saved.loc[proc, "top_9_squared_loss_share"] * 100.0)
        saved_top10 = float(df_tail_saved.loc[proc, "top_10_squared_loss_share"] * 100.0)

        if abs(top9_share - saved_top9) > 1e-4:
            raise ValueError(f"Top 9 mismatch for {proc}: recalc {top9_share} vs saved {saved_top9}")
        if abs(top10_share - saved_top10) > 1e-4:
            raise ValueError(f"Top 10 mismatch for {proc}: recalc {top10_share} vs saved {saved_top10}")

        print(f"{proc:<10} | {top9_share:<18.2f} | {saved_top9:<18.2f} | {top10_share:<18.2f} | EXACT MATCH")

    print(f"         [OK] All 7 procedures match across top-9 and top-10 squared loss concentration.")

    # Phase 5: Primary Moving-Block Bootstrap Inference (L=12)
    print("\n[PHASE 5] Recalculating primary moving-block bootstrap inference (L=12, B=10,000, seed=20261029)...")
    pivot_loss = eligible.pivot(index="origin_period", columns="procedure_id", values="abs_error")
    recalc_l12 = run_bootstrap_family(pivot_loss, block_length=12, n_boot=10000, seed=20261029)

    inf_json = DATA_DIR / "inference_summary_DEVELOPMENT_FIT_V1.json"
    if not inf_json.exists():
        raise FileNotFoundError(f"Missing inference summary file: {inf_json}")
    with open(inf_json, "r", encoding="utf-8") as f:
        saved_inf = json.load(f)

    saved_l12 = saved_inf["primary_family_L12"]["comparisons"]

    print(f"{'Contrast':<18} | {'Delta MAE':<10} | {'Pct Red (%)':<11} | {'Raw p':<8} | {'Holm p':<8} | {'Status'}")
    print("-" * 82)
    for c_recalc in recalc_l12:
        c_name = c_recalc["name"]
        c_saved = next((s for s in saved_l12 if s["name"] == c_name), None)
        if c_saved is None:
            raise ValueError(f"Saved contrast not found for {c_name}")

        if not np.isclose(c_recalc["effect_mean"], c_saved["effect_mean"], atol=1e-5):
            raise ValueError(f"Effect mean mismatch for {c_name}: recalc {c_recalc['effect_mean']} vs saved {c_saved['effect_mean']}")
        if not np.isclose(c_recalc["pct_reduction"], c_saved["pct_reduction"], atol=1e-5):
            raise ValueError(f"Pct reduction mismatch for {c_name}: recalc {c_recalc['pct_reduction']} vs saved {c_saved['pct_reduction']}")
        if not np.isclose(c_recalc["ci_lower_95"], c_saved["ci_lower_95"], atol=1e-3):
            raise ValueError(f"CI lower mismatch for {c_name}: recalc {c_recalc['ci_lower_95']} vs saved {c_saved['ci_lower_95']}")
        if not np.isclose(c_recalc["ci_upper_95"], c_saved["ci_upper_95"], atol=1e-3):
            raise ValueError(f"CI upper mismatch for {c_name}: recalc {c_recalc['ci_upper_95']} vs saved {c_saved['ci_upper_95']}")
        if not np.isclose(c_recalc["raw_p_value"], c_saved["raw_p_value"], atol=1e-4):
            raise ValueError(f"Raw p mismatch for {c_name}: recalc {c_recalc['raw_p_value']} vs saved {c_saved['raw_p_value']}")
        if not np.isclose(c_recalc["holm_adjusted_p"], c_saved["holm_adjusted_p"], atol=1e-4):
            raise ValueError(f"Holm p mismatch for {c_name}: recalc {c_recalc['holm_adjusted_p']} vs saved {c_saved['holm_adjusted_p']}")

        print(f"{c_name:<18} | {c_recalc['effect_mean']:<10.2f} | {c_recalc['pct_reduction']:<11.2f} | {c_recalc['raw_p_value']:<8.4f} | {c_recalc['holm_adjusted_p']:<8.4f} | EXACT MATCH")

    # Confirm all Holm p-values are non-significant
    min_holm_l12 = min(c["holm_adjusted_p"] for c in recalc_l12)
    if min_holm_l12 < 0.1764:
        raise ValueError(f"Expected minimum L=12 Holm p >= 0.1764, found {min_holm_l12}")
    print(f"         [OK] Confirmed: Zero contrasts statistically significant after Holm adjustment (minimum p_Holm = {min_holm_l12:.4f}).")

    # Phase 6: Sensitivity Verification Across Alternative Block Lengths (L=6, L=18)
    print("\n[PHASE 6] Recalculating and validating sensitivity across block lengths L in {6, 18}...")
    for L, key in [(6, "sensitivity_L6"), (18, "sensitivity_L18")]:
        recalc_sens = run_bootstrap_family(pivot_loss, block_length=L, n_boot=10000, seed=20261029)
        saved_sens = saved_inf[key]["comparisons"]

        for c_recalc in recalc_sens:
            c_name = c_recalc["name"]
            c_saved = next((s for s in saved_sens if s["name"] == c_name), None)
            if c_saved is None:
                raise ValueError(f"Saved sensitivity contrast not found for L={L}, {c_name}")

            if not np.isclose(c_recalc["effect_mean"], c_saved["effect_mean"], atol=1e-5):
                raise ValueError(f"L={L} effect mismatch for {c_name}")
            if not np.isclose(c_recalc["pct_reduction"], c_saved["pct_reduction"], atol=1e-5):
                raise ValueError(f"L={L} pct reduction mismatch for {c_name}")
            if not np.isclose(c_recalc["ci_lower_95"], c_saved["ci_lower_95"], atol=1e-3):
                raise ValueError(f"L={L} CI lower mismatch for {c_name}")
            if not np.isclose(c_recalc["ci_upper_95"], c_saved["ci_upper_95"], atol=1e-3):
                raise ValueError(f"L={L} CI upper mismatch for {c_name}")
            if not np.isclose(c_recalc["raw_p_value"], c_saved["raw_p_value"], atol=1e-4):
                raise ValueError(f"L={L} raw p mismatch for {c_name}")
            if not np.isclose(c_recalc["holm_adjusted_p"], c_saved["holm_adjusted_p"], atol=1e-4):
                raise ValueError(f"L={L} Holm p mismatch for {c_name}")

        min_sens_holm = min(c["holm_adjusted_p"] for c in recalc_sens)
        expected_min = 0.0720 if L == 6 else 0.1996
        if not np.isclose(min_sens_holm, expected_min, atol=1e-4):
            raise ValueError(f"L={L} min Holm p mismatch: recalc {min_sens_holm} vs expected {expected_min}")
        print(f"         Block length L={L:<2}: all 4 contrasts verified; minimum p_Holm = {min_sens_holm:.4f} >= {expected_min:.4f} (Non-significant)")

    print("\n" + "=" * 82)
    print("ALL REPLICATION CHECKS PASSED: 100% MATHEMATICAL & INFERENTIAL EQUIVALENCE")
    print("=" * 82)


if __name__ == "__main__":
    replicate_all()

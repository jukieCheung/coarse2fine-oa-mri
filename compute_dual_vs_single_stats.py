#!/usr/bin/env python3
"""Canonical Dual-vs-single statistical comparison for the MICAD submission.

Reproducible statistics over the nine held-out-test prediction CSVs
(``*_eval_results/*_preds.csv``). This script is the single source of truth
for every Delta / CI / p-value reported in the manuscript.

Protocol (recorded here and in the output metadata):
  * OA metrics (AUC, Acc, F1): Dual vs Single-OA.
  * KL metrics (macro-AUC OvR, Acc, macro-F1): Dual vs Single-KL.
  * OA accuracy: McNemar's test on paired predictions
    (exact binomial if discordant pairs b+c < 25, else chi-square with
    Edwards continuity correction; statsmodels ``mcnemar``).
  * All other metrics: paired subject-level bootstrap of the performance
    difference (Dual - Single), B=10000 resamples, seed=42
    (numpy.default_rng), percentile 95% CI, two-sided p-value
    p = 2*min((#{d*<=0}+1)/(B+1), (#{d*>=0}+1)/(B+1)).
  * Benjamini-Hochberg FDR over the full family of 18 primary tests
    (3 backbones x 6 metrics), written to stats_recomputed_with_fdr.csv.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

B = 10000
SEED = 42
BASE = __import__("os").environ.get("OACTF_BASE", ".")  # repo root (override with OACTF_BASE)

BACKBONES = ["resnet3d", "m3t", "mamba"]
PREDS = {
    ("resnet3d", "single_oa"): f"{BASE}/results/resnet3d_single_oa_preds.csv",
    ("resnet3d", "single_kl"): f"{BASE}/results/resnet3d_single_kl_preds.csv",
    ("resnet3d", "dual"):      f"{BASE}/results/resnet3d_dual_preds.csv",
    ("m3t", "single_oa"):      f"{BASE}/results/m3t_single_oa_preds.csv",
    ("m3t", "single_kl"):      f"{BASE}/results/m3t_single_kl_preds.csv",
    ("m3t", "dual"):           f"{BASE}/results/m3t_dual_preds.csv",
    ("mamba", "single_oa"):    f"{BASE}/results/mamba_single_oa_preds.csv",
    ("mamba", "single_kl"):    f"{BASE}/results/mamba_single_kl_preds.csv",
    ("mamba", "dual"):         f"{BASE}/results/mamba_dual_preds.csv",
}
KL_COLS = [f"prob_kl_{k}" for k in range(5)]


def load():
    dfs = {}
    for key, path in PREDS.items():
        df = pd.read_csv(path).sort_values("CMT_ID").reset_index(drop=True)
        dfs[key] = df
    return dfs


def check_pairing(dfs):
    ids = dfs[("resnet3d", "dual")]["CMT_ID"].values
    for key, df in dfs.items():
        assert np.array_equal(df["CMT_ID"].values, ids), f"ID mismatch in {key}"
        assert np.array_equal(df["label_kl"].values, dfs[("resnet3d", "dual")]["label_kl"].values)
        assert np.array_equal(df["label_oa"].values, dfs[("resnet3d", "dual")]["label_oa"].values)
    return ids


def oa_metric(df, metric, idx=None):
    y = df["label_oa"].values if idx is None else df["label_oa"].values[idx]
    p = df["prob_oa"].values if idx is None else df["prob_oa"].values[idx]
    yhat = df["pred_oa"].values if idx is None else df["pred_oa"].values[idx]
    if metric == "AUC":
        return roc_auc_score(y, p)
    if metric == "Acc":
        return accuracy_score(y, yhat)
    if metric == "F1":
        return f1_score(y, yhat)
    raise ValueError(metric)


def kl_metric(df, metric, idx=None):
    y = df["label_kl"].values if idx is None else df["label_kl"].values[idx]
    yhat = df["pred_kl"].values if idx is None else df["pred_kl"].values[idx]
    if metric == "M-AUC":
        P = df[KL_COLS].values if idx is None else df[KL_COLS].values[idx]
        return roc_auc_score(y, P, multi_class="ovr", average="macro")
    if metric == "Acc":
        return accuracy_score(y, yhat)
    if metric == "M-F1":
        return f1_score(y, yhat, average="macro")
    raise ValueError(metric)


def bootstrap_delta(df_a, df_b, fn, metric, rng, n):
    """Paired bootstrap of fn(df_a) - fn(df_b); df_a=Dual, df_b=Single."""
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        deltas[b] = fn(df_a, metric, idx) - fn(df_b, metric, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((np.sum(deltas <= 0) + 1) / (B + 1), (np.sum(deltas >= 0) + 1) / (B + 1))
    return float(lo), float(hi), float(min(p, 1.0))


def mcnemar_acc(df_a, df_b):
    """McNemar on accuracy (correctness) of df_a (Dual) vs df_b (Single), OA task."""
    ya = (df_a["pred_oa"].values == df_a["label_oa"].values)
    yb = (df_b["pred_oa"].values == df_b["label_oa"].values)
    # table: [[both correct, a correct b wrong], [a wrong b correct, both wrong]]
    table = [[int(np.sum(ya & yb)), int(np.sum(ya & ~yb))],
             [int(np.sum(~ya & yb)), int(np.sum(~ya & ~yb))]]
    b_disc, c_disc = table[0][1], table[1][0]
    exact = (b_disc + c_disc) < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    return float(res.pvalue), ("exact" if exact else "chi2+continuity"), table


def main():
    dfs = load()
    ids = check_pairing(dfs)
    n = len(ids)
    rng = np.random.default_rng(SEED)
    rows = []
    for bb in BACKBONES:
        for task, metrics, single_key, fn in [
            ("OA", ["AUC", "Acc", "F1"], (bb, "single_oa"), oa_metric),
            ("KL", ["M-AUC", "Acc", "M-F1"], (bb, "single_kl"), kl_metric),
        ]:
            df_d = dfs[(bb, "dual")]
            df_s = dfs[single_key]
            for metric in metrics:
                v_d = fn(df_d, metric)
                v_s = fn(df_s, metric)
                delta = v_d - v_s
                lo, hi, p_boot = bootstrap_delta(df_d, df_s, fn, metric, rng, n)
                test_used = "paired_bootstrap"
                p = p_boot
                mcnemar_note = ""
                if task == "OA" and metric == "Acc":
                    p_mc, flavor, table = mcnemar_acc(df_d, df_s)
                    p = p_mc
                    test_used = f"mcnemar_{flavor}"
                    mcnemar_note = json.dumps(table)
                rows.append(dict(
                    backbone=bb, task=task, metric=metric,
                    single=round(v_s, 6), dual=round(v_d, 6),
                    delta=round(delta, 6), ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                    p_value=p, test=test_used, mcnemar_table=mcnemar_note,
                    n_subjects=n,
                ))
    out = pd.DataFrame(rows)
    out.to_csv(f"{BASE}/stats_recomputed.csv", index=False)
    print(out.to_string(index=False))

    # Benjamini-Hochberg over the full family of 18 tests
    m = len(out)
    order = np.argsort(out["p_value"].values, kind="stable")
    q = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, out["p_value"].values[i] * m / rank)
        q[i] = running
    out["q_value_BH"] = np.round(q, 6)
    out["sig_fdr_0.05"] = out["q_value_BH"] < 0.05
    out.to_csv(f"{BASE}/stats_recomputed_with_fdr.csv", index=False)

    meta = dict(
        bootstrap=dict(B=B, seed=SEED, resampling="paired subject-level",
                       ci="percentile 95%",
                       p_value="two-sided: 2*min((#{d<=0}+1)/(B+1), (#{d>=0}+1)/(B+1))"),
        mcnemar="exact binomial if discordant pairs < 25, else chi-square with Edwards continuity correction",
        fdr="Benjamini-Hochberg over the full family of 18 primary Dual-vs-single tests",
        oa_eval="Single-KL OA metrics use prob_oa = sum of p_KL(k>=2); Dual OA metrics use the dedicated OA head",
    )
    with open(f"{BASE}/stats_recomputed_protocol.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nWrote stats_recomputed.csv, stats_recomputed_with_fdr.csv, stats_recomputed_protocol.json")


if __name__ == "__main__":
    main()

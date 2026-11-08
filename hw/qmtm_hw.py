"""Projected AP(TM) cost of the QMTM additions (literal cap L, relative QM via bias columns, R1). MODEL-BASED PROJECTIONS.
Base model (AP(TM), ESWA 2026, Eq. (11) and Table 3): per class update
    cycles = 127 + 236 log2(s) + (8N+16) M,  M = 2 x features, N = TA state bits (8).
    Phase 3 per literal: enables 4 cmp + 6 wr; saturating increment and decrement (Alg. 1) 2N+1 cmp + 2N+2 wr each.
    Inference: 5K + 1 cycles (+ ceil(log2 m) reduction tree); clause evaluation = one compare-tm (all rows).
Additions (our design, Section 5 of the paper):
  * n_inc counter, c = ceil(log2(L+2)) bits per row (holds L+1), recounted after Phase 3 of every class update (needed by the literal
    cap of Abeyrathna et al. and by relative QM): 1 clearing write + per literal: 1 cmp (tag rows whose TA MSB is 1),
    1 wr (inc_en), saturating c-bit increment (2c+1 cmp + 2c+2 wr) -> 4c+5 cycles per literal.
  * Cap check (pyTsetlinMachine semantics: a firing clause receives Type Ia only if n_inc <= L, otherwise Type Ib): once per class
    update, n_inc <= L for L = 2^k is (bits >= k all 0) OR (n_inc == L): 2 compares + 1 write = 3 cycles, stored in a row flag;
    per literal, the extra Type Ib enable branch (fired AND over the cap) adds 1 compare + 1 write = 2 cycles.
  * Relative QM, clamped: tol_j = min(floor((1-theta) n_inc,j), t_max), t_max = 5. t_max bias cells per row; cell t (1..t_max) is set to mismatch iff
    tol_j < t, i.e. n_inc,j < ceil(t/(1-theta)). One global HD-CAM threshold t_max then reproduces the per-row rule exactly.
    Bias refresh after each recount: per cell 1 clearing write + a c-bit magnitude test against a constant (<= 2c cmp, bounded
    here by 2c) + 1 tagged write -> t_max (2c+2) cycles per class update.
  * Inference with QM: one threshold compare-tm (HD-CAM tunable Hamming threshold, Garzon et al. JSSC 2025) instead of an exact
    compare-tm: cycle count unchanged; t_max extra compared cells per row.
  * Polarity-aware QM (QP, study-5 revision): negated literals strict, tolerance on included positive literals,
    tau+_j = min(floor((1-theta) n_pos,j), t_max), exact for n_pos,j <= 6 at theta = 0.01 and 0.10. Inference: exact compare-tm with the
    positive-literal columns driven true (only negated literals can discharge) -> flag F1 (1 write); threshold compare-tm with the
    negated-literal columns driven true, on the same t_max bias cells -> flag F2 (1 write); exact compare on F1 = F2 = 1:
    3 compares + 2 writes instead of 1 compare = +4 cycles per clause evaluation. Training: an n_pos counter (c' = 3 bits),
    incremented in the same recount for the n positive literals (2c'+1 cmp + 2c'+2 wr each, reusing the include tag and inc_en),
    and the bias refresh from n_pos: t_max (2c'+2). Row: + c' counter bits + 2 flag bits (bias cells shared with QM).
  * R1: one extra compared bit (the literal's input value) in the Type Ib decrement-enable compare: 0 cycles.
Energy: first-order estimate = operation-count ratio (compares and writes on the 64-cell control subarray versus the same
operations on the literal subarray are taken as equal cost); reported only as a ratio to the base."""
import json, math

N = 8


def base_cycles(M, s):
    return 127 + 236 * math.log2(s) + (8 * N + 16) * M


TMAX = 5   # clamped QM tolerance (hardware rule): tol_j = min(floor((1-theta) n_inc,j), TMAX)


def overhead(M, L, theta=None):
    nsat = L + 1 if theta is None else max(L + 1, math.ceil(TMAX / (1 - theta)))   # counter must resolve n_inc up to here
    c = math.ceil(math.log2(nsat + 1))       # saturating counter
    recount = 1 + (4 * c + 5) * M + 3 + 2 * M  # recount, cap flag, extra Type Ib enable per literal
    tmax = 0 if theta is None else TMAX
    bias = tmax * (2 * c + 2)
    return dict(c=c, tmax=tmax, recount=recount, bias=bias, extra_cols=c + tmax)


def qp_overhead(M, L, theta):
    c = math.ceil(math.log2(L + 2))                      # n_inc counter for the literal budget
    cp = math.ceil(math.log2(math.ceil(TMAX / (1 - theta)) + 1))   # n_pos counter resolves the clamp point
    recount = 1 + (4 * c + 5) * M + 3 + 2 * M + (4 * cp + 3) * (M // 2)
    return dict(c=c, cp=cp, recount=recount, bias=TMAX * (2 * cp + 2), extra_cols=c + cp + TMAX + 2, infer=4)


def row_bits(M, K):
    return M * (N + 1) + 51 + math.ceil(math.log2(max(K, 2)))


if __name__ == "__main__":
    D = json.load(open("/home/claude/qmtm/paper/data/datasets.json"))
    out = {}
    for ds, d in D.items():
        M, K = 2 * d["features"], d["K"]
        W = row_bits(M, K)
        r = dict(M=M, K=K, row_bits=W)
        for cfg, L, th in [("L32", 32, None), ("L64", 64, None), ("QM90_L32", 32, 0.90), ("QM95_L32", 32, 0.95), ("QM95_L64", 64, 0.95)]:
            o = overhead(M, L, th)
            for s in (2.5, 10.0):
                b = base_cycles(M, s); x = o["recount"] + o["bias"]
                r[f"{cfg}_s{s}"] = dict(base_cycles=b, extra_cycles=x, train_overhead_pct=100 * x / b, extra_cols=o["extra_cols"],
                                       area_overhead_pct=100 * o["extra_cols"] / W, tmax=o["tmax"], c=o["c"])
        for cfg, L, th in [("QP1_L64", 64, 0.01), ("QP10_L64", 64, 0.10)]:
            o = qp_overhead(M, L, th)
            for s in (2.5, 10.0):
                b = base_cycles(M, s); x = o["recount"] + o["bias"]
                r[f"{cfg}_s{s}"] = dict(base_cycles=b, extra_cycles=x, train_overhead_pct=100 * x / b, extra_cols=o["extra_cols"],
                                       area_overhead_pct=100 * o["extra_cols"] / W, tmax=TMAX, c=o["c"], cp=o["cp"], infer_cycles=o["infer"],
                                       over_L64_pct=100 * (x - overhead(M, L)["recount"]) / b)
        out[ds] = r
    json.dump(out, open("/home/claude/qmtm/hw/qmtm_hw.json", "w"), indent=1)
    q = [out[d][k] for d in out for k in out[d] if k.startswith("QP")]
    print("QP: train +%.1f to +%.1f %%, over L64 +%.1f to +%.1f pp, row +%.4f to +%.3f %%" % (min(x["train_overhead_pct"] for x in q), max(x["train_overhead_pct"] for x in q),
          min(x["over_L64_pct"] for x in q), max(x["over_L64_pct"] for x in q), min(x["area_overhead_pct"] for x in q), max(x["area_overhead_pct"] for x in q)))
    for ds in ["mnist", "imdb", "r52_V0", "news20_V20000", "dorothea_V0"]:
        r = out[ds]
        print(f"{ds:15s} M={r['M']:6d} W={r['row_bits']:7d}  " + "  ".join(
            f"{k}: +{r[k + '_s10.0']['train_overhead_pct']:.1f}% cyc, +{r[k + '_s10.0']['area_overhead_pct']:.3f}% bits" for k in ["L32", "QM95_L64"]))

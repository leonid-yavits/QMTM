"""Binary versus graded clause evaluation in the AP(TM) cycle model (Section 8 of the QMTM paper).
Binary clause (strict TM, R1, QM): one compare + one write per inference (QM needs a threshold-sensed compare).
Graded clause (FPTM): miss count m_j clipped at L_F, then score max(0, b_j - m_j).
  Binary-sensed AP: built feature by feature. Per used feature: one compare on the feature's literal columns + one
  write (2 cycles), then either a saturating thermometer update (3 cycles per level, min(L_F, t) levels at feature t)
  or a c-bit counter increment with the AP(TM) saturating increment (2c+1 compares + 2c+2 writes, Sec. 4.3),
  c = ceil(log2(L_F + 1)). The cheaper counter is taken. Final clip: a c-bit subtraction, 4c cycles (bound).
  Threshold-sensed CAM: L_F threshold compares + writes at L_F reference settings (2 L_F cycles).
Vote: binary clauses one bit-plane; graded clauses c bit-planes, each reduced like the binary vote (5K+1+log2 rows).
Energy proxy (cell operations): compare of a full row = M (+ tau_max bias cells for QM) cells; bit-serial step = 1 cell
per row per cycle. All features in use: an upper bound on the FPTM cost. Break-even used-feature fraction: fraction
of features FPTM may use before the binary arm loses its energy-efficiency advantage (bit-serial case)."""
import math, json
TAU = 5
# encoding: n features, K classes, binary arm, its clauses per class, whether at equal accuracy, arm is QM?
CASES = [("R52 (V=1k)", 1000, 52, "R1", 10, True, False), ("R8 (V=5k)", 5000, 8, "R1", 20, True, False),
         ("UCI Adult", 162, 2, "TM", 40, True, False), ("MNIST", 784, 10, "QM", 160, False, True),
         ("Fashion-MNIST", 784, 10, "QM", 160, False, True), ("IMDb", 5000, 2, "R1", 160, False, False)]
def c_bits(LF): return math.ceil(math.log2(LF + 1))
def graded_bitserial(n, LF):
    c = c_bits(LF); thermo = sum(2 + 3 * min(LF, t) for t in range(1, n + 1)); binc = n * (2 + 4 * c + 3) + 4 * c
    return min(thermo, binc)
def rows_for(name, n, K, arm, C, eq, isqm, n_used=None, LFs=(4, 16, 64)):
    """Cost ratios for one encoding. n_used: features used by FPTM (default all n, the upper bound)."""
    nu = n if n_used is None else n_used
    M = 2 * n; rB, rF = C * K, 6 * K
    res = []
    for LF in LFs:
        c = c_bits(LF)
        voteB = 5 * K + 1 + math.ceil(math.log2(rB)); voteF = c * (5 * K + 1 + math.ceil(math.log2(rF)))
        latB = 2 + voteB
        eB = rB * (M + (TAU if isqm else 0) + 1)
        for sense in ("binary", "threshold"):
            if sense == "binary":
                if isqm: continue      # QM needs threshold sensing
                cF = graded_bitserial(nu, LF); eF = rF * cF
            else:
                cF = 2 * LF; eF = rF * LF * (M + 1)
            latF = cF + voteF
            res.append(dict(encoding=name, arm=arm, C=C, equal_accuracy=eq, LF=LF, sensing=sense, n_used=nu,
                            latency=latF / latB, thr_area=(latF / latB) * (rF / rB), eff=eF / eB, edp=(eF * latF) / (eB * latB),
                            breakeven=(min(1.0, eB / (rF * graded_bitserial(n, LF))) if sense == "binary" else None)))
    return res
def main():
    import os
    out = [r for case in CASES for r in rows_for(*case)]
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "binary_vs_graded.json"), "w"), indent=1)
    return out
if __name__ == "__main__":
    out = main()
    def rng(rows, k):
        v = [r[k] for r in rows]; return (min(v), max(v))
    for eq in (True, False):
        for sense in ("binary", "threshold"):
            rows = [r for r in out if r["equal_accuracy"] == eq and r["sensing"] == sense]
            if not rows: continue
            print(f"equal_acc={eq} sensing={sense}: n={len(rows)}")
            for k in ("latency", "thr_area", "eff", "edp"):
                lo, hi = rng(rows, k); print(f"   {k:9s} {lo:10.2f} to {hi:12.2f}")
            if sense == "binary":
                print("   breakeven used-feature fraction:", {f"{r['encoding']} LF={r['LF']}": round(r['breakeven'], 3) for r in rows})
    for r in out: print(r["encoding"], r["LF"], r["sensing"], {k: round(r[k], 2) for k in ("latency", "thr_area", "eff", "edp")})

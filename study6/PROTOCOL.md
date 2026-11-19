# Study 6 protocol: binary against graded clause evaluation (written 19 Nov 2026, before any study-6 run)

Purpose. Study 6 supports a re-scoped paper that compares binary clauses (vanilla TM, R1, QM: one compare per clause) with
graded clauses (Fuzzy-Pattern TM, FPTM: a failed-literal count clipped at L_F) for associative in-memory inference. It
replaces the bounds of the current hardware comparison with measured model statistics and maps accuracy against the number
of threshold searches per clause. All FPTM runs use the reference code (`fptm/FPTMVariants.jl` with MODE = 0, bit-exact to
upstream, commit in `fptm/UPSTREAM_COMMIT.txt`). Binary runs use `study1/train.py` unchanged.

Common settings. Study-1 encodings and splits (exported by `export6.py`). FPTM: T = 16, L = 2 L_F, S = 1000 if n >= 1000
else n, 256 states, include limit 128, 40 epochs, validation every 5 epochs. Every FPTM run logs, in addition to the
accuracies, the number of used features (features with at least one included literal in any clause), the used positive and
negative literal columns, all clause lengths, the histogram of clause scores on the test set, and the test accuracy of the
vote with every clause score clipped at Q, for Q in {1, 2, 3, 4, 8, 16, 32, 64} with Q <= L_F (Q threshold searches per
clause on a threshold-sensed CAM; Q = L_F is the full vote and must reproduce the test accuracy, checked by `fptm6_job.py`).

E1. FPTM re-run of Study 2 part A with logging. Nine datasets (adult, mnist, fmnist, har, imdb, r8_V5000, r52_V1000,
news20_V1000, sector_V1000) x C in {5, 10} (6 and 10 FPTM clauses per class) = 18 cells. Tuning: L_F in {4, 16, 64},
seed 1, last-epoch validation; ties to the smaller L_F (`select6.py`). Confirmation: seeds 101 to 105, test. These runs
replace the lost per-run records of Study 2 part A. The Study 2 part A contrasts are recomputed from them as a check; the
reported contrasts remain those of Section 6.6 unless the two disagree in sign, which would be reported.

E2. Vote levels. 6 clauses per class, L_F in {1, 2, 4, 8, 16, 32, 64}, the nine datasets, seeds 101 to 103, test. L_F = 1
gives a binary clause trained by FPTM. Descriptive: accuracy against the number of threshold searches per clause.

E3. FPTM clause budget. 2, 4, 6, 10, 20 and 40 clauses per class at the L_F selected in E1 for C = 5, on the six encodings of
the budget sweep (adult, r8_V5000, r52_V1000, mnist, fmnist, imdb), seeds 101 to 103, test. With the binary sweep this gives
the clause count at which each clause type reaches the other.

E4. Binary arms at 320 and 640 clauses per class on mnist, fmnist and imdb, where no binary arm reached FPTM with up to 160.
Arms: the most accurate binary configuration at C = 160 in the existing logs (QM95_L32 s = 10 on mnist, QM90_L32 s = 10 on
fmnist, re-tuned R1 AND_L64_fb1 s = 1.5 on imdb) and the most accurate vanilla TM configuration at C = 160. T = C/2, seeds
101 to 103, no re-tuning.

E5 (pending from round 5, separate patch `qmtm_clamp.patch`). QM accuracy with the hardware clamp tau_max = 5.

Analysis (`analyze6.py`, descriptive). Per cell: selected L_F, test accuracy, used-feature fraction, clause lengths,
quantized-vote accuracy. Equal-accuracy points from E3 and the binary logs (best arm over all logs, seeds 101 and above).
Table 9 cost ratios recomputed with the measured used-feature counts (`hw/binary_vs_graded.py`, `rows_for`). No new
hypothesis tests are planned beyond the E1 check; any test added later will be labelled exploratory.

## Amendments

# QMTM: qualified-majority clauses and targeted forgetting for clause-constrained Tsetlin machines

Code, protocols and raw result logs for the paper

> "Qualified-majority clauses and targeted forgetting for clause-constrained Tsetlin machines on associative processors"
> (under review).

Each protocol (`study*/PROTOCOL.md`) was written before the runs it covers and amended only before the affected
runs. The protocols are internal documents, not external registrations.

The paper studies two modifications of the Tsetlin machine (TM) for small clause budgets:

* **Relative qualified-majority (QM) clauses.** A clause fires if at most `floor((1 - theta) * n_inc)` of its
  `n_inc` included literals are false on the sample. `theta = 1` is the vanilla TM.
* **Targeted Type Ib feedback (R1).** A non-firing clause forgets only the literals that are false on the sample.

Both are combined with the literal budget `L` of pyTsetlinMachine and mapped onto the AP(TM) associative processor
(Expert Systems with Applications 334 (2026) 134830,
doi:10.1016/j.eswa.2026.134830).

## Layout

| Path | Content |
|---|---|
| `qm/pyTsetlinMachine/` | pyTsetlinMachine 0.6.6 sources with the QM, polarity-aware QM, R1 and fast-sampler patches applied (MIT license in `qm/LICENSE`) |
| `apply_*.py` | the patches, applicable to a clean copy of pyTsetlinMachine 0.6.6 |
| `qmtm.py` | Python wrapper (`QMTM` class) |
| `validate.py`, `check_numpy.py` | bit-exactness test against the unmodified library and an independent NumPy reference |
| `prep_data.py`, `prep_imdb.py` | MNIST, Fashion-MNIST, UCI Adult and IMDb encodings (as in AP(TM)) |
| `study1/` | Study 1: R1 and QM against the vanilla TM, 13 datasets, 20 encodings, 48 cells. `PROTOCOL.md`, data preparation, `train.py`, selection, statistics, job lists and all logs |
| `study2/` | Study 2: out-of-sample density rule (part C, six new datasets) and clause-budget sweep (part B, C = 40, 80, 160) |
| `study4/` | Baseline controls: extended vanilla TM family (L = 32, s = 1.5) on all study-1 and study-2 cells, and a re-tuned vanilla TM in the budget sweep |
| `study5/` | Symmetric control (s = 1.5 also offered to TMR and QM, and to QMR in Study 1), literal-budget ablation (TMR and QM without L), and the budget sweep with TMR and QM re-tuned. `PROTOCOL.md` with amendments A1 to A3 |
| `hw/qmtm_hw.py` | analytical AP(TM) cost model of the additions (projections, not measurements) |
| `hw/clausemax.py` | largest-clause check behind the clamped QM tolerance (`clausemax.out`) |
| `paper/` | scripts that build the paper's tables, number macros and figures from the logs |

## Studies and their names in the paper

| Folder | Name in the paper |
|---|---|
| `study1/` | Study 1 (Section 6.1) |
| `study2/` | Study 2: density rule (Section 6.2) and budget sweep (Section 6.5) |
| `study4/` | Baseline controls, TM_ext (Section 6.3) |
| `study5/` | Symmetric control (Section 6.3), literal-budget ablation (Section 6.4) and the sweep with the rule arm re-tuned (Section 6.5) |

There is no `study3/`: that study concerned a baseline outside the scope of the paper.

## Logs

Every training run appends one JSON line with the dataset, clause budget `C`, threshold `T`, specificity `s`,
configuration, seed, validation curve and test accuracy.

* `study1/logs/tune.jsonl` (1,560 tuning runs, seed 1, validation), `confirm.jsonl` (1,440 test runs, seeds 101 to 105),
  `selection.json`, `stats1.json`, `stats1.txt`.
* `study2/logs/tune.jsonl`, `confirmC.jsonl` (density rule), `sweep.jsonl` (budget sweep), `selection2.json`, `stats2.json`.
* `study4/logs/tuneD1_s1.jsonl`, `tuneD1_s2.jsonl`, `tuneD2.jsonl` (tuning), `confirmD1.jsonl`, `confirmD2.jsonl` (test runs),
  `selection4.json`, `stats4.json`, `sweep4.json`.
* `study5/logs/tuneE1_s1.jsonl`, `tuneE1_s2.jsonl` (336 and 60 tuning runs, symmetric control), `confirmE1.jsonl` (360 test runs),
  `E2.jsonl` (520 runs, literal-budget ablation, seeds 101 to 103), `tuneE1b.jsonl` and `confirmE1b.jsonl` (168 and 81 runs,
  budget sweep with TMR and QM re-tuned), `selection5.json`, `selectionE1b.json`, `stats5.json`, `stats5.txt`, `sweepE1b.json`.
  The files ending in `_noimdb` hold the 45-cell results computed before the IMDb data were restored (amendment A3).

## Reproducing

1. Build the library:
   ```
   cd qm/pyTsetlinMachine
   gcc -O3 -march=native -fPIC -shared -I. ConvolutionalTsetlinMachine.c EmbeddingTsetlinMachine.c \
       MultiClassConvolutionalTsetlinMachine.c Tools.c IndexedTsetlinMachine.c -o ../libTM.so -lm
   ```
2. Download the datasets and encode them with `prep_data.py`, `prep_imdb.py`, `study1/prep_*.py` and `study2/prep2.py`
   (each docstring lists the source files and the encoding). Data files are not redistributed here.
3. Study 1: `study1/pipeline.sh` (tuning, selection, confirmation), then `python3 study1/stats1.py`.
4. Study 2: run every line of `study2/jobs/tuneC.txt`, then `python3 study2/select2.py`, then `study2/pipeline4.sh`
   (density-rule confirmations and budget sweep), then `python3 study2/stats2.py`.
5. Baseline controls: `python3 study4/make_jobs4.py`, then `study4/pipeline6.sh` (tuning, selection, confirmation),
   then `python3 study4/stats4.py` and `python3 study4/sweep4.py`. `pipeline5.sh` to `pipeline8.sh` are the restarts
   used in our runs. Each skips runs that are already logged.
6. Symmetric control, ablation and re-tuned sweep: `python3 study5/make_jobs5.py` and `python3 study5/make_jobsE1b.py`, then
   `study5/run_all5.sh` (tuning, selection, test runs, ablation, sweep), then `study5/run_imdb5.sh` for the IMDb cells,
   then `python3 study5/stats5.py`.
7. Paper tables, number macros and figures: the scripts in `paper/scripts/`. `hw/qmtm_hw.py` evaluates the cost model for
   s = 1.5, 2.5, 5 and 10, the values selected in the experiments.

The scripts use absolute paths rooted at `/home/claude/qmtm` (code) and `/home/claude/data` (encoded data).
Replace these prefixes to run elsewhere.

All runs use `QMTM_FAST=1`: the literals that receive feedback are drawn with an independent Bernoulli(1/s) mask per
literal instead of pyTsetlinMachine's sampled-count scheme. The marginal selection probability is the same. With the
original sampler, `theta = 1` and standard feedback, the patched library reproduces pyTsetlinMachine bit for bit.

## Scope of this release

In `study1/PROTOCOL.md` and `study1/prep_more.py`, a reference to an unpublished manuscript was removed for anonymity.
Nothing else was changed. `study2/PROTOCOL.md` is reproduced unchanged. Its part A (a Fuzzy-Pattern TM baseline) is
outside the scope of the paper, and its code and runs are not included. The feedback-mask control planned in
`study5/PROTOCOL.md` (E3) was withdrawn before any of its runs (amendment A1).

# QMTM: qualified-majority clauses and targeted forgetting for clause-constrained Tsetlin machines

Code, pre-registered protocols and raw result logs for the paper

> "Qualified-majority clauses and targeted forgetting for clause-constrained Tsetlin machines on associative processors"
> (under review).

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
| `hw/qmtm_hw.py` | analytical AP(TM) cost model of the additions (projections, not measurements) |
| `hw/clausemax.py` | largest-clause check behind the clamped QM tolerance (`clausemax.out`) |
| `paper/` | scripts that build the paper's tables, number macros and figures from the logs |

## Logs

Every training run appends one JSON line with the dataset, clause budget `C`, threshold `T`, specificity `s`,
configuration, seed, validation curve and test accuracy.

* `study1/logs/tune.jsonl` (1,560 tuning runs, seed 1, validation), `confirm.jsonl` (1,440 test runs, seeds 101 to 105),
  `selection.json`, `stats1.json`, `stats1.txt`.
* `study2/logs/tune.jsonl`, `confirmC.jsonl` (density rule), `sweep.jsonl` (budget sweep), `selection2.json`, `stats2.json`.

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

The scripts use absolute paths rooted at `/home/claude/qmtm` (code) and `/home/claude/data` (encoded data).
Replace these prefixes to run elsewhere.

All runs use `QMTM_FAST=1`: the literals that receive feedback are drawn with an independent Bernoulli(1/s) mask per
literal instead of pyTsetlinMachine's sampled-count scheme. The marginal selection probability is the same. With the
original sampler, `theta = 1` and standard feedback, the patched library reproduces pyTsetlinMachine bit for bit.

## Scope of this release

`study2/PROTOCOL.md` is reproduced unchanged. In `study1/PROTOCOL.md` and `study1/prep_more.py`, a reference to an unpublished manuscript was removed for anonymity. Nothing else was changed. Its part A (a Fuzzy-Pattern TM baseline) is outside the scope of the
paper, and its code and runs are not included.

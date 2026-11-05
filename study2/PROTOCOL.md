# Study 2 protocol (written 1 Nov 2026, after study 1 and before any study-2 run)

Purpose: test the study-1 conclusions that a reviewer would challenge first.

## A. Fuzzy-Pattern TM (FPTM) baseline
Implementation: the authors' reference code FuzzyPatternTM.jl (github.com/BooBSD/FuzzyPatternTM, commit in fptm/UPSTREAM_COMMIT.txt), unmodified,
Julia 1.13.3, one thread per run. Driver fptm/run_fptm.jl, logging fptm/fptm_job.py.
Data: the study-1 encodings and splits (fptm/export.py reproduces study1/train.py's split).
Clauses: FPTM splits clauses equally by polarity, so clauses_num = 2*ceil(C/2) (C = 5 gives 6, one more than the TM arms).
Grid (per dataset and C): LF in {4, 16, 64}, L = 2*LF, T = 16, S = 1000 if n >= 1000 else n (s = round(n/S) random forgets,
following the authors' examples: s = 1 for 784-bit MNIST, s = n/1000 for IMDb), 256 states, include limit 128. 40 epochs.
Selection on validation (seed 1, last epoch), ties to smaller LF. Confirmation: seeds 101-105, test.
Scope (compute: the reference code gives feedback to all K classes per sample at O(n) per updated clause):
  stage A (always): adult, mnist, fmnist, imdb, r8_V5000, r52_V1000, har, phishing, dorothea_V5000, sector_V1000, news20_V1000;
  stage B (if time allows, run in this order): r52_V5000, news20_V5000, sector_V5000, rcv1_V5000, r52_V0.
  Excluded: dorothea_V0 (100,000 features exceed the implementation's 16-bit literal index), ledgar_V5000, sector_V20000,
  news20_V20000 (estimated 0.7-3.8 h per 40-epoch run).
Contrasts (Welch per cell, Holm within contrast over the cells run): FPTM - TM, FPTM - best QM-family arm, FPTM - best study-1 arm.
Study-1 arms are compared using their existing confirmation runs (seeds 101-105), unchanged.

## B. Clause-budget sweep
Datasets: the 8 study-1 variants with C = 5/10/20 (adult, mnist, fmnist, imdb, r8_V5000, r52_V1000, r52_V5000, r52_V0).
Arms: TM, TMR, QM, each with the (cfg, s) selected at C = 20 in study 1. C in {40, 80, 160}, T = C/2. Seeds 101-103. Test accuracy.
Reported: accuracy versus C per arm (with study-1 C = 5/10/20 seeds 101-103 for continuity), and the smallest C at which TM's mean
reaches the mean of TMR (or QM) at C = 5 and at C = 10. Descriptive; no new hypothesis test.

## C. Out-of-sample density rule
Rule fixed from study 1: training-set density < 3 % -> R1 family (TMR configs), otherwise QM family (QM configs).
New datasets (not used in study 1), chosen before any run:
  sparse text: AG News (4 classes), TREC-6 (6 classes), Emotion (dair-ai, 6 classes);
  dense: USPS (LIBSVM, 10 classes), letter (LIBSVM, 26 classes), satimage (LIBSVM, 6 classes).
Encodings: text as study 1 (binary bag of words, lowercase alphabetic tokens >= 2 chars, NLTK stopwords removed, df >= 2, chi-square
selection to V = 5,000 on the training part, or the full vocabulary if smaller); dense features: 10 quantile thresholds from the
training split (as HAR). Validation: the official one if present, else stratified 10 % of train (seed 0).
Grid: C in {5, 10}, s in {2.5, 10}, T = 3/5; arms TM {AND, AND_L64}, the rule arm (TMR configs or QM configs), and the other family
for reference. Selection on validation (seed 1); confirmation seeds 101-105 on test.
Hypothesis: the rule arm beats TM. Test: Welch per cell, Holm over the 12 cells; also a one-sided sign test over cells.
The rule is considered confirmed if it has more significant wins than losses and no significant loss; refuted otherwise.

No change after results are seen without an amendment recorded below.

## Amendments
A1 (1 Nov 2026, before any study-2 run). TREC-6 keeps stopwords: its classes are defined by question words (what, who, how, when,
where), which the NLTK stopword list removes. TREC-6 copy: SetFit/TREC-QC (coarse labels, label_coarse), because CogComp/trec is
script-based on Hugging Face and the original host is not reachable from the workspace. AG News: fancyzhx/ag_news. Emotion: dair-ai/emotion (split config).

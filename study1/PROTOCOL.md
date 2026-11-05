# Study 1 protocol (written 31 Oct 2026, before any study-1 result)

Question: do R1 (targeted Type Ib feedback) and qualified-majority clauses (QM, relative; QP, polarity-aware)
improve the Tsetlin machine over the vanilla TM at small clause budgets, and does any gain grow with input sparsity?

## Datasets
Dense or tabular: MNIST, Fashion-MNIST, UCI Adult, IMDb (encodings of AP(TM) Sec. 6.1),
UCI HAR and UCI Phishing when downloadable.
Sparse text: Reuters R8 and R52 (single-label ApteMod subsets, NLTK copy, prep_reuters.py), and, when downloadable,
20 Newsgroups, sector, rcv1.multiclass, LEDGAR, Dorothea.
Text encoding: binary bag of words, chi-square selection on the training part only.
Sparsity sweep: R52 (and later news20 / sector) at V = 1,000, 5,000 and the full vocabulary.
Validation: the dataset's own validation split, else a stratified 10 % of the training split (seed 0). The test split is used only in confirmation.

## Arms (each arm = a family of configurations; the best member is chosen on validation)
- TM (vanilla): AND, AND_L64
- TMR (R1): AND_L32_fb1, AND_L64_fb1
- QM (relative QM, standard feedback): QM90_L32_fb0, QM95_L32_fb0, QM95_L64_fb0
- QMR (relative QM + R1): QM90_L32_fb1, QM95_L32_fb1
- QP (polarity-aware QM, standard feedback): QP1_L64_fb0, QP10_L64_fb0
- QPR (polarity-aware QM + R1): QP1_L64_fb1, QP10_L64_fb1
Clause budgets C = 5, 10, 20 per class; T = 3, 5, 10. Specificity s in {2.5, 5, 10}, chosen per arm. 40 epochs, QMTM_FAST=1.

## Selection and confirmation
Tuning: seed 1, validation accuracy after 40 epochs. Per (dataset, C, arm): the (cfg, s) with the highest validation accuracy
(ties: smaller s, then first listed cfg). Confirmation: seeds 101-105 with the selected settings, test accuracy.

## Contrasts and statistics
Primary: TMR - TM; QM - TM; QP - TM; QMR - TMR; QPR - TMR; best QM-family arm (chosen on validation among QM, QP, QMR, QPR) - best non-QM arm (TM, TMR).
Unit: seed. Welch t-test on the five test accuracies per arm; Holm correction within each contrast over all (dataset, C) cells.
Report every cell, wins and losses, with mean and SD. Sparsity analysis: gain versus active features per sample and density (descriptive; Spearman correlation over cells).
No change to datasets, arms, grid or tests after results are seen without recording it below as an amendment.

## Amendments
A1 (31 Oct 2026, before any tuning result on these datasets was inspected; only a 2-epoch timing pilot of one
configuration, QP10_L64_fb1 at C = 20, was run). Downloaded: UCI HAR, UCI Phishing, Dorothea, sector (LIBSVM sector.scale files),
news20 and rcv1.multiclass (LIBSVM). LEDGAR could not be downloaded (Hugging Face CDN blocked) and is dropped.
Encodings: prep_more.py (docstring). rcv1 test: 50,000 documents sampled from the official test file (seed 0).
Because of compute (2 cores), the added datasets use C = 5 and 10 only and s in {2.5, 10}; all 13 configurations.
Variants: har, phishing, dorothea_V0 (100,000 features), dorothea_V5000, sector_V1000/V5000/V20000,
news20_V1000/V5000/V20000, rcv1_V5000. sector_V0 (55,197) and news20_V0 / rcv1_V20000 are excluded for run time and memory.
train.py now keeps inputs in uint8 (memory); verified bit-identical on a completed tuning run (adult, C=5, s=2.5, QM90_L32_fb0).
A2 (31 Oct 2026, before any LEDGAR run). LEDGAR became downloadable (Hugging Face CDN allowed) and is reinstated:
ledgar_V5000 (prep_ledgar.py), same grid as A1 (C = 5, 10; s in {2.5, 10}; 13 configurations).

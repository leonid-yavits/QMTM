# Study 5 protocol: symmetric specificity, literal-budget ablation and feedback-mask control
(written 8 Nov 2026, before any study-5 run)

Purpose: answer three co-author review points on the EAAI manuscript.
(1) Study 4 offered s = 1.5 (and L = 32) only to the vanilla TM. TMR and QM selected the smallest offered s (2.5) in
    50 and 40 of the 60 cells. On seed-1 validation, s = 1.5 alone gives TM_ext a median +1.8 pp, L = 32 alone +0.0 pp.
    The R1 and QM gains against TM_ext may therefore partly reflect the untuned s of the modified arms.
(2) Section 3.1 states that QM (and, implicitly, R1) needs the literal budget L. No run without L exists for TMR, QM or
    QMR in studies 1, 2 and 4. Earlier evidence is exploratory only (QM on the four AP(TM) benchmarks at large clause
    budgets, 28 Oct; R1 in the genome ablation, 30 Oct) and is not part of the released logs.
(3) All accuracy results use the independent Bernoulli(1/s) feedback mask (QMTM_FAST=1). pyTsetlinMachine uses a
    different sampler, and AP(TM) uses its own hardware mask.
Code: study1/train.py, with two additions that do not change any earlier run: the QM/QP configuration name may omit
"_L<L>" (no literal budget, as AND and AND_fb<fb> already allow), and runs with QMTM_FAST=0 log core = "orig".
All runs: 40 epochs, T as in the original cell.

## E1. Symmetric specificity (s = 1.5 for the modified arms)
Cells: the 48 study-1 cells and the 12 study-2 part-C cells (as in study 4, D1).
Extended families (each = the original family x (the cell's original s grid plus s = 1.5)):
  TMR_ext = {AND_L32_fb1, AND_L64_fb1};  QM_ext = {QM90_L32_fb0, QM95_L32_fb0, QM95_L64_fb0};
  QMR_ext = {QM90_L32_fb1, QM95_L32_fb1} (study-1 cells only, as QMR exists only in study 1).
New tuning runs: every member at s = 1.5, seed 1, validation. Original tuning runs are reused for the other s values.
Selection: highest seed-1 validation accuracy over the extended family; ties: smaller s, then the listed order.
If the selection differs from the original one, the new setting is confirmed with seeds 101-105 on test.
Otherwise the original confirmation runs are reused.
Analyses (Welch per cell, Holm within each contrast over the cells of that study, as in studies 1, 2 and 4):
  study 1: TMR_ext - TM_ext; QM_ext - TM_ext; QMR_ext - TMR_ext; best QM-family arm - best of {TM_ext, TMR_ext},
           with the best QM-family arm chosen on validation among {QM_ext, QMR_ext, QP, QPR} (QP and QPR unchanged);
           TMR_ext - TMR and QM_ext - QM (effect of offering s = 1.5 to the modified arm itself).
  study 2 C: rule arm - TM_ext, with the rule arm = TMR_ext below 3 % training density and QM_ext otherwise;
           confirmation criterion of study 2 unchanged (Holm wins, losses, sign test over the 12 cells).
QP and QPR are not re-tuned. This is stated wherever they appear.

## E2. Literal-budget ablation (no L)
Cells: the same 60 cells. Arms: TMR and QM (QMR is not ablated, for compute).
For each cell and arm, take the E1 selection (cfg, s) and remove "_L<L>" from cfg
(AND_L32_fb1 -> AND_fb1; QM95_L64_fb0 -> QM95_fb0). No further tuning.
Runs: seeds 101-103 on test (validation curve logged). Three seeds, for compute (2 cores).
Reported per cell: test accuracy without L minus with L on the same seeds 101-103, Welch with Holm over the cells of
each arm; the no-L arm against TM_ext on seeds 101-103 (Welch, Holm); included literals per clause with and without L.
Because three seeds give low power, the paper also reports, descriptively, the median difference and the number of
cells in which the no-L arm is within 1 pp of the budgeted arm.
Interpretation fixed in advance: the paper may state that an arm works without L only if the no-L arm has no
significant loss against the budgeted arm, its median difference is above -1 pp, and it is within 1 pp in at least
three quarters of the cells. Otherwise the paper states that the arm needs L, with the counts.

## E3. Feedback-mask control
E3a (pyTsetlinMachine sampler). Cells: every study-1 or study-2 C cell in which TMR_ext, QM_ext, QMR_ext or the rule arm
differs significantly from TM_ext after E1 (Holm), in either direction. In each such cell, the selected TM_ext setting
and the selected setting of each significant arm are re-run with QMTM_FAST=0 (pyTsetlinMachine 0.6.6 sampler),
seeds 101-105, test. Reported: the same contrast with both arms on the original sampler (Welch, Holm over these cells),
and the per-arm change in mean test accuracy between the two samplers.
E3b (AP(TM) mask). To be specified in an amendment from the authors' definition of the AP(TM) mask, before any E3b run.
Same cells, settings and seeds as E3a. If no definition is supplied, E3b is not run and the paper states this.

## E1b. Budget sweep (study-2 part B), run last and only if compute allows
8 encodings, C = 40, 80, 160, T = C/2. For TMR and QM: candidates = the original sweep setting plus every member of
the arm's family at s = 1.5. Selection on seed-1 validation (ties: smaller s, original setting first, then the listed
order); seeds 101-103 on test if changed. Descriptive, as part B and study 4 D2. If not run, the paper states that the
sweep keeps the original TMR and QM settings.

## Order of execution
E1 tuning; E1 selection and confirmation; E2; E3a (and E3b if specified); E1b.

Disclosure: study 5 was designed after all study-1, 2 and 4 results were known, in response to a co-author review.
It is reported as an added control, not as part of the original protocols. Study 4 is not changed.

## Amendments

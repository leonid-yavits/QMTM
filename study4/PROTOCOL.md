# Study 4 protocol: baseline-strength controls (written 5 Nov 2026, before any study-4 run)

Purpose: answer the referee objection that the vanilla TM baseline is weaker than the modified arms.
(1) The TM family {AND, AND_L64} has no literal budget of 32, while TMR, QM and QMR include L = 32.
(2) TM selects s = 2.5, the lowest grid value, in 42 of 48 study-1 cells.
(3) In the study-2 budget sweep (part B), TM at C = 40, 80, 160 reuses its C = 20 setting (as do TMR and QM).
Study 4 strengthens only the vanilla TM. All other arms, their selections and their test runs are unchanged.
Code: study1/train.py unchanged (QMTM_FAST=1, 40 epochs).

## D1. Extended TM family on the study-1 cells (48) and the study-2 part-C cells (12)
TM_ext = {AND, AND_L64, AND_L32} x (the cell's original s grid plus s = 1.5). T as in the original cell.
New tuning runs (seed 1, validation): AND_L32 at the original s grid; AND, AND_L64, AND_L32 at s = 1.5.
Selection: highest seed-1 validation accuracy over TM_ext (original and new runs); ties: smaller s, then the order
AND, AND_L64, AND_L32. If the selection differs from the original TM selection, the new setting is confirmed with
seeds 101-105 on test. Otherwise the original confirmation runs are reused.
Analyses (Welch per cell, Holm within contrast, as in studies 1 and 2), reported next to the original ones:
  study 1: TMR - TM_ext, QM - TM_ext, QP - TM_ext, best QM-family arm - best of {TM_ext, TMR};
  study 2 C: rule arm - TM_ext over the 12 cells, the confirmation criterion of study 2 unchanged.

## D2. Re-tuned TM in the budget sweep (study-2 part B: 8 encodings, C = 40, 80, 160, T = C/2)
Grid per (dataset, C): {AND_L32, AND_L64} x {1.5, 2.5}, plus the TM setting used in the original sweep.
Selection on seed-1 validation (ties: smaller s, then the listed order, original setting first).
Confirmation: seeds 101-103 on test (original sweep runs reused if the original setting is selected).
TMR and QM keep their original sweep runs. Reported: the sweep table and figure with TM re-tuned, and the smallest
re-tuned TM budget that reaches the rule arm at C = 5 and C = 10. Descriptive, as part B.

Disclosure: study 4 was designed after all study-1 and study-2 results were known, in response to a review.
It can only make the baseline stronger. It is reported as an added control, not as part of the original protocols.

## Amendments

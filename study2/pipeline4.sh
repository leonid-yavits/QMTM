#!/bin/sh
# Paper runs only (no FPTM): clause-budget sweep and density-rule confirmation. Re-runnable.
cd /home/claude/qmtm/study2
cat jobs/confirmC.txt jobs/sweepB.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo PAPERRUNSDONE

#!/bin/sh
# Study 4 tuning (PROTOCOL.md). Re-runnable: train.py skips completed keys.
cd /home/claude/qmtm/study4
cat jobs/tuneD1.txt jobs/tuneD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo TUNE4DONE

#!/bin/sh
# Study 5, E1 tuning. Re-runnable: train.py skips completed keys.
cd /home/claude/qmtm/study5
cat jobs/tuneE1_noimdb.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo TUNE5DONE

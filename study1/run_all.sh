#!/bin/sh
# Runs all study-1 tuning jobs (train.py skips completed keys), two at a time.
cd /home/claude/qmtm/study1
cat jobs/tune1.txt jobs/tune2.txt jobs/tune3.txt | xargs -P 2 -I{} sh -c "{}" >> logs/tune_run.out 2>&1
echo TUNEALLDONE

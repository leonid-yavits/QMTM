#!/bin/sh
# Study 4 full pipeline (PROTOCOL.md): tuning, selection, confirmation. Re-runnable: train.py skips completed keys.
cd /home/claude/qmtm/study4
cat jobs/tuneD1.txt jobs/tuneD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo TUNE4DONE
python3 select4.py
cat jobs/confirmD1.txt jobs/confirmD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo STUDY4DONE

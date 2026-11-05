#!/bin/sh
# Study 1 full pipeline: all tuning, then selection, then test confirmation (seeds 101-105).
# Re-runnable: train.py skips completed keys; select.py is deterministic.
cd /home/claude/qmtm/study1
cat jobs/tune1.txt jobs/tune2.txt jobs/tune3.txt | xargs -P 2 -I{} sh -c "{}" >> logs/tune_run.out 2>&1
echo TUNEALLDONE
python3 select.py
cat jobs/confirm.txt | xargs -P 2 -I{} sh -c "{}" >> logs/confirm_run.out 2>&1
echo CONFIRMALLDONE

#!/bin/sh
# Study 5 pipeline (PROTOCOL.md order: E1 tuning, E1 selection and confirmation, E2). Idempotent: train.py skips completed keys.
cd /home/claude/qmtm/study5
cat jobs/tuneE1_noimdb.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "TUNE5DONE $(date)" >> logs/stage.txt
python3 select5.py >> logs/stage.txt 2>&1
cat jobs/confirmE1.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "CONFIRM5DONE $(date)" >> logs/stage.txt
cat jobs/E2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "E2DONE $(date)" >> logs/stage.txt
python3 stats5.py > logs/stats5.txt 2>&1
echo "ALL5DONE $(date)" >> logs/stage.txt

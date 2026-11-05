#!/bin/sh
# Sweep confirmations for the cells whose D2 tuning is complete run first; the remaining IMDb tuning and its confirmations follow.
cd /home/claude/qmtm/study4
python3 select4.py
cat jobs/confirmD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo CONFIRMD2PARTDONE
cat jobs/tuneD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo TUNE4DONE
python3 select4.py
cat jobs/confirmD1.txt jobs/confirmD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo STUDY4DONE

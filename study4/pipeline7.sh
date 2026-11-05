#!/bin/sh
# D1 tuning is complete, so its confirmations run first. D2 tuning, selection and confirmations follow.
cd /home/claude/qmtm/study4
cat jobs/confirmD1.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo CONFIRMD1DONE
cat jobs/tuneD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo TUNE4DONE
python3 select4.py
cat jobs/confirmD1.txt jobs/confirmD2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo STUDY4DONE

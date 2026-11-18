#!/bin/sh
# Study 5, all stages in protocol order (idempotent; re-launch after any interruption).
cd /home/claude/qmtm/study5
./pipeline5.sh
grep -v ' imdb ' jobs/tuneE1b.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "TUNEE1BDONE $(date)" >> logs/stage.txt
python3 selectE1b.py >> logs/stage.txt 2>&1
cat jobs/confirmE1b.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
python3 selectE1b.py >> logs/stage.txt 2>&1
echo "E1BDONE $(date)" >> logs/stage.txt

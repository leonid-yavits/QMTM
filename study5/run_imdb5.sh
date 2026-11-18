#!/bin/sh
# Study 5, IMDb cells (amendment A2: run once the data are available; data restored 13 Nov 2026).
# Same order and scripts as run_all5.sh. Idempotent: train.py skips completed keys, so all other cells are no-ops.
cd /home/claude/qmtm/study5
grep ' imdb ' jobs/tuneE1.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "IMDB_TUNE5DONE $(date)" >> logs/stage.txt
python3 select5.py >> logs/stage.txt 2>&1
grep ' imdb ' jobs/confirmE1.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "IMDB_CONFIRM5DONE $(date)" >> logs/stage.txt
grep ' imdb ' jobs/E2.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "IMDB_E2DONE $(date)" >> logs/stage.txt
python3 stats5.py > logs/stats5.txt 2>&1
grep ' imdb ' jobs/tuneE1b.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
echo "IMDB_TUNEE1BDONE $(date)" >> logs/stage.txt
python3 selectE1b.py >> logs/stage.txt 2>&1
grep ' imdb ' jobs/confirmE1b.txt | xargs -P 2 -I{} sh -c "{}" >> logs/run.out 2>&1
python3 selectE1b.py >> logs/stage.txt 2>&1
echo "IMDB_ALLDONE $(date)" >> logs/stage.txt

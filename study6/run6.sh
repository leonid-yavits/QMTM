#!/usr/bin/env bash
# Study 6 pipeline. Run from the repository root on the lab server.
#   QMTM=<repo root, default /home/claude/qmtm>  QMTM_DATA=<FPTM export dir, default /home/claude/data/fptm>  JULIA=<julia binary>
#   P=<parallel jobs, default nproc>
# Every job skips runs already in its log, so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-$(nproc)}
run() { echo "== $1 ($(wc -l < study6/jobs/$1) jobs)"; xargs -P "$P" -I{} bash -c '{} || echo "JOB FAILED: {}" >&2' < study6/jobs/$1; }
# 0. export the nine encodings for Julia (same splits as study1/train.py)
for ds in adult mnist fmnist har imdb r8_V5000 r52_V1000 news20_V1000 sector_V1000; do
  [ -f "${QMTM_DATA:-/home/claude/data/fptm}/$ds/meta.txt" ] || python3 study6/export6.py "$ds"
done
python3 study6/make_jobs6.py tune
run E1_tune.txt
python3 study6/select6.py
python3 study6/make_jobs6.py confirm
run E1_confirm.txt
run E2_levels.txt
run E3_clauses.txt
run E4_binary.txt
python3 study6/analyze6.py

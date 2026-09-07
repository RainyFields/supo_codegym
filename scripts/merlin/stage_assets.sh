#!/bin/bash
# Snapshot the repo (supo_codegym + patched external/verl, no .git) to HDFS job-assets.
# The venv/python tarball and the byted-wandb overlay are staged separately
# (see logs/stage_venv.log; rebuild with /tmp/supo_stage/stage_venv.sh only when the venv changes).
set -euo pipefail
H=/opt/tiger/yarn_deploy/hadoop/bin/hdfs
export HADOOP_OPTS="-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true" HADOOP_CLIENT_OPTS="-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true"  # hadoop-env.sh pins preferIPv4Stack=true -> "Protocol family unavailable" on IPv6 namenode failover
DST=hdfs://harunava/home/byte_arnold_va_ssd/mlsys/users/xiaoxuan/supo_codegym/job-assets
TMP=/tmp/supo_stage; mkdir -p $TMP
cd /
tar czf $TMP/supo-repo.tar.gz \
  --exclude='home/tiger/xiaoxuan/supo_codegym/logs' --exclude='*/__pycache__' --exclude='*.pyc' \
  --exclude='home/tiger/xiaoxuan/supo_codegym/.git' --exclude='home/tiger/xiaoxuan/external/verl/.git' \
  --exclude='home/tiger/xiaoxuan/external/verl/.venv' --exclude='home/tiger/xiaoxuan/supo_codegym/wandb' \
  home/tiger/xiaoxuan/supo_codegym home/tiger/xiaoxuan/external/verl
ls -la $TMP/supo-repo.tar.gz
$H dfs -put -f $TMP/supo-repo.tar.gz /home/tiger/xiaoxuan/supo_codegym/scripts/merlin/job_entrypoint.sh $DST/ 2>&1 | grep -v "config refresh\|lock file" || true
$H dfs -ls $DST 2>&1 | grep -v "config refresh\|lock file"
echo "REPO_STAGE_DONE $(TZ=America/Los_Angeles date)"

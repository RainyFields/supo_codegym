#!/bin/bash
# tar repo, then upload supo-repo.tar.gz + job_entrypoint.sh: CLI(IPv4) -> CLI(IPv6) -> fuse; verify md5 via fuse
H=/opt/tiger/yarn_deploy/hadoop/bin/hdfs; DST=hdfs://harunava/home/byte_arnold_va_ssd/mlsys/users/xiaoxuan/supo_codegym/job-assets
FUSE=/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/job-assets; SRC=/home/tiger/xiaoxuan/supo_codegym/scripts/merlin/job_entrypoint.sh
cd / && tar czf /tmp/supo_stage/supo-repo.tar.gz --exclude='home/tiger/xiaoxuan/supo_codegym/logs' --exclude='*/__pycache__' --exclude='*.pyc' --exclude='home/tiger/xiaoxuan/supo_codegym/.git' --exclude='home/tiger/xiaoxuan/external/verl/.git' --exclude='home/tiger/xiaoxuan/external/verl/.venv' --exclude='home/tiger/xiaoxuan/supo_codegym/wandb' home/tiger/xiaoxuan/supo_codegym home/tiger/xiaoxuan/external/verl
echo "[stage] tar done $(date)"; ls -la /tmp/supo_stage/supo-repo.tar.gz | awk '{print $5}'
ok=0
for flags in "-Djava.net.preferIPv4Stack=true" "-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true"; do
  export HADOOP_OPTS="$flags" HADOOP_CLIENT_OPTS="$flags"
  if timeout 600 $H dfs -put -f /tmp/supo_stage/supo-repo.tar.gz $SRC $DST/ 2>&1 | grep -q 'put: Failed'; then echo "[stage] CLI put failed with $flags"; else ok=1; echo "[stage] CLI put ok with $flags"; break; fi
done
if [ $ok = 0 ]; then echo "[stage] fuse fallback $(date)"; cp /tmp/supo_stage/supo-repo.tar.gz $FUSE/supo-repo.tar.gz.tmp && mv -f $FUSE/supo-repo.tar.gz.tmp $FUSE/supo-repo.tar.gz; cp $SRC $FUSE/job_entrypoint.sh; fi
sleep 5; md5sum /tmp/supo_stage/supo-repo.tar.gz $FUSE/supo-repo.tar.gz $SRC $FUSE/job_entrypoint.sh | cut -c1-12 | paste -sd' '
echo "[stage] SMALL_STAGE_DONE $(date)"

#!/usr/bin/env python3
"""Emit a `merlin-cli job-v2 runs create` spec for one training arm.

Mirrors the user's proven ARCO batch-job spec (group 765 ark-eng-algorithm, cluster 4
cloudnative-maliva, image modelchef-gpu:1.0.0.38, HDFS volume, entrypoint copied from
HDFS job-assets). GPU type strings use underscores (create API requirement).

  python scripts/merlin/make_job_spec.py --arm supo --gpu h100 > jobs/supo_h100.json
"""
import argparse
import json

QUEUES = {
    "h100": ("H100_SXM_80GB", "compute-89-aliyun.va-cloudnative-aigcp-ark.eng.algorithm-guarantee"),
    "a100": ("A100_SXM_80GB", "compute-89-aliyun.va-cloudnative-ai-ark.eng.algorithm-guarantee"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["grpo", "supo"])
ap.add_argument("--gpu", default="h100", choices=list(QUEUES))
ap.add_argument("--n_gpus", type=int, default=8)
ap.add_argument("--run_tag", default="")
ap.add_argument("--total_steps", default="100")
ap.add_argument("--save_freq", default="10")
ap.add_argument("--test_freq", default="10")
ap.add_argument("--extra_env", default="", help="comma-separated K=V pairs forwarded to the entrypoint")
ap.add_argument("--robust", action="store_true", help="use Merlin Robust Training (auto-restart) with role name executor")
args = ap.parse_args()

gpu_type, queue = QUEUES[args.gpu]
env = {
    "ARM": args.arm,
    "RUN_TAG": args.run_tag,
    "TOTAL_STEPS": args.total_steps,
    "SAVE_FREQ": args.save_freq,
    "TEST_FREQ": args.test_freq,
    "PUT_PAR": "16",
    "MAX_CKPT_KEEP": "2",   # 2 local ckpts (~230 GB) so a slow mirror is not rotated away         # parallel hdfs puts for the checkpoint mirror (~55 MB/s at 8)
    "MIN_GPUS": str(args.n_gpus),
    "N_GPUS": str(args.n_gpus),
}
for kv in filter(None, args.extra_env.split(",")):
    k, v = kv.split("=", 1)
    env[k] = v
role_name = "executor" if args.robust else "worker"
spec = {
    "name": f"supo-codegym-{args.arm}{args.run_tag}-{args.gpu}",
    "namespace": "/user/xiaoxuan.lei",
    "job_type": "train",
    "description": f"SUPO Table-1 CodeGym replication, arm={args.arm}, Qwen3.5-9B, verl (arXiv 2510.06727)",
    "attachments": [
        {
            "kind": "hdfs_volume",
            "hdfs_volume_attachment": {
                "hdfs_volumes": [
                    {
                        "access_mode": "RW",
                        "extra": "",
                        "mnt": "/mnt/hdfs/mlsys",
                        "path": "hdfs://harunava/home/byte_arnold_va_ssd/mlsys",
                        "roles": [role_name],
                    }
                ]
            },
        }
    ]
    + ([{"kind": "robust", "robust_attachment": {"use_robust_training": True}}] if args.robust else []),
    "job_config": {
        "job_template_config": {
            "entrypoint_full_script": (
                "cp /mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/job-assets/job_entrypoint.sh /tmp/supo_entry.sh "
                "&& bash /tmp/supo_entry.sh"
            ),
            "git_repo": {"repo_name": ""},
            "image_meta": {"image_source": "url", "image_url": "aliyun-va-hub.byted.org/arnold/modelchef-gpu:1.0.0.38"},
            "env_map": env,
        }
    },
    "resource_config": {
        "resource_backend": "arnold",
        "arnold_resource_config": {
            "cluster_ids": [4],
            "cluster_select_mode": "single",
            "group_id": 765,
            "roles": [
                {
                    "name": role_name,
                    "num": 1,
                    "gpu": args.n_gpus,
                    "gpu_type": gpu_type,
                    "cpu": 96 if args.n_gpus == 8 else 12 * args.n_gpus,
                    "memory_mb": 1544192 if args.n_gpus == 8 else 196608 * args.n_gpus,
                    "queue_name": queue,
                }
            ],
        },
    },
}
print(json.dumps(spec, indent=1))

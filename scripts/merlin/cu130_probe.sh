#!/bin/bash
# 1-GPU probe: does the full cu130 stack (torch 2.11+cu130, flash-attn 2.8.3 cu13, vLLM 0.24 cu130) run on
# an R535 pod when the CUDA 13.0 forward-compat libcuda is put on LD_LIBRARY_PATH?
set -uo pipefail
XD=/home/tiger/xiaoxuan; ASSETS=/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/job-assets
echo "[probe] node=$(hostname) $(date)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
mkdir -p /tmp/c13 && tar xzf $ASSETS/cuda-13.0-compat.tar.gz -C /tmp/c13 && echo "[probe] compat: $(ls /tmp/c13/compat | tr '\n' ' ')"
echo "[probe] restoring cu130 venv from $(ls $ASSETS/cu130_parts/ | grep -c part) chunks $(date)"; cat $ASSETS/cu130_parts/envs-supo-cu130.tar.gz.part* > /tmp/envs-supo-cu130.tar.gz && echo "[probe] tarball md5 $(md5sum /tmp/envs-supo-cu130.tar.gz | cut -c1-12) (manifest: $(cat $ASSETS/cu130_parts/MANIFEST))" && tar xzf /tmp/envs-supo-cu130.tar.gz -C / || { echo "[probe] FATAL venv untar"; exit 43; }
PY=$XD/envs/supo-cu130/bin/python; $PY -c "import torch; print('[probe] torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "[probe] staging model $(date)"; mkdir -p /tmp/models && cp -r /mnt/hdfs/mlsys/models/Qwen3.5-9B /tmp/models/ && echo "[probe] model staged $(date)"
export LD_LIBRARY_PATH=/tmp/c13/compat:${LD_LIBRARY_PATH:-} TMPDIR=/tmp/supo_tmp; mkdir -p $TMPDIR
echo "===== A. torch without compat (expect failure) ====="
env -u LD_LIBRARY_PATH $PY -c "import torch; print(torch.zeros(1,device='cuda')+1)" 2>&1 | tail -1
echo "===== B. torch with CUDA-13.0 compat ====="
$PY - <<'PYEOF' 2>&1 | grep -v Warning
import torch, time
print("driver_api", torch.cuda.driver_version() if hasattr(torch.cuda,'driver_version') else '?', "device", torch.cuda.get_device_name(0))
a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); torch.cuda.synchronize(); t=time.time()
for _ in range(20): c=a@a
torch.cuda.synchronize(); print(f"matmul 8192^2 bf16 x20: {(time.time()-t):.2f}s  ({20*2*8192**3/(time.time()-t)/1e12:.0f} TFLOP/s) sum={c.float().sum().item():.3e}")
x=torch.randn(1000,1000,device='cuda'); print("cublas/cusolver:", torch.linalg.inv(x).shape, "conv:", torch.nn.functional.conv2d(torch.randn(1,3,64,64,device='cuda'), torch.randn(8,3,3,3,device='cuda')).shape)
import torch.distributed as dist, os
os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29511', RANK='0', WORLD_SIZE='1')
dist.init_process_group('nccl'); t=torch.ones(1024,device='cuda'); dist.all_reduce(t); print("nccl all_reduce OK", t[0].item()); dist.destroy_process_group()
from flash_attn import flash_attn_varlen_func
q=torch.randn(2048,16,128,device='cuda',dtype=torch.bfloat16); cu=torch.tensor([0,1024,2048],device='cuda',dtype=torch.int32)
o=flash_attn_varlen_func(q,q,q,cu,cu,1024,1024,causal=True); print("flash_attn varlen OK", o.shape, o.float().abs().mean().item())
print("B OK")
PYEOF
echo "===== C. vLLM generate (Qwen3.5-9B, 1 GPU) ====="
VLLM_USE_FLASHINFER_SAMPLER=0 timeout 1500 $PY - <<'PYEOF' 2>&1 | grep -v Warning | grep -i 'probe\|error\|Traceback\|vllm\|generated\|tok/s' | tail -25
import time
from vllm import LLM, SamplingParams
t=time.time()
llm=LLM(model='/tmp/models/Qwen3.5-9B', dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=False, additional_config={'gdn_prefill_backend':'triton'})
print(f"[probe] vLLM engine up in {time.time()-t:.0f}s")
sp=SamplingParams(temperature=0, max_tokens=128)
t=time.time(); outs=llm.generate(["Write a Python function that returns the n-th Fibonacci number, then briefly explain it."]*8, sp); dt=time.time()-t
ntok=sum(len(o.outputs[0].token_ids) for o in outs); print(f"[probe] generated {ntok} tokens for 8 prompts in {dt:.1f}s ({ntok/dt:.0f} tok/s)")
print("[probe] sample:", outs[0].outputs[0].text[:300].replace('\n',' | ')); print("[probe] C OK")
PYEOF
echo "[probe] done $(date)"; exit 0

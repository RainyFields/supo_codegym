"""Unit tests for the SUPO advantage estimator and the trainer-side overlong mask / dummy padding."""
import types
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_supo_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_advantage():
    # prompt A: 2 rollouts (r=1 with 3 trajs, r=0 with 1 traj); prompt B: 2 rollouts both r=1 (std 0)
    uid = np.array(["A", "A", "A", "A", "B", "B"], dtype=object)
    gen_uid = np.array(["a1", "a1", "a1", "a2", "b1", "b2"], dtype=object)
    L = 5
    rewards = torch.zeros(6, L)
    for i, r in enumerate([1, 1, 1, 0, 1, 1]):
        rewards[i, -1] = r
    mask = torch.ones(6, L)
    mask[1, 3:] = 0
    adv, ret = compute_supo_advantage(rewards, mask, uid, gen_uid)
    # group A stats over rollouts {1, 0}: mean .5, std (unbiased) .7071 -> adv +-0.7071
    assert torch.allclose(adv[0, 0], torch.tensor(0.5 / (0.70710678 + 1e-6)), atol=1e-4), adv[0, 0]
    assert torch.allclose(adv[0], adv[2]) and torch.allclose(adv[0, :3], adv[1, :3])
    assert adv[1, 3] == 0 and adv[1, 4] == 0  # masked tokens
    assert torch.allclose(adv[3, 0], -adv[0, 0])
    assert torch.all(adv[4:] == 0)  # zero-variance group
    # sanity: identical to GRPO when there is one trajectory per rollout: group [1,0,0]
    r3 = torch.zeros(3, L); r3[0, -1] = 1
    adv2, _ = compute_supo_advantage(r3, torch.ones(3, L), np.array(["C"] * 3, dtype=object), np.array(["c1", "c2", "c3"], dtype=object))
    assert torch.allclose(adv2[0, 0], torch.tensor((2 / 3) / (0.57735 + 1e-6)), atol=1e-3), adv2[0, 0]
    assert torch.allclose(adv2[1, 0], torch.tensor((-1 / 3) / (0.57735 + 1e-6)), atol=1e-3)
    # single-rollout group -> zero advantage (mean 0, std 1 convention: adv = score, zeroed by design below)
    adv3, _ = compute_supo_advantage(r3[:1], torch.ones(1, L), np.array(["D"], dtype=object), np.array(["d1"], dtype=object))
    assert torch.allclose(adv3[0, 0], torch.tensor(1.0))
    print("advantage ok")


def test_postprocess():
    n, L = 13, 6
    batch = TensorDict(
        {
            "response_mask": torch.ones(n, L, dtype=torch.long),
            "rm_scores": torch.zeros(n, L),
            "attention_mask": torch.ones(n, 2 * L, dtype=torch.long),
        },
        batch_size=n,
    )
    gen_uid = np.array([f"g{i//2}" for i in range(n)], dtype=object)
    overlong = np.array([i % 5 == 0 for i in range(n)], dtype=object)
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array([f"u{i//4}" for i in range(n)], dtype=object),
            "gen_uid": gen_uid,
            "overlong": overlong,
            "finished": np.array([not o for o in overlong], dtype=object),
            "num_tool_calls": np.array([3] * n, dtype=object),
            "num_summaries": np.array([1] * n, dtype=object),
        },
        meta_info={"global_steps": 1},
    )
    fake_self = types.SimpleNamespace(config=OmegaConf.create({"trainer": {"n_gpus_per_node": 8, "nnodes": 1}}))
    out, metrics = RayPPOTrainer._supo_postprocess_batch(fake_self, data)
    assert len(out) == 16 and metrics["supo/num_dummy_rows"] == 3
    rm = out.batch["response_mask"]
    for i in range(n):
        assert (rm[i].sum() == 0) == bool(overlong[i])
    assert rm[n:].sum() == 0
    assert len(set(out.non_tensor_batch["uid"][n:].tolist())) == 3 and all(u.startswith("dummy") for u in out.non_tensor_batch["uid"][n:])
    assert out.meta_info["global_steps"] == 1
    assert abs(metrics["supo/trajs_per_rollout"] - 13 / 7) < 1e-6
    chunks = out.chunk(8)
    assert all(len(c) == 2 for c in chunks)
    print("postprocess ok", {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    test_advantage()
    test_postprocess()
    print("ALL ADVANTAGE/PADDING TESTS PASSED")

# Time distribution — 66 steps (35..100), mean step 875s (14.6 min)

| phase                                         | mean s | median s | max s  | share of step | n  |
|-----------------------------------------------|--------|----------|--------|---------------|----|
| rollout generation (agent loop, all rollouts) | 571.4  | 486.4    | 1429.8 | 65.3%         | 65 |
| old log-prob recompute (actor fwd)            | 46.2   | 44.2     | 141.8  | 5.3%          | 65 |
| advantage computation                         | 1.3    | 1.3      | 1.8    | 0.1%          | 65 |
| actor update (PPO mini-batches)               | 244.8  | 238.8    | 463.8  | 28.0%         | 65 |
| checkpoint save                               | 29.1   | 30.0     | 30.8   | 3.3%          | 13 |
| validation (eval set)                         | 329.6  | 320.4    | 370.6  | 37.7%         | 7  |
| TOTAL step                                    | 875.0  | 793.8    | 1737.4 | 100.0%        | 65 |

## Rollout statistics

| quantity                       | mean over steps | last step |
|--------------------------------|-----------------|-----------|
| train reward mean              | 0.908           | 0.952     |
| response tokens per row (mean) | 5591.322        | 5592.765  |

Bottleneck: rollout generation (64% of step time). Generation is bounded by the slowest rollout (see agent_loop/slowest/*): long-horizon rollouts (many turns) dominate wall-clock, not tokens.

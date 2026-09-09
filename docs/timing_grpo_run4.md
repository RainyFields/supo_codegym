# Time distribution — 17 steps (0..16), mean step 665s (11.1 min)

| phase                                         | mean s | median s | max s  | share of step | n  |
|-----------------------------------------------|--------|----------|--------|---------------|----|
| rollout generation (agent loop, all rollouts) | 423.6  | 371.1    | 996.0  | 63.7%         | 16 |
| old log-prob recompute (actor fwd)            | 41.9   | 35.0     | 84.8   | 6.3%          | 16 |
| advantage computation                         | 1.3    | 1.2      | 1.7    | 0.2%          | 16 |
| actor update (PPO mini-batches)               | 188.0  | 176.1    | 320.5  | 28.3%         | 16 |
| checkpoint save                               | 27.2   | 30.4     | 32.4   | 4.1%          | 3  |
| validation (eval set)                         | 340.5  | 340.5    | 340.5  | 51.2%         | 1  |
| TOTAL step                                    | 665.2  | 621.9    | 1199.1 | 100.0%        | 16 |

## Rollout statistics

| quantity                       | mean over steps | last step |
|--------------------------------|-----------------|-----------|
| train reward mean              | 0.861           | 0.888     |
| response tokens per row (mean) | 3708.531        | 4493.564  |

Bottleneck: rollout generation (60% of step time). Generation is bounded by the slowest rollout (see agent_loop/slowest/*): long-horizon rollouts (many turns) dominate wall-clock, not tokens.

# Time distribution — 61 steps (40..100), mean step 661s (11.0 min)

| phase                                         | mean s | median s | max s  | share of step | n  |
|-----------------------------------------------|--------|----------|--------|---------------|----|
| rollout generation (agent loop, all rollouts) | 534.1  | 404.1    | 1703.2 | 80.8%         | 60 |
| old log-prob recompute (actor fwd)            | 20.4   | 19.3     | 79.3   | 3.1%          | 60 |
| advantage computation                         | 0.5    | 0.5      | 0.9    | 0.1%          | 60 |
| actor update (PPO mini-batches)               | 92.9   | 91.3     | 190.7  | 14.1%         | 60 |
| checkpoint save                               | 30.4   | 31.1     | 32.5   | 4.6%          | 12 |
| validation (eval set)                         | 290.2  | 260.8    | 438.6  | 43.9%         | 6  |
| TOTAL step                                    | 660.6  | 539.3    | 1821.5 | 100.0%        | 60 |

## Rollout statistics

| quantity                       | mean over steps | last step |
|--------------------------------|-----------------|-----------|
| train reward mean              | 0.857           | 0.916     |
| response tokens per row (mean) | 1640.215        | 1686.454  |

Bottleneck: rollout generation (80% of step time). Generation is bounded by the slowest rollout (see agent_loop/slowest/*): long-horizon rollouts (many turns) dominate wall-clock, not tokens.

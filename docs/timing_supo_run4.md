# Time distribution — 50 steps (0..49), mean step 674s (11.2 min)

| phase                                         | mean s | median s | max s  | share of step | n  |
|-----------------------------------------------|--------|----------|--------|---------------|----|
| rollout generation (agent loop, all rollouts) | 546.7  | 416.2    | 1735.2 | 81.1%         | 49 |
| old log-prob recompute (actor fwd)            | 20.6   | 19.3     | 77.3   | 3.1%          | 49 |
| advantage computation                         | 0.5    | 0.5      | 1.1    | 0.1%          | 49 |
| actor update (PPO mini-batches)               | 93.7   | 90.8     | 220.0  | 13.9%         | 49 |
| checkpoint save                               | 30.6   | 31.7     | 34.5   | 4.5%          | 9  |
| validation (eval set)                         | 290.9  | 299.0    | 376.2  | 43.2%         | 4  |
| TOTAL step                                    | 673.8  | 556.2    | 1852.3 | 100.0%        | 49 |

## Rollout statistics

| quantity                       | mean over steps | last step |
|--------------------------------|-----------------|-----------|
| train reward mean              | 0.830           | 0.835     |
| response tokens per row (mean) | 1642.162        | 1573.179  |

Bottleneck: rollout generation (80% of step time). Generation is bounded by the slowest rollout (see agent_loop/slowest/*): long-horizon rollouts (many turns) dominate wall-clock, not tokens.

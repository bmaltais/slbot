# Karpathy Mod Experiment Report
*Generated: 2026-04-08 21:52:21*

## Summary

| Metric | Value |
|--------|-------|
| Total experiments | 339 |
| Keep rate | 66.4% |
| Kept | 225 |
| Discarded | 114 |
| Inconclusive | 0 |
| Best improvement | +4.41% |
| Worst regression | -8.72% |
| Avg improvement | +0.69% |
| Trend | **IMPROVING** |

## Strategy Effectiveness

| Strategy | Count | Kept | Keep Rate | Avg Improvement | Best |
|----------|------:|-----:|----------:|----------------:|-----:|
| explore | 115 | 78 | 67.8% | +0.54% | +4.28% |
| radical | 114 | 77 | 67.5% | +0.68% | +4.28% |
| tweak | 110 | 70 | 63.6% | +0.85% | +4.41% |

## Stage Performance

| Stage | Name | Count | Kept | Keep Rate | Avg Improvement |
|------:|------|------:|-----:|----------:|----------------:|
| S1 | FOOD_VECTOR | 80 | 60 | 75.0% | +0.48% |
| S2 | WALL_AVOID | 61 | 49 | 80.3% | +0.63% |
| S3 | ENEMY_AVOID | 62 | 42 | 67.7% | +2.35% |
| S4 | MASS_MANAGEMENT | 58 | 35 | 60.3% | +0.14% |
| S5 | MASTERY_SURVIVAL | 64 | 39 | 60.9% | +0.05% |
| S6 | APEX_PREDATOR | 14 | 0 | 0.0% | -0.09% |

## Top 5 Best Kept Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7334 | +4.41% | tweak | S3 | [tweak] S3/ENEMY_AVOID: enemy_approach_penalty: 0.5000 -> 0.5727 |
| R7342 | +4.28% | tweak | S3 | [tweak] S3/ENEMY_AVOID: gamma: 0.9500 -> 0.9368 |
| R7349 | +4.28% | explore | S3 | [explore] S3/ENEMY_AVOID: death_snake: -40 -> -39.59706296492766; enemy_proximit |
| R7354 | +4.28% | tweak | S3 | [tweak] S3/ENEMY_AVOID: starvation_penalty: 0.0089 -> 0.0087 |
| R7358 | +4.28% | tweak | S3 | [tweak] S3/ENEMY_AVOID: food_shaping: 0.1000 -> 0.1044 |

## Top 5 Worst Discarded Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7317 | -8.72% | explore | S3 | [explore] S3/ENEMY_AVOID: starvation_penalty: 0.0080 -> 0.0114; food_reward: 5.0 |
| R7322 | -8.71% | radical | S3 | [radical] S3/ENEMY_AVOID: starvation_grace_steps: 60 -> 74; death_wall: -40 -> - |
| R7304 | -8.57% | explore | S3 | [explore] S3/ENEMY_AVOID: gamma: 0.9500 -> 0.9291; enemy_proximity_penalty: 1.50 |
| R7308 | -5.96% | explore | S2 | [explore] S2/WALL_AVOID: food_shaping: 0.1500 -> 0.0965; survival_escalation: 0. |
| R7318 | -4.41% | tweak | S2 | [tweak] S2/WALL_AVOID: gamma: 0.9300 -> 0.9399 |

## Charts

### Experiment Overview Dashboard
![Experiment Overview Dashboard](charts/karpathy_overview.png)

### Strategy Effectiveness
![Strategy Effectiveness](charts/karpathy_strategy.png)

### Stage Analysis
![Stage Analysis](charts/karpathy_stages.png)

### Metric Evolution
![Metric Evolution](charts/karpathy_metrics.png)

### Parameter Impact Analysis
![Parameter Impact Analysis](charts/karpathy_parameters.png)

### Improvement Distribution
![Improvement Distribution](charts/karpathy_distribution.png)

### Strategy x Stage Heatmap
![Strategy x Stage Heatmap](charts/karpathy_heatmap.png)

### Interactive Explorer
[Open Experiment Explorer](charts/karpathy_experiment_explorer.html)

## Trend Assessment

**IMPROVING**

Keep rate is increasing over time. The mutation system is learning what works.

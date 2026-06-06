# oxide-canary

> Progressive GPU kernel rollout with ternary health verdicts and automatic rollback.

## Background Theory

Deploying a new GPU kernel to a live fleet is one of the riskiest operations in accelerated computing. Unlike CPU code, a buggy kernel can crash the display driver, corrupt GPU memory shared by other processes, or produce silent numerical errors that cascade through a model's forward pass. Traditional deployment strategies — blue/green cutover, feature flags, manual rollback — are designed for stateless microservices, not for stateful kernels whose behavior depends on SM version, occupancy, and input shape.

`oxide-canary` adapts the canary deployment pattern to GPU kernels using a **ternary verdict model**:

- `Healthy (+1)`: The canary kernel is clearly better than the baseline.
- `Monitoring (0)`: The canary is similar to the baseline; keep observing.
- `Rollback (-1)`: The canary is worse; revert all traffic to baseline immediately.

This ternary model avoids the binary fallacy of "deploy or don't deploy." It introduces an explicit middle state for uncertainty, which is the appropriate stance when sample sizes are small and GPU noise is high.

The theoretical foundation combines **sequential hypothesis testing** with **progressive traffic shifting**. Rather than making a single go/no-go decision, `oxide-canary` moves through a staged rollout — 5%, 25%, 50%, 100% — and evaluates metrics at each stage. A `Rollback` verdict at any stage instantly resets traffic to 0%, limiting blast radius.

## How It Works

### CanaryRelease

A `CanaryRelease` defines a rollout between a `baseline_kernel` and a `canary_kernel`. It tracks:

- `traffic_percent`: Current canary traffic share.
- `stage_index`: Position in the fixed `ROLLOUT_STAGES` array.

Methods:

- `new(baseline, canary)` starts at 5% traffic.
- `advance_stage()` moves to the next rollout stage.
- `rollback()` resets traffic to 0%.
- `is_fully_rollouted()` checks for 100%.

### Metrics and Thresholds

The `Metrics` struct captures three primary signals:

- `latency_us`: Average latency per kernel invocation.
- `error_rate`: Fraction of invocations that failed.
- `throughput`: Ops per second.

`Thresholds` define how much better or worse the canary must be to exit the `Monitoring` state:

- `improvement = 0.10`: Canary error rate must be < 90% of baseline for `Healthy`.
- `degradation = 0.10`: Canary error rate must be > 110% of baseline for `Rollback`.

### CanaryManager

`CanaryManager` is the operational brain. It maintains rolling windows of baseline and canary metrics, aggregates them, and produces a verdict:

1. Aggregate recent baseline samples.
2. Aggregate recent canary samples.
3. Compare error rates using thresholds.
4. If both error rates are zero, use latency as a tie-breaker.
5. Return `Healthy`, `Monitoring`, or `Rollback`.

`tick()` runs one full evaluation cycle and automatically applies the verdict: advancing the stage on `Healthy`, rolling back on `Rollback`, and holding on `Monitoring`.

## Experiments

The test suite encodes the following claims:

```rust
#[test]
fn test_rollout_stages_progression() {
    // 5% → 25% → 50% → 100% progression is correct.
}

#[test]
fn test_healthy_verdict_when_canary_has_lower_error_rate() {
    // 1% canary error vs 5% baseline error → Healthy.
}

#[test]
fn test_tick_rolls_back_on_bad_verdict() {
    // A Rollback verdict resets traffic to 0% and routes to baseline.
}

#[test]
fn test_progressive_rollout_full_cycle() {
    // Four consecutive Healthy ticks reach full rollout.
}
```

A larger experiment: deploy two identical kernels except the canary uses a new shared-memory layout. Under a simulated load of 10,000 requests with 5% natural noise:

- Measure false-rollback rate when kernels are actually identical.
- Measure detection time when canary error rate is 10× baseline.
- Measure stage progression stability under bursty traffic patterns.
- Compare latency-only verdicts vs. error-rate-first verdicts.

## Applications

- **Kernel version rollout**: Deploy a new optimized matmul kernel to 5% of traffic, then expand based on metrics.
- **Driver compatibility validation**: Run a new CUDA driver build as a canary on a subset of nodes.
- **Construct upgrades**: Use with `oxide-constructs` to safely upgrade git-resident kernel versions.
- **A/B benchmarking**: Compare two kernel implementations under live traffic, with automatic winner selection.
- **Incident response**: Wire `Rollback` events to `oxide-circuit-breaker` to trip fleet-wide protection.

## Open Questions

1. **Metric dimensionality**: We use error rate and latency. Should GPU-specific signals (memory bandwidth, warp divergence, SM occupancy) be first-class inputs?
2. **Stage duration**: Stages currently advance on verdict, not time. Should there be a minimum observation window to prevent noisy early decisions?
3. **Multi-canary competition**: What happens when three candidate kernels compete against the same baseline simultaneously?
4. **Causal inference**: Correlation between canary traffic and errors is not causation. How do we distinguish kernel bugs from correlated infrastructure issues?

## Cross-Links

- [SuperInstance agent-knowledge / DEPLOYMENT-AND-OPERATIONS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/DEPLOYMENT-AND-OPERATIONS.md) — Fleet-wide deployment philosophy.
- [SuperInstance agent-knowledge / TERNARY-NUMBERS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TERNARY-NUMBERS.md) — The ternary verdict framework.
- [SuperInstance agent-knowledge / FAULT-TOLERANCE.md](https://github.com/SuperInstance/agent-knowledge/blob/main/FAULT-TOLERANCE.md) — How canary rollback fits into fault tolerance.
- [SuperInstance agent-knowledge / TESTING-AS-PROOF.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TESTING-AS-PROOF.md) — Why canary evidence must be testable.
- `oxide-circuit-breaker` — Receives rollback signals and protects the fleet.
- `oxide-fleet` — Routes traffic between baseline and canary agents.
- `oxide-constructs` — Supplies kernel versions to compare.

## Quick Start

```rust
use oxide_canary::{CanaryManager, CanaryRelease, Metrics, Verdict};

let release = CanaryRelease::new("v1.0", "v2.0");
let mut mgr = CanaryManager::new(release).with_window_size(3);

// Baseline samples: 5% error rate.
for _ in 0..3 {
    mgr.record_baseline(Metrics::new(100.0, 0.05, 1000.0));
}

// Canary samples: 1% error rate, faster.
for _ in 0..3 {
    mgr.record_canary(Metrics::new(80.0, 0.01, 1100.0));
}

let verdict = mgr.tick();
assert_eq!(verdict, Verdict::Healthy);
assert_eq!(mgr.release().traffic_percent, 25.0); // advanced from 5% → 25%
```

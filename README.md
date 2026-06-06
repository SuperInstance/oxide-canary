# oxide-canary

> Progressive, self-healing canary deployments for GPU kernel versions.

## Why Canary?

Deploying a new GPU kernel is a high-stakes operation. A single regression in a CUDA or ROCm kernel can bring down an inference pipeline, corrupt training checkpoints, or silently degrade model accuracy for hours before anyone notices. Traditional blue-green deployments work well for stateless web services, but they fall short when the unit of deployment is a compiled kernel whose behavior can only be validated under real-world load and hardware conditions.

**Canary releases** solve this by exposing the new kernel to a small, controlled slice of production traffic and observing how it behaves *in situ*. If metrics hold steady, you gradually increase traffic. If something goes wrong, you roll back before most users are affected. The catch is that canary logic itself needs to be robust, deterministic, and fast enough to act before human operators can page themselves.

`oxide-canary` is a small, zero-dependency Rust library that encodes exactly that logic: ternary health verdicts, configurable thresholds, progressive traffic stages, and automatic advance/rollback decisions. It is designed to drop into an inference server, a training orchestrator, or any system where GPU kernels are hot-swapped and need guard rails.

## The Ternary Verdict

Most monitoring systems force you into a binary world: green or red, pass or fail. In practice, canary metrics are noisy. A kernel might be slightly slower because a single SM is throttling, or because the batch size distribution shifted. Treating every blip as a rollback leads to deployment fatigue; ignoring it leads to incidents.

`oxide-canary` uses a **three-state verdict**:

| Verdict | Value | Meaning | Action |
|---------|-------|---------|--------|
| `Healthy` | `+1` | Canary is statistically better than baseline. | Advance to the next traffic stage. |
| `Monitoring` | `0` | Metrics are within tolerance. No strong signal either way. | Hold current traffic; keep sampling. |
| `Rollback` | `-1` | Canary is statistically worse than baseline. | Immediately reset traffic to 0 %. |

This ternary model eliminates alert fatigue while still ensuring that genuinely bad kernels are pulled from traffic within a single evaluation tick.

## How It Works

### 1. Define a Release

A `CanaryRelease` pairs a stable baseline kernel with a candidate canary kernel and tracks how much traffic the canary is receiving:

```rust
use oxide_canary::CanaryRelease;

let mut release = CanaryRelease::new("kernel-v1.4", "kernel-v1.5-candidate");
// Traffic starts at 5 % automatically.
assert_eq!(release.traffic_percent, 5.0);
```

Traffic moves through fixed stages: **5 % → 25 % → 50 % → 100 %**. Each stage acts as a gate; the canary must prove itself before it is allowed to see more load.

### 2. Collect Metrics

During each sampling window, record `Metrics` snapshots for both the baseline and the canary:

```rust
use oxide_canary::{CanaryManager, Metrics};

let release = CanaryRelease::new("stable", "candidate");
let mut mgr = CanaryManager::new(release);

// Record samples from the production stream.
mgr.record_baseline(Metrics::new(latency_us: 120.0, error_rate: 0.02, throughput: 4_000.0));
mgr.record_canary(Metrics::new(latency_us: 95.0, error_rate: 0.01, throughput: 4_200.0));
```

`Metrics` captures three signals:

- **`latency_us`** – average kernel execution latency in microseconds.
- **`error_rate`** – fraction of invocations that returned an error, timed out, or produced NaNs.
- **`throughput`** – operations per second sustained over the window.

The manager maintains a rolling history (default window: 10 samples) and automatically aggregates them into a single averaged snapshot when it is time to evaluate.

### 3. Evaluate

Calling `evaluate()` compares the aggregated canary snapshot against the baseline using configurable thresholds:

```rust
use oxide_canary::Verdict;

match mgr.evaluate() {
    Verdict::Healthy   => println!("Promote the canary."),
    Verdict::Monitoring => println!("Hold and observe."),
    Verdict::Rollback  => println!("Pull the canary immediately."),
}
```

The evaluator uses **error rate as the primary signal**, because correctness trumps performance. When both kernels report zero errors, latency becomes the tie-breaker. By default, a 10 % improvement is required for `Healthy`, and a 10 % degradation triggers `Rollback`. You can tune these boundaries:

```rust
use oxide_canary::Thresholds;

let mgr = CanaryManager::new(release)
    .with_thresholds(Thresholds {
        improvement: 0.05, // 5 % better → Healthy
        degradation: 0.15, // 15 % worse → Rollback
    })
    .with_window_size(20);
```

### 4. Automate with `tick()`

For hands-off operation, call `tick()` once per evaluation cycle. It runs `evaluate()` and mutates the release state automatically:

```rust
let verdict = mgr.tick();
// Healthy  → traffic advances to the next stage.
// Rollback → traffic drops to 0 %.
// Monitoring → nothing changes.
```

Because `tick()` is deterministic and side-effect-free except for the release state, it is safe to invoke from a cron loop, an async timer, or a streaming metrics pipeline.

## Routing

The manager exposes a simple `route()` method that returns `"canary"` or `"baseline"`. In a real system you would replace this with weighted random routing, sticky sessions, or shard-based splitting, but the built-in method gives you a deterministic starting point for integration tests and local development.

```rust
assert_eq!(mgr.route(), "canary"); // when traffic_percent > 0
```

After a rollback, `route()` flips back to `"baseline"` so that no further requests hit the bad kernel.

## Architecture at a Glance

```
┌─────────────────┐     ┌─────────────────┐
│  Baseline Kernel│     │  Canary Kernel  │
│   (stable)      │     │   (candidate)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         CanaryManager                   │
│  ┌─────────────┐  ┌─────────────┐      │
│  │rolling      │  │rolling      │      │
│  │baseline     │  │canary       │      │
│  │history      │  │history      │      │
│  └─────────────┘  └─────────────┘      │
│         │                │              │
│         └──────┬─────────┘              │
│                ▼                        │
│         aggregate()                     │
│                │                        │
│                ▼                        │
│         evaluate() ──► Verdict          │
│                │                        │
│         Healthy / Monitoring / Rollback │
└─────────────────────────────────────────┘
```

The entire state machine fits in a few hundred lines of safe Rust with no external dependencies, making it easy to audit, fuzz, and embed into larger binaries.

## Testing Philosophy

The crate ships with an exhaustive unit-test suite covering:

- Stage progression and boundary conditions (e.g., advancing past 100 % is a no-op).
- Rollback semantics and traffic reset.
- Routing decisions before and after rollback.
- Verdict logic for healthy, degraded, and neutral metric pairs.
- The full progressive rollout cycle from 5 % to 100 %.
- Latency tie-breaking when error rates are identically zero.
- Metrics aggregation correctness, including empty-sample handling.

Run the suite with:

```bash
cargo test
```

## Ecosystem

`oxide-canary` focuses on one thing: the state machine and evaluation logic for canary releases. It does not prescribe how you collect GPU metrics, how you load kernel binaries, or how you route RPC requests. For a higher-level orchestration layer that wraps kernel lifecycle management, traffic splitting, and fleet-wide rollouts, see [**SuperInstance**](https://github.com/SuperInstance/SuperInstance) — a complementary project that builds on the same principles at the cluster level.

## License

This project is licensed under the MIT License.

---

*Canary safely.*

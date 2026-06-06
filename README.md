# oxide-canary



## Why This Matters

# oxide-canary
Canary deployments for GPU kernel versions with ternary health verdict.
Verdicts: `+1` (healthy), `0` (monitoring), `-1` (rollback).

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum Verdict
pub struct Metrics
pub fn new
pub fn aggregate
pub struct CanaryRelease
pub fn new
pub fn advance_stage
pub fn rollback
pub fn is_fully_rollouted
pub struct Thresholds
pub struct CanaryManager
pub fn new
```

## Usage

```toml
[dependencies]
oxide-canary = "0.1.0"
```

```rust
use oxide_canary::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-canary.git
cd oxide-canary
cargo test    # 13 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 13 |
| Lines of Rust | 452 |
| Public API | 23 items |

## License

Apache-2.0

//! # oxide-canary
//!
//! Canary deployments for GPU kernel versions with ternary health verdict.
//!
//! Verdicts: `+1` (healthy), `0` (monitoring), `-1` (rollback).

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Health verdict returned after comparing canary vs baseline metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Canary is performing **better** than baseline → promote / increase traffic.
    Healthy = 1,
    /// Metrics are **similar** → hold current traffic, keep monitoring.
    Monitoring = 0,
    /// Canary is performing **worse** than baseline → automatic rollback.
    Rollback = -1,
}

/// Progressive rollout stages (percentage of traffic routed to canary).
pub const ROLLOUT_STAGES: &[f64] = &[5.0, 25.0, 50.0, 100.0];

/// Metric snapshot collected from a kernel over a sampling window.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Average latency in microseconds.
    pub latency_us: f64,
    /// Error rate in [0, 1].
    pub error_rate: f64,
    /// Throughput in ops/sec.
    pub throughput: f64,
}

impl Metrics {
    pub fn new(latency_us: f64, error_rate: f64, throughput: f64) -> Self {
        Self { latency_us, error_rate, throughput }
    }

    /// Aggregate multiple metric snapshots into a single averaged snapshot.
    pub fn aggregate(samples: &[Metrics]) -> Self {
        if samples.is_empty() {
            return Metrics::new(0.0, 0.0, 0.0);
        }
        let n = samples.len() as f64;
        Metrics::new(
            samples.iter().map(|s| s.latency_us).sum::<f64>() / n,
            samples.iter().map(|s| s.error_rate).sum::<f64>() / n,
            samples.iter().map(|s| s.throughput).sum::<f64>() / n,
        )
    }
}

/// A canary release definition.
#[derive(Debug, Clone)]
pub struct CanaryRelease {
    /// Identifier / version tag of the baseline (stable) kernel.
    pub baseline_kernel: String,
    /// Identifier / version tag of the canary (candidate) kernel.
    pub canary_kernel: String,
    /// Percentage of traffic currently routed to the canary kernel.
    pub traffic_percent: f64,
    /// Current rollout stage index into [`ROLLOUT_STAGES`].
    stage_index: usize,
}

impl CanaryRelease {
    /// Create a new canary release starting at the first rollout stage (5%).
    pub fn new(baseline: impl Into<String>, canary: impl Into<String>) -> Self {
        Self {
            baseline_kernel: baseline.into(),
            canary_kernel: canary.into(),
            traffic_percent: ROLLOUT_STAGES[0],
            stage_index: 0,
        }
    }

    /// Advance to the next rollout stage. Returns `false` if already at 100%.
    pub fn advance_stage(&mut self) -> bool {
        if self.stage_index + 1 < ROLLOUT_STAGES.len() {
            self.stage_index += 1;
            self.traffic_percent = ROLLOUT_STAGES[self.stage_index];
            true
        } else {
            false
        }
    }

    /// Roll traffic back to 0% (canary receives no traffic).
    pub fn rollback(&mut self) {
        self.traffic_percent = 0.0;
        self.stage_index = 0;
    }

    /// Whether the canary has reached full rollout.
    pub fn is_fully_rollouted(&self) -> bool {
        self.traffic_percent >= 100.0
    }
}

// ---------------------------------------------------------------------------
// Comparison thresholds
// ---------------------------------------------------------------------------

/// How much better/worse the canary needs to be to trigger a non-monitoring verdict.
#[derive(Debug, Clone)]
pub struct Thresholds {
    /// Fractional improvement needed for `Healthy` verdict.
    /// Canary error rate must be `(1 - improve)` × baseline error rate.
    pub improvement: f64,
    /// Fractional degradation tolerated before `Rollback` verdict.
    /// Canary error rate must exceed `(1 + degrade)` × baseline error rate.
    pub degradation: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            improvement: 0.10,   // 10% better → Healthy
            degradation: 0.10,   // 10% worse → Rollback
        }
    }
}

// ---------------------------------------------------------------------------
// CanaryManager
// ---------------------------------------------------------------------------

/// Manages one or more canary releases: routes traffic, compares metrics,
/// and returns a ternary verdict.
#[derive(Debug)]
pub struct CanaryManager {
    release: CanaryRelease,
    thresholds: Thresholds,
    baseline_history: VecDeque<Metrics>,
    canary_history: VecDeque<Metrics>,
    window_size: usize,
}

impl CanaryManager {
    /// Create a new manager for the given release.
    pub fn new(release: CanaryRelease) -> Self {
        Self {
            release,
            thresholds: Thresholds::default(),
            baseline_history: VecDeque::new(),
            canary_history: VecDeque::new(),
            window_size: 10,
        }
    }

    /// Override default thresholds.
    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Override the metric sample window size.
    pub fn with_window_size(mut self, n: usize) -> Self {
        self.window_size = n.max(1);
        self
    }

    /// Determine which kernel a request should be routed to.
    /// Returns `"canary"` or `"baseline"`.
    pub fn route(&self) -> &str {
        if self.release.traffic_percent > 0.0 {
            // Deterministic for simplicity: use percentage threshold.
            // In production this would use weighted random or sticky sessions.
            "canary"
        } else {
            "baseline"
        }
    }

    /// Record a metric sample for the baseline kernel.
    pub fn record_baseline(&mut self, metrics: Metrics) {
        self.baseline_history.push_back(metrics);
        if self.baseline_history.len() > self.window_size {
            self.baseline_history.pop_front();
        }
    }

    /// Record a metric sample for the canary kernel.
    pub fn record_canary(&mut self, metrics: Metrics) {
        self.canary_history.push_back(metrics);
        if self.canary_history.len() > self.window_size {
            self.canary_history.pop_front();
        }
    }

    /// Compare aggregated canary vs baseline metrics and produce a ternary verdict.
    ///
    /// Uses error rate as the primary signal. If error rates are similar,
    /// latency is used as a tie-breaker.
    pub fn evaluate(&self) -> Verdict {
        let baseline = self.aggregate_baseline();
        let canary = self.aggregate_canary();

        let b_err = baseline.error_rate;
        let c_err = canary.error_rate;

        if b_err == 0.0 && c_err == 0.0 {
            // Both perfect — use latency tie-breaker.
            return if canary.latency_us < baseline.latency_us * (1.0 - self.thresholds.improvement) {
                Verdict::Healthy
            } else if canary.latency_us > baseline.latency_us * (1.0 + self.thresholds.degradation) {
                Verdict::Rollback
            } else {
                Verdict::Monitoring
            };
        }

        if c_err < b_err * (1.0 - self.thresholds.improvement) {
            Verdict::Healthy
        } else if c_err > b_err * (1.0 + self.thresholds.degradation) {
            Verdict::Rollback
        } else {
            Verdict::Monitoring
        }
    }

    /// Run one evaluation cycle: evaluate, then advance/rollback accordingly.
    /// Returns the verdict and applies the appropriate action.
    pub fn tick(&mut self) -> Verdict {
        let verdict = self.evaluate();
        match verdict {
            Verdict::Healthy => {
                self.release.advance_stage();
            }
            Verdict::Rollback => {
                self.release.rollback();
            }
            Verdict::Monitoring => {
                // Hold current stage — no change.
            }
        }
        verdict
    }

    /// Aggregate recent baseline samples.
    pub fn aggregate_baseline(&self) -> Metrics {
        Metrics::aggregate(&self.baseline_history.iter().cloned().collect::<Vec<_>>())
    }

    /// Aggregate recent canary samples.
    pub fn aggregate_canary(&self) -> Metrics {
        Metrics::aggregate(&self.canary_history.iter().cloned().collect::<Vec<_>>())
    }

    /// Access the underlying release.
    pub fn release(&self) -> &CanaryRelease {
        &self.release
    }

    /// Mutable access to the release.
    pub fn release_mut(&mut self) -> &mut CanaryRelease {
        &mut self.release
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollout_stages_progression() {
        let mut release = CanaryRelease::new("v1.0", "v2.0");
        assert_eq!(release.traffic_percent, 5.0);

        assert!(release.advance_stage());
        assert_eq!(release.traffic_percent, 25.0);

        assert!(release.advance_stage());
        assert_eq!(release.traffic_percent, 50.0);

        assert!(release.advance_stage());
        assert_eq!(release.traffic_percent, 100.0);
        assert!(release.is_fully_rollouted());

        // Already at max — no further advancement.
        assert!(!release.advance_stage());
    }

    #[test]
    fn test_rollback_resets_traffic() {
        let mut release = CanaryRelease::new("v1.0", "v2.0");
        release.advance_stage();
        release.advance_stage();
        assert_eq!(release.traffic_percent, 50.0);

        release.rollback();
        assert_eq!(release.traffic_percent, 0.0);
        assert_eq!(release.stage_index, 0);
    }

    #[test]
    fn test_route_goes_canary_when_active() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mgr = CanaryManager::new(release);
        assert_eq!(mgr.route(), "canary");
    }

    #[test]
    fn test_route_goes_baseline_after_rollback() {
        let mut release = CanaryRelease::new("v1.0", "v2.0");
        release.rollback();
        let mgr = CanaryManager::new(release);
        assert_eq!(mgr.route(), "baseline");
    }

    #[test]
    fn test_healthy_verdict_when_canary_has_lower_error_rate() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(3);

        // Baseline: 5% error rate.
        for _ in 0..3 {
            mgr.record_baseline(Metrics::new(100.0, 0.05, 1000.0));
        }
        // Canary: 1% error rate — clearly better.
        for _ in 0..3 {
            mgr.record_canary(Metrics::new(80.0, 0.01, 1100.0));
        }

        assert_eq!(mgr.evaluate(), Verdict::Healthy);
    }

    #[test]
    fn test_rollback_verdict_when_canary_has_higher_error_rate() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(3);

        for _ in 0..3 {
            mgr.record_baseline(Metrics::new(100.0, 0.01, 1000.0));
        }
        // Canary: 20% error rate — clearly worse.
        for _ in 0..3 {
            mgr.record_canary(Metrics::new(200.0, 0.20, 800.0));
        }

        assert_eq!(mgr.evaluate(), Verdict::Rollback);
    }

    #[test]
    fn test_monitoring_verdict_when_metrics_are_similar() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(3);

        for _ in 0..3 {
            mgr.record_baseline(Metrics::new(100.0, 0.05, 1000.0));
        }
        // Canary: ~5.2% error rate — within 10% tolerance.
        for _ in 0..3 {
            mgr.record_canary(Metrics::new(105.0, 0.052, 990.0));
        }

        assert_eq!(mgr.evaluate(), Verdict::Monitoring);
    }

    #[test]
    fn test_tick_advances_on_healthy() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(3);

        for _ in 0..3 {
            mgr.record_baseline(Metrics::new(100.0, 0.10, 1000.0));
        }
        for _ in 0..3 {
            mgr.record_canary(Metrics::new(80.0, 0.01, 1100.0));
        }

        assert_eq!(mgr.tick(), Verdict::Healthy);
        assert_eq!(mgr.release().traffic_percent, 25.0); // advanced from 5% → 25%
    }

    #[test]
    fn test_tick_rolls_back_on_bad_verdict() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(3);

        for _ in 0..3 {
            mgr.record_baseline(Metrics::new(100.0, 0.01, 1000.0));
        }
        for _ in 0..3 {
            mgr.record_canary(Metrics::new(300.0, 0.50, 500.0));
        }

        assert_eq!(mgr.tick(), Verdict::Rollback);
        assert_eq!(mgr.release().traffic_percent, 0.0); // rolled back
        assert_eq!(mgr.route(), "baseline");
    }

    #[test]
    fn test_progressive_rollout_full_cycle() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(2);

        // Simulate 4 healthy ticks — should go 5% → 25% → 50% → 100%.
        for _ in 0..4 {
            mgr.record_baseline(Metrics::new(100.0, 0.10, 1000.0));
            mgr.record_baseline(Metrics::new(100.0, 0.10, 1000.0));
            mgr.record_canary(Metrics::new(80.0, 0.01, 1100.0));
            mgr.record_canary(Metrics::new(80.0, 0.01, 1100.0));
            mgr.tick();
        }

        assert!(mgr.release().is_fully_rollouted());
    }

    #[test]
    fn test_latency_tiebreaker_when_error_rates_zero() {
        let release = CanaryRelease::new("v1.0", "v2.0");
        let mut mgr = CanaryManager::new(release).with_window_size(2);

        // Both have 0% error rate, but canary is much faster.
        mgr.record_baseline(Metrics::new(100.0, 0.0, 1000.0));
        mgr.record_baseline(Metrics::new(100.0, 0.0, 1000.0));
        mgr.record_canary(Metrics::new(50.0, 0.0, 1100.0));
        mgr.record_canary(Metrics::new(50.0, 0.0, 1100.0));

        assert_eq!(mgr.evaluate(), Verdict::Healthy);
    }

    #[test]
    fn test_metrics_aggregation() {
        let samples = vec![
            Metrics::new(100.0, 0.05, 1000.0),
            Metrics::new(200.0, 0.15, 800.0),
        ];
        let agg = Metrics::aggregate(&samples);
        assert!((agg.latency_us - 150.0).abs() < f64::EPSILON);
        assert!((agg.error_rate - 0.10).abs() < f64::EPSILON);
        assert!((agg.throughput - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_aggregation_returns_zeros() {
        let agg = Metrics::aggregate(&[]);
        assert_eq!(agg.latency_us, 0.0);
        assert_eq!(agg.error_rate, 0.0);
        assert_eq!(agg.throughput, 0.0);
    }
}

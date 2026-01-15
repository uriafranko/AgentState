//! Metrics module for tracking latency and performance
//!
//! Provides timing utilities and aggregated statistics for monitoring
//! the performance of the AgentState engine.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Individual timing measurement
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    pub duration: Duration,
    pub timestamp: Instant,
}

/// Aggregated statistics for an operation
#[derive(Debug, Clone, Default)]
pub struct OperationStats {
    /// Number of calls
    pub count: u64,
    /// Total time spent
    pub total_time: Duration,
    /// Minimum latency observed
    pub min_time: Option<Duration>,
    /// Maximum latency observed
    pub max_time: Option<Duration>,
    /// Last recorded latency
    pub last_time: Option<Duration>,
}

impl OperationStats {
    /// Record a new timing
    pub fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total_time += duration;
        self.last_time = Some(duration);

        self.min_time = Some(match self.min_time {
            Some(min) => min.min(duration),
            None => duration,
        });

        self.max_time = Some(match self.max_time {
            Some(max) => max.max(duration),
            None => duration,
        });
    }

    /// Get average latency
    pub fn avg_time(&self) -> Option<Duration> {
        if self.count > 0 {
            Some(self.total_time / self.count as u32)
        } else {
            None
        }
    }

    /// Get average latency in milliseconds
    pub fn avg_ms(&self) -> Option<f64> {
        self.avg_time().map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get last latency in milliseconds
    pub fn last_ms(&self) -> Option<f64> {
        self.last_time.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get min latency in milliseconds
    pub fn min_ms(&self) -> Option<f64> {
        self.min_time.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get max latency in milliseconds
    pub fn max_ms(&self) -> Option<f64> {
        self.max_time.map(|d| d.as_secs_f64() * 1000.0)
    }
}

/// Operation types that can be tracked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Generating embeddings from text
    Embed,
    /// Classifying intent
    Classify,
    /// Saving to database
    DbSave,
    /// Searching database
    DbSearch,
    /// Full process() call
    Process,
    /// Full store() call
    Store,
    /// Full search() call
    Search,
}

impl Operation {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Embed => "embed",
            Operation::Classify => "classify",
            Operation::DbSave => "db_save",
            Operation::DbSearch => "db_search",
            Operation::Process => "process",
            Operation::Store => "store",
            Operation::Search => "search",
        }
    }
}

/// Metrics collector for tracking operation latencies
#[derive(Debug, Clone)]
pub struct Metrics {
    stats: Arc<Mutex<HashMap<Operation, OperationStats>>>,
    enabled: bool,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(HashMap::new())),
            enabled: true,
        }
    }

    /// Create a disabled metrics collector (no-op)
    pub fn disabled() -> Self {
        Self {
            stats: Arc::new(Mutex::new(HashMap::new())),
            enabled: false,
        }
    }

    /// Check if metrics collection is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a timing for an operation
    pub fn record(&self, op: Operation, duration: Duration) {
        if !self.enabled {
            return;
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.entry(op).or_default().record(duration);
        }
    }

    /// Get stats for a specific operation
    pub fn get_stats(&self, op: Operation) -> Option<OperationStats> {
        self.stats.lock().ok()?.get(&op).cloned()
    }

    /// Get stats for all operations
    pub fn get_all_stats(&self) -> HashMap<Operation, OperationStats> {
        self.stats.lock().ok().map(|s| s.clone()).unwrap_or_default()
    }

    /// Reset all statistics
    pub fn reset(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.clear();
        }
    }

    /// Start a timer that will automatically record when dropped
    pub fn start_timer(&self, op: Operation) -> Timer {
        Timer {
            op,
            start: Instant::now(),
            metrics: self.clone(),
        }
    }

    /// Format a summary of all metrics
    pub fn summary(&self) -> String {
        let stats = self.get_all_stats();
        if stats.is_empty() {
            return "No metrics recorded.".to_string();
        }

        let mut lines = vec!["Latency Metrics:".to_string()];
        lines.push(format!(
            "{:<12} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "Operation", "Count", "Last(ms)", "Avg(ms)", "Min(ms)", "Max(ms)"
        ));
        lines.push("-".repeat(62));

        // Sort by operation name for consistent output
        let mut ops: Vec<_> = stats.iter().collect();
        ops.sort_by_key(|(op, _)| op.name());

        for (op, stat) in ops {
            lines.push(format!(
                "{:<12} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                op.name(),
                stat.count,
                stat.last_ms().unwrap_or(0.0),
                stat.avg_ms().unwrap_or(0.0),
                stat.min_ms().unwrap_or(0.0),
                stat.max_ms().unwrap_or(0.0),
            ));
        }

        lines.join("\n")
    }
}

/// RAII timer that records duration when dropped
pub struct Timer {
    op: Operation,
    start: Instant,
    metrics: Metrics,
}

impl Timer {
    /// Get elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and record the duration
    /// Returns the elapsed duration
    pub fn stop(self) -> Duration {
        let duration = self.start.elapsed();
        self.metrics.record(self.op, duration);
        duration
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // Record on drop if not already stopped
        let duration = self.start.elapsed();
        self.metrics.record(self.op, duration);
    }
}

/// Convenience macro for timing a block of code
#[macro_export]
macro_rules! time_operation {
    ($metrics:expr, $op:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        $metrics.record($op, start.elapsed());
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_stats_recording() {
        let mut stats = OperationStats::default();

        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));
        stats.record(Duration::from_millis(15));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_time, Some(Duration::from_millis(10)));
        assert_eq!(stats.max_time, Some(Duration::from_millis(20)));
        assert_eq!(stats.last_time, Some(Duration::from_millis(15)));

        // Average should be (10+20+15)/3 = 15ms
        let avg = stats.avg_time().unwrap();
        assert_eq!(avg, Duration::from_millis(15));
    }

    #[test]
    fn test_metrics_collection() {
        let metrics = Metrics::new();

        metrics.record(Operation::Embed, Duration::from_millis(50));
        metrics.record(Operation::Embed, Duration::from_millis(60));
        metrics.record(Operation::Classify, Duration::from_millis(5));

        let embed_stats = metrics.get_stats(Operation::Embed).unwrap();
        assert_eq!(embed_stats.count, 2);
        assert_eq!(embed_stats.avg_ms().unwrap(), 55.0);

        let classify_stats = metrics.get_stats(Operation::Classify).unwrap();
        assert_eq!(classify_stats.count, 1);
    }

    #[test]
    fn test_disabled_metrics() {
        let metrics = Metrics::disabled();

        metrics.record(Operation::Embed, Duration::from_millis(50));

        assert!(metrics.get_stats(Operation::Embed).is_none());
    }

    #[test]
    fn test_timer() {
        let metrics = Metrics::new();

        {
            let _timer = metrics.start_timer(Operation::Process);
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = metrics.get_stats(Operation::Process).unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.last_ms().unwrap() >= 10.0);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new();

        metrics.record(Operation::Embed, Duration::from_millis(50));
        assert!(metrics.get_stats(Operation::Embed).is_some());

        metrics.reset();
        assert!(metrics.get_stats(Operation::Embed).is_none());
    }
}

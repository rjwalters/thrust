//! Trial execution and result tracking

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::space::ParameterValue;

/// Optimization objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Objective {
    /// Maximize this metric
    Maximize(String),
    /// Minimize this metric
    Minimize(String),
}

impl Objective {
    /// Get the metric name
    pub fn metric_name(&self) -> &str {
        match self {
            Objective::Maximize(name) | Objective::Minimize(name) => name,
        }
    }

    /// Check if this objective is maximization
    pub fn is_maximization(&self) -> bool {
        matches!(self, Objective::Maximize(_))
    }
}

/// A single trial configuration and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Trial ID
    pub id: usize,
    /// Hyperparameter configuration
    pub config: HashMap<String, ParameterValue>,
    /// Trial result (if completed)
    pub result: Option<TrialResult>,
}

/// Results from evaluating a trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Metric values (name -> value)
    pub metrics: HashMap<String, f64>,
    /// Training duration in seconds
    pub duration_secs: f64,
    /// Whether the trial succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl TrialResult {
    /// Create a successful result
    pub fn success(metrics: HashMap<String, f64>, duration_secs: f64) -> Self {
        Self { metrics, duration_secs, success: true, error: None }
    }

    /// Create a failed result
    pub fn failure(error: String) -> Self {
        Self { metrics: HashMap::new(), duration_secs: 0.0, success: false, error: Some(error) }
    }

    /// Get metric value
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }
}

impl Trial {
    /// Create a new trial
    pub fn new(id: usize, config: HashMap<String, ParameterValue>) -> Self {
        Self { id, config, result: None }
    }

    /// Check if trial is complete
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }

    /// Get result (if available)
    pub fn result(&self) -> Option<&TrialResult> {
        self.result.as_ref()
    }
}

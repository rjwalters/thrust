//! Parameter space definitions for hyperparameter optimization

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A parameter value (can be continuous or discrete)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    /// Continuous floating-point value
    Continuous(f64),
    /// Discrete integer value
    Discrete(i64),
}

impl ParameterValue {
    /// Get as f64 (for continuous params or discrete cast to float)
    pub fn as_f64(&self) -> f64 {
        match self {
            ParameterValue::Continuous(v) => *v,
            ParameterValue::Discrete(v) => *v as f64,
        }
    }

    /// Get as i64 (for discrete params)
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ParameterValue::Discrete(v) => Some(*v),
            _ => None,
        }
    }
}

/// Parameter definition in search space
#[derive(Debug, Clone)]
pub enum Parameter {
    /// Continuous parameter with min, max, and optional log scale
    Continuous { name: String, min: f64, max: f64, log_scale: bool },
    /// Discrete parameter with allowed values
    Discrete { name: String, values: Vec<i64> },
}

impl Parameter {
    /// Get parameter name
    pub fn name(&self) -> &str {
        match self {
            Parameter::Continuous { name, .. } => name,
            Parameter::Discrete { name, .. } => name,
        }
    }

    /// Sample a random value from this parameter's range
    pub fn sample(&self) -> ParameterValue {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        match self {
            Parameter::Continuous { min, max, log_scale, .. } => {
                let value = if *log_scale {
                    // Sample in log space
                    let log_min = min.ln();
                    let log_max = max.ln();
                    (rng.r#gen::<f64>() * (log_max - log_min) + log_min).exp()
                } else {
                    rng.r#gen::<f64>() * (max - min) + min
                };
                ParameterValue::Continuous(value)
            }
            Parameter::Discrete { values, .. } => {
                let idx = rng.gen_range(0..values.len());
                ParameterValue::Discrete(values[idx])
            }
        }
    }

    /// Normalize a value to [0, 1] range (for GP input)
    pub fn normalize(&self, value: &ParameterValue) -> f64 {
        match (self, value) {
            (Parameter::Continuous { min, max, log_scale, .. }, ParameterValue::Continuous(v)) => {
                if *log_scale {
                    let log_val = v.ln();
                    let log_min = min.ln();
                    let log_max = max.ln();
                    (log_val - log_min) / (log_max - log_min)
                } else {
                    (v - min) / (max - min)
                }
            }
            (Parameter::Discrete { values, .. }, ParameterValue::Discrete(v)) => {
                // Map to [0, 1] based on position in values list
                let idx = values.iter().position(|&x| x == *v).unwrap_or(0);
                idx as f64 / (values.len() - 1).max(1) as f64
            }
            _ => 0.5, // Mismatch, return middle value
        }
    }
}

/// Search space for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct SearchSpace {
    parameters: Vec<Parameter>,
}

impl SearchSpace {
    /// Create a new empty search space
    pub fn new() -> Self {
        Self { parameters: Vec::new() }
    }

    /// Add a continuous parameter
    pub fn add_continuous(
        mut self,
        name: impl Into<String>,
        min: f64,
        max: f64,
        log_scale: bool,
    ) -> Self {
        self.parameters
            .push(Parameter::Continuous { name: name.into(), min, max, log_scale });
        self
    }

    /// Add a discrete parameter
    pub fn add_discrete(mut self, name: impl Into<String>, values: Vec<i64>) -> Self {
        self.parameters.push(Parameter::Discrete { name: name.into(), values });
        self
    }

    /// Get number of parameters
    pub fn dim(&self) -> usize {
        self.parameters.len()
    }

    /// Sample a random configuration
    pub fn sample(&self) -> HashMap<String, ParameterValue> {
        self.parameters.iter().map(|p| (p.name().to_string(), p.sample())).collect()
    }

    /// Normalize a configuration to [0, 1]^d vector (for GP)
    pub fn normalize(&self, config: &HashMap<String, ParameterValue>) -> Vec<f64> {
        self.parameters
            .iter()
            .map(|p| config.get(p.name()).map(|v| p.normalize(v)).unwrap_or(0.5))
            .collect()
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.iter().find(|p| p.name() == name)
    }

    /// Get all parameters
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space() {
        let space = SearchSpace::new()
            .add_continuous("lr", 1e-4, 1e-3, true)
            .add_discrete("n_steps", vec![64, 128, 256]);

        assert_eq!(space.dim(), 2);

        let config = space.sample();
        assert_eq!(config.len(), 2);
        assert!(config.contains_key("lr"));
        assert!(config.contains_key("n_steps"));
    }

    #[test]
    fn test_normalization() {
        let space = SearchSpace::new()
            .add_continuous("x", 0.0, 10.0, false)
            .add_discrete("y", vec![1, 2, 3]);

        let mut config = HashMap::new();
        config.insert("x".to_string(), ParameterValue::Continuous(5.0));
        config.insert("y".to_string(), ParameterValue::Discrete(2));

        let normalized = space.normalize(&config);
        assert_eq!(normalized.len(), 2);
        assert!((normalized[0] - 0.5).abs() < 1e-6); // x=5.0 -> 0.5 in [0, 10]
        assert!((normalized[1] - 0.5).abs() < 1e-6); // y=2 -> middle value
    }
}

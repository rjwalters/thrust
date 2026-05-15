//! Weight export/import utilities for trained models

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use anyhow::Result;

use super::ExportedModel;

impl ExportedModel {
    /// Save model to JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model = serde_json::from_str(&contents)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use crate::inference::LayerWeights;

    fn create_test_model() -> ExportedModel {
        let feature_weights = LayerWeights::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2, 2);
        let policy_weights = LayerWeights::new(vec![1.0, -1.0, -1.0, 1.0], vec![0.0, 0.0], 2, 2);
        ExportedModel::new(2, 2, 2, feature_weights, policy_weights, None)
    }

    #[test]
    fn test_json_roundtrip() -> Result<()> {
        let model = create_test_model();
        let temp_file = NamedTempFile::new()?;

        model.save_json(temp_file.path())?;
        let loaded = ExportedModel::load_json(temp_file.path())?;

        assert_eq!(model.input_dim, loaded.input_dim);
        assert_eq!(model.output_dim, loaded.output_dim);
        assert_eq!(model.hidden_dim, loaded.hidden_dim);

        Ok(())
    }
}

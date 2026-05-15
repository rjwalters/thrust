//! Experience buffers and replay management
//!
//! This module handles storage and sampling of experience for training.

// Training-feature modules still owe public docs; tracked as the
// `--features training` follow-up to #33. Re-enable as `#![warn(missing_docs)]`
// once those items are documented.
#![allow(missing_docs)]

pub mod rollout;

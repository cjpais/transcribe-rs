use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

use super::engine::ModelVariant;
use super::model::MoonshineError;

/// KV Cache for Moonshine decoder.
///
/// Manages key-value cache state for both self-attention (decoder) and
/// cross-attention (encoder) across autoregressive decoding steps.
pub struct KVCache {
    cache: HashMap<String, ArrayD<f32>>,
    num_layers: usize,
}

impl KVCache {
    /// Create a new empty KV cache for the given model variant.
    pub fn new(variant: &ModelVariant) -> Self {
        let num_layers = variant.num_layers();
        let num_heads = variant.num_key_value_heads();
        let head_dim = variant.head_dim();

        let mut cache = HashMap::new();

        // Initialize empty cache tensors for all layers
        // Shape: (0, num_heads, 1, head_dim) - sequence length starts at 0
        for i in 0..num_layers {
            for attention_type in &["decoder", "encoder"] {
                for kv_type in &["key", "value"] {
                    let key = format!("past_key_values.{}.{}.{}", i, attention_type, kv_type);
                    let empty_tensor = ArrayD::<f32>::zeros(IxDyn(&[0, num_heads, 1, head_dim]));
                    cache.insert(key, empty_tensor);
                }
            }
        }

        Self { cache, num_layers }
    }

    /// Get all cache tensors as inputs for the decoder.
    pub fn get_inputs(&self) -> Vec<(String, ArrayD<f32>)> {
        let mut inputs = Vec::new();

        for i in 0..self.num_layers {
            for attention_type in &["decoder", "encoder"] {
                for kv_type in &["key", "value"] {
                    let key = format!("past_key_values.{}.{}.{}", i, attention_type, kv_type);
                    if let Some(tensor) = self.cache.get(&key) {
                        inputs.push((key, tensor.clone()));
                    }
                }
            }
        }

        inputs
    }

    /// Update cache from decoder outputs.
    ///
    /// # Cache Update Logic
    ///
    /// - First iteration (`use_cache_branch=false`): Update ALL caches (decoder + encoder)
    /// - Subsequent iterations (`use_cache_branch=true`): Update ONLY decoder caches
    ///
    /// The encoder cross-attention cache is computed once and reused for all subsequent tokens.
    pub fn update_from_outputs(
        &mut self,
        outputs: &ort::session::SessionOutputs,
        use_cache_branch: bool,
    ) -> Result<(), MoonshineError> {
        for i in 0..self.num_layers {
            for attention_type in &["decoder", "encoder"] {
                // Skip encoder cache updates after first iteration
                if use_cache_branch && *attention_type == "encoder" {
                    continue;
                }

                for kv_type in &["key", "value"] {
                    // Output names are "present.{i}.{type}.{kv}" (not "present_key_values")
                    let output_key = format!("present.{}.{}.{}", i, attention_type, kv_type);
                    let cache_key = format!("past_key_values.{}.{}.{}", i, attention_type, kv_type);

                    if let Some(output) = outputs.get(&output_key) {
                        let tensor = output
                            .try_extract_array::<f32>()
                            .map_err(|e| MoonshineError::Ort(e))?;
                        self.cache.insert(cache_key, tensor.to_owned());
                    }
                }
            }
        }

        Ok(())
    }
}

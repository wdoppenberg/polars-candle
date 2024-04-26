// Needed for the `polars_expr` macro
#![allow(clippy::unused_unit)]

use crate::candle_ext::utils::get_device;
use glowrs::{PoolingStrategy, SentenceTransformer};
use polars::error::PolarsResult;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

// Output functions
fn array_f32_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "array_float",
        DataType::List(Box::new(DataType::Float32)),
    ))
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceArg {
    Cpu,
    Gpu,
}

#[derive(Deserialize)]
pub struct EmbeddingKwargs {
    /// Huggingface model repository name
    pub model_repo: String,

    /// Pooling strategy
    pub pooling: PoolingStrategy,

    /// Normalize embeddings
    pub normalize: bool,

    /// Device (CPU, GPU)
    pub device: DeviceArg,
}

#[polars_expr(output_type_func = array_f32_output)]
pub fn embed_text(s: &[Series], kwargs: EmbeddingKwargs) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let name = ca.name();
    let len = ca.len();

    // Set device
    let is_cpu = match kwargs.device {
        DeviceArg::Cpu => true,
        DeviceArg::Gpu => false,
    };
    let device = get_device(is_cpu)
        .map_err(|e| polars_err!(ComputeError: "Could not select device: {}", e))?;

    let model = SentenceTransformer::from_repo_string(&kwargs.model_repo, &device)
        .map_err(|e| polars_err!(ComputeError: "Failed to load model: {}", e))?;

    let sentences: Vec<Option<&str>> = ca.into_iter().collect();

    // Separate sentences into those with Some and None
    let (some_sentences_idx, _): (Vec<_>, Vec<_>) = sentences
        .iter()
        .enumerate()
        .partition(|(_, sentence)| sentence.is_some());

    // Handle Some sentences:
    let some_sentences: Vec<&str> = some_sentences_idx
        .iter()
        .filter_map(|(_, sentence)| **sentence)
        .collect();

    // Embed the sentences
    let embeddings = model
        .encode_batch(some_sentences, kwargs.normalize, kwargs.pooling)
        .map_err(|e| polars_err!(ComputeError: "Encoding failed with error:\n{}", e))?;

    let (_, emb_dim) = embeddings.dims2().map_err(
        |e| polars_err!(ComputeError: "Getting embeddings dimensions failed with error:\n{}", e),
    )?;

    let emb_arr = embeddings.to_vec2::<f32>().map_err(
		|e| polars_err!(ComputeError: "Converting embeddings to Vec<Vec<f32>> failed with error:\n{}", e)
	)?;

    // Prepare final arr with None for missing entries
    let mut arr = vec![None; len];
    some_sentences_idx
        .into_iter()
        .zip(emb_arr.into_iter())
        .for_each(|((idx, _), emb)| {
            arr[idx] = Some(emb);
        });

    // TODO: How to directly create a array of fixed size list arrays?
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        &format!("{}_embedding", name),
        len,
        8,
        DataType::Float32,
    );

    arr.iter()
        .for_each(|embedding_option| match embedding_option {
            Some(embedding) => builder.append_slice(embedding.as_slice()),
            None => builder.append_null(),
        });

    let dtype = DataType::Array(Box::new(DataType::Float32), emb_dim);

    let out = builder.finish().into_series().cast(&dtype)?;

    Ok(out)
}

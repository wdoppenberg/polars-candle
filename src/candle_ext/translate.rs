use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::candle_ext::utils::DeviceArg;


#[derive(Deserialize)]
pub struct TranslateKwargs {
    /// Huggingface model repository name
    model_repo: String,
    
    /// What language to translate to.
    to_language: String,
    
    /// What language to translate from (optional).
    from_language: Option<String>,

    /// Device (CPU, GPU)
    device: DeviceArg
}


#[polars_expr(output_type=String)]
pub fn translate(s: &[Series], kwargs: TranslateKwargs) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let name = ca.name();
    let len = ca.len();
    
    
    todo!()
}
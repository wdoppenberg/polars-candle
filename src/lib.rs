use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult};

pub mod candle_ext;

#[pymodule]
#[pyo3(name = "polars_candle")]
fn polars_candle(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

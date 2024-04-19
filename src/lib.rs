use pyo3::{pymodule, PyResult, Python};
use pyo3::types::PyModule;

pub mod candle_ext;

#[pymodule]
#[pyo3(name = "polars_candle")]
fn polars_candle(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    Ok(())
}

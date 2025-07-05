use pyo3::prelude::*;
use temporal;

#[pymodule]
fn ts2graph_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Expose temporal graph functions
    temporal::register(m)?;
    Ok(())
}

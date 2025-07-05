mod temporal;
mod utils;

use pyo3::prelude::*;
use temporal::k_hop_graph;

#[pymodule]
fn ts2graph_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Expose temporal graph functions
    temporal::register(m)?;
    Ok(())
}

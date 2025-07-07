pub mod graph;
pub mod python_bindings;

// pub use graph::k_hop_graph_rs;
pub use python_bindings::register;

use pyo3::prelude::*;

#[pymodule]
fn ts2graph_rs(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    // Expose temporal graph functions
    python_bindings::register(py, &m)?;

    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Flavien Buron")?;

    Ok(())
}

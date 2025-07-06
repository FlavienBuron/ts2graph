pub mod graph;
pub mod python_bindings;
pub mod utils;

pub use graph::k_hop_graph_rs;
pub use python_bindings::register;

use pyo3::prelude::*;
pub use temporal::temporal::k_hop_graph_rs;

#[pymodule]
fn ts2graph_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Expose temporal graph functions
    temporal::register(m)?;

    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Flavien Buron");

    Ok(())
}

mod temporal;
mod utils;

use pyo3::prelude::*;
pub use temporal::k_hop_graph_rs;

#[cfg(feature = "python")]
#[pymodule]
fn ts2graph_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Expose temporal graph functions
    temporal::register(m)?;

    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Flavien Buron");

    Ok(())
}

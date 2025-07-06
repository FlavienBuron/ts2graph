use super::conversions::TensorConverter;
use crate::graph::k_hop_graph_rs;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyfunction]
#[pyo3(signature = (time_steps, num_nodes, k, bidirectional=true, decay_name=None))]
pub fn k_hop_graph(
    py: Python<'_>,
    time_steps: i64,
    num_nodes: i64,
    k: i64,
    bidirectional: bool,
    decay_name: Option<String>,
) -> PyResult<(PyObject, PyObject)> {
    let decay_name_str = decay_name.as_deref();
    let (edge_index, edge_weight) =
        k_hop_graph_rs(time_steps, num_nodes, k, bidirectional, decay_name_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Graph generation failed!: {e}")))?;

    // Convert to Numpy
    let edge_index_numpy = TensorConverter::tensor_to_numpy(py, &edge_index)?;
    let edge_weight_numpy = TensorConverter::tensor_to_numpy(py, &edge_weight)?;

    Ok((edge_index_numpy, edge_weight_numpy))
}

pub fn register(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(k_hop_graph, py)?)?;
    Ok(())
}

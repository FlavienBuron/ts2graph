use super::conversions::TensorConverter;
use crate::graph::temporal::k_hop_graph as k_hop_rs;
use crate::graph::temporal::recurrence_graph_rs;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyModule};

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
    let logging = PyModule::import(py, "logging")?;
    logging.call_method1("info", ("Test Rust from logging",))?;
    let decay_name_str = decay_name.as_deref();
    let (edge_index, edge_weight) =
        k_hop_rs(time_steps, num_nodes, k, bidirectional, decay_name_str)
            .map_err(|e| PyRuntimeError::new_err(format!("K-hop graph generation failed!: {e}")))?;

    // Convert to Numpy
    let edge_index_numpy = TensorConverter::tensor_to_numpy(py, &edge_index)?;
    let edge_weight_numpy = TensorConverter::tensor_to_numpy(py, &edge_weight)?;

    Ok((edge_index_numpy, edge_weight_numpy))
}

#[pyfunction]
#[pyo3(signature = (x, radius, embedding_dim=None, time_lag=1))]
pub fn recurrence_graph(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    radius: f64,
    embedding_dim: Option<i64>,
    time_lag: i64,
) -> PyResult<(PyObject, PyObject)> {
    let x_tensor = TensorConverter::from_torch_tensor(py, x)?;

    let (edge_index, edge_weight) = recurrence_graph_rs(&x_tensor, radius, embedding_dim, time_lag)
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Recurrence graph generation failed!: {e}"))
        })?;

    // Convert to Numpy
    let edge_index_numpy = TensorConverter::tensor_to_numpy(py, &edge_index)?;
    let edge_weight_numpy = TensorConverter::tensor_to_numpy(py, &edge_weight)?;

    Ok((edge_index_numpy, edge_weight_numpy))
}

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(k_hop_graph, py)?)?;
    m.add_function(wrap_pyfunction!(recurrence_graph, py)?)?;
    Ok(())
}

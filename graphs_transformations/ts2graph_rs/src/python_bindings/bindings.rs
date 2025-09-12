use super::conversions::TensorConverter;
use crate::graph::temporal::k_hop_graph as k_hop_rs;
use crate::graph::temporal::{recurrence_graph_rs, tsnet_vg};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
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
        k_hop_rs(time_steps, num_nodes, k, bidirectional, decay_name_str)
            .map_err(|e| PyRuntimeError::new_err(format!("K-hop graph generation failed!: {e}")))?;

    // Convert to Numpy
    let edge_index_numpy = TensorConverter::tensor_to_numpy(py, &edge_index)?;
    let edge_weight_numpy = TensorConverter::tensor_to_numpy(py, &edge_weight)?;

    Ok((edge_index_numpy, edge_weight_numpy))
}

#[pyfunction]
#[pyo3(signature = (x, radius, embedding_dim=None, time_lag=1, self_loop=false))]
pub fn recurrence_graph(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<f64>,
    radius: f64,
    embedding_dim: Option<usize>,
    time_lag: usize,
    self_loop: bool,
) -> PyResult<(PyObject, PyObject)> {
    // Extract slice and shape
    let array = x.as_array();
    if array.ndim() != 1 {
        return Err(PyRuntimeError::new_err(
            "input must be 1D time series array",
        ));
    }

    // Copy into Vec<f64>
    let data: Vec<f64> = array.iter().copied().collect();
    let net =
        recurrence_graph_rs(&data, radius, embedding_dim, time_lag, self_loop).map_err(|e| {
            PyRuntimeError::new_err(format!("Recurrence graph generation failed!: {e}"))
        })?;

    if net.edge_index.len() % 2 != 0 {
        return Err(PyRuntimeError::new_err(
            "edge_index length is not divisible by 2",
        ));
    }
    let num_edges = net.edge_index.len() / 2;

    let edges_index_array = Array2::from_shape_vec((2, num_edges), net.edge_index.clone())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape edge_index: {e}")))?;

    let edge_index_numpy: PyObject = edges_index_array
        .into_pyarray(py)
        .into_pyobject(py)?
        .unbind()
        .into();

    let edge_weight_numpy: PyObject = PyArray1::from_vec(py, net.edge_weight).into();

    Ok((edge_index_numpy, edge_weight_numpy))
}

#[pyfunction]
pub fn visibility_graph(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    method: &str,
    directed: bool,
    limit: Option<i64>,
    num_cores: Option<usize>,
) -> PyResult<(PyObject, PyObject)> {
    let x_tensor = TensorConverter::from_torch_tensor(py, x)?;

    let net = tsnet_vg(&x_tensor, method, directed, limit, num_cores).map_err(|e| {
        PyRuntimeError::new_err(format!("Recurrence graph generation failed!: {e}"))
    })?;

    // Convert to Numpy
    let edge_index_numpy = TensorConverter::tensor_to_numpy(py, &net.edge_index)?;
    let edge_weight_numpy = TensorConverter::tensor_to_numpy(py, &net.edge_weight)?;

    Ok((edge_index_numpy, edge_weight_numpy))
}

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(k_hop_graph, py)?)?;
    m.add_function(wrap_pyfunction!(recurrence_graph, py)?)?;
    m.add_function(wrap_pyfunction!(visibility_graph, py)?)?;
    Ok(())
}

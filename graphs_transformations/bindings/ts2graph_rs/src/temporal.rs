use numpy::{IntoPyArray, PyArray2, PyArrayDyn};
use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyValueError, prelude::*};
use tch::{Device, Kind, Tensor};

use crate::utils::{DecayFunction, GraphError};
///////////////////////////////////////////////////////////////////////////

pub fn k_hop_graph_rs(
    time_steps: i64,
    num_nodes: i64,
    k: i64,
    bidirectional: bool,
    decay_name: Option<&str>,
) -> Result<(Tensor, Tensor), String> {
    // Validate inputs
    if time_steps <= 0 {
        return Err(GraphError("time_steps must be positive".to_string()).into());
    }
    if num_nodes <= 0 {
        return Err(GraphError("num_nodes must be positive".to_string()).into());
    }
    if k < 0 {
        return Err(GraphError("k must be non-negative".to_string()).into());
    }

    let device = Device::Cpu;

    if k == 0 || time_steps < 2 {
        let edge_index = Tensor::zeros(&[2, 0], (Kind::Int64, device));
        let edge_weight = Tensor::ones(&[0], (Kind::Float, device));
        return Ok((edge_index, edge_weight));
    }

    let decay_fn = decay_name.as_deref().and_then(DecayFunction::from_str);
    let max_k = std::cmp::min(k, time_steps - 1);

    let estimated_edges = calculate_estimated_edges(time_steps, num_nodes, max_k, bidirectional);

    // Build graph structure
    let (src_indices, dst_indices, weights_values) =
        build_temporal_edges(time_steps, num_nodes, max_k, bidirectional, decay_fn);

    let src_tensor = Tensor::from_slice(&src_indices).to_device(device);
    let dst_tensor = Tensor::from_slice(&dst_indices).to_device(device);
    let weights_tensor = Tensor::from_slice(&weights_values).to_device(device);

    // Stack source and destination indices to create edge_index [2, num_edges]
    let edge_index = Tensor::stack(&[src_tensor, dst_tensor], 0);

    Ok((edge_index, weights_tensor))
}

///
fn calculate_estimated_edges(
    time_steps: i64,
    num_nodes: i64,
    max_k: i64,
    bidirectional: bool,
) -> usize {
    let multiplier = if bidirectional { 2 } else { 1 };
    (time_steps * max_k * num_nodes * multiplier) as usize
}

/// Build temporal edges with weights
fn build_temporal_edges(
    time_steps: i64,
    num_nodes: i64,
    max_k: i64,
    bidirectional: bool,
    decay_fn: Option<DecayFunction>,
) -> (Vec<i64>, Vec<i64>, Vec<f32>) {
    let estimated_edges = calculate_estimated_edges(time_steps, num_nodes, max_k, bidirectional);

    let mut src_indices = Vec::with_capacity(estimated_edges);
    let mut dst_indices = Vec::with_capacity(estimated_edges);
    let mut weights_values = Vec::with_capacity(estimated_edges);

    /// Create edges for each node
    for node in 0..num_nodes {
        let base = node * time_steps;

        for offset in 1..=max_k {
            let edge_len = time_steps - offset;
            let weight = match decay_fn {
                Some(df) => df.decay_value(offset, max_k),
                None => 1.0,
            };

            for i in 0..edge_len {
                let src = base + i;
                let dst = base + i + offset;
                edges.push(src);
                edges.push(dst);
                weights.push(weight);

                if bidirectional {
                    edges.push(dst);
                    edges.push(src);
                    weights.push(weight)
                }
            }
        }
    }

    (src_indices, dst_indices, weights_values)
}

///////////////////////////////////////////////////////////////////////////

mod python_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(signature = (time_steps, num_nodes, k, bidirectional=true, decay_name=None))]
    pub fn k_hop_graph(
        time_steps: i64,
        num_nodes: i64,
        k: i64,
        bidirectional: bool,
        decay_name: Option<String>,
    ) -> Result<(PyObject, PyObject), PyErr> {
        // Call the pure Rust function
        let decay_name_str = decay_name.as_deref();
        let (edge_index, edge_weight) =
            k_hop_graph_rs(time_steps, num_nodes, k, bidirectional, decay_name_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        Ok((edge_index.into_py_object(), edge_weight.into_py_object()))
    }

    pub fn register(m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(k_hop_graph, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::register;

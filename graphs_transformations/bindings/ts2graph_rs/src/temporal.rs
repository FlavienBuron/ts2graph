use numpy::{IntoPyArray, PyArray2};
use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyValueError, prelude::*};
use tch::{Device, Kind, Tensor};

use crate::utils::DecayFunction;

#[derive(Debug)]
pub struct GraphError(String);

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Graph error: {}", self.0)
    }
}

impl std::error::Error for GraphError {}

impl From<GraphError> for PyErr {
    fn from(value: GraphError) -> PyErr {
        PyValueError::new_err(value.0)
    }
}

#[pyfunction]
#[pyo3(signature = (x, num_nodes, k, bidirectional=true, decay_name=None))]
pub fn k_hop_graph(
    py: Python<'_>,
    x: &PyAny,
    num_nodes: i64,
    k: i64,
    bidirectional: bool,
    decay_name: Option<String>,
) -> Result<(Tensor, Tensor)> {
    // Validate inputs
    if num_nodes <= 0 {
        return Err(GraphError("num_nodes must be positive".to_string()).into());
    }
    if k < 0 {
        return Err(GraphError("k must be non-negative".to_string()).into());
    }

    let tensor = if let Ok(tensor) = x.extract::<Tensor>() {
        tensor
    } else {
        return Err(GraphError("Input must be a PyTorch tensor".to_string()).into());
    };

    let x = tensor.to(Device::Cpu);
    let size = x.size();

    if size.len() < 2 {
        return Err(GraphError("Input tensor must be at least 2D".to_string()).into());
    }

    let time_steps = size[0];

    if k == 0 || time_steps < 2 {
        let edge_index = Tensor::empty(&[2, 0], (Kind::Int64, Device::Cpu));
        let edge_weight = Tensor::ones(&[0], (Kind::Float, Device::Cpu));
        return Ok((edge_index, edge_weight));
    }

    let decay_fn = decay_name.as_deref().and_then(DecayFunction::from_str);
    let max_k = std::cmp::min(k, time_steps - 1);

    let estimated_edges =
        (time_steps * max_k * num_nodes * if bidirectional { 2 } else { 1 }) as usize;
    let mut edges = Vec::with_capacity(estimated_edges * 2);
    let mut weights = Vec::with_capacity(estimated_edges);

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

    let edge_index = Tensor::from_slice(&edges).view([-1, 2]).transpose(0, 1);
    let edge_weight = Tensor::from_slice(&weights);

    Ok((edge_index, edge_weight))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(k_hop_graph, m)?)?;
    Ok(())
}

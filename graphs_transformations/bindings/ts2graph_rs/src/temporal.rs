use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tch::{Device, Kind, Tensor};

use crate::utils::DecayFunction;

#[pyfunction]
pub fn k_hop_graph(
    x: Tensor,
    num_nodes: i64,
    k: i64,
    bidirectional: bool,
    decay_name: Option<String>,
) -> Result<(Tensor, Tensor)> {
    let x = x.to(Device::Cpu);
    let size = x.size();
    let time_steps = size[0];

    if k == 0 || time_steps < 2 {
        let edge_index = Tensor::empty(&[2, 0], (Kind::Int64, Device::Cpu));
        let edge_weight = Tensor::ones(&[0], (Kind::Float, Device::Cpu));
        return Ok((edge_index, edge_weight));
    }

    let decay_fn = decay_name.as_deref().and_then(DecayFunction::from_str);

    let max_k = std::cmp::min(k, time_steps - 1);
    let mut edges = Vec::with_capacity((time_steps * k * num_nodes * 2) as usize);
    let mut weights = Vec::with_capacity((time_steps * k * num_nodes * 2) as usize);

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

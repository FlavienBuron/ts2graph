use itertools::Itertools;
use tch::{Kind, Tensor};

fn embed_timeseries(x: &[f32], dim: usize, tau: usize) -> Result<Vec<Vec<f32>>, String> {
    let n = x.len();
    if n < (dim - 1) * tau + 1 {
        return Err("".into());
    }
}

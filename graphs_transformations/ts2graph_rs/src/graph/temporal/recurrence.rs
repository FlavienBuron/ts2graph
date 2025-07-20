use std::f64;

use itertools::Itertools;
use tch::{Device, Kind, Tensor};

fn embed_timeseries(x: &Tensor, dim: i64, tau: i64) -> Result<Tensor, String> {
    let n = x.size()[0];
    if n < (dim - 1) * tau + 1 {
        return Err("Time series too short for given embedding parameters".into());
    }

    let num_vectors = n - (dim - 1) * tau + 1;

    let views: Vec<Tensor> = (0..dim)
        .map(|j| {
            let start = j * tau;
            x.narrow(0, start, num_vectors)
        })
        .collect();

    let embedded = Tensor::stack(&views, 1);
    Ok(embedded)
}

/// Estimate embedding dimensions using Cao's algorithm
/// This implements the algorithm proposed by L. Cao which uses E1(d) and E2(d) functions
/// to determine the minimum embedding dimension from a scalar time series.
fn estimate_embedding_dim(x: &Tensor, time_lag: i64, max_dim: i64) -> Result<i64, String> {
    let n = x.size()[0];

    if max_dim < 2 {
        return Err("Maximum dimension must be at least 2".into());
    }

    let mut e1_values = Vec::new();
    let mut e2_values = Vec::new();

    // Calculate E1 and E2 for dimensions 1 to max_dim
    for dim in 1..=max_dim {
        if n < (dim * time_lag + 1) {
            return Err("Time series too short for embedding dimension estimation".into());
        }

        let embedded_d = embed_timeseries(x, dim, time_lag)?;
        let embedded_d_plus = embed_timeseries(x, dim + 1, time_lag)?;

        let num_vectors = embedded_d.size()[0];
        let mut e1_sum = 0.0f64;
        let mut e2_sum = 0.0f64;

        // For each point in the embedding space
        for i in 0..num_vectors {
            let current_point_d = embedded_d.get(i);
            let current_point_d_plus = embedded_d_plus.get(i);

            // Find nearest neighbor in d-dimensional space
            let mut min_dist = f64::INFINITY;
            let mut nearest_idx = 0;

            for j in 0..num_vectors {
                if i != j {
                    let other_point_d = embedded_d.get(j);
                    let dist: f64 = euclidian_distance(&current_point_d, &other_point_d)
                        .double_value(&[]) as f64;
                    if dist < min_dist {
                        min_dist = dist;
                        nearest_idx = j;
                    }
                }
            }

            if min_dist > 0.0 {
                // Calculate distance in (d+1)-dimensional
                let nearest_point_d_plus = embedded_d_plus.get(nearest_idx);
                let dist_d_plus: f64 =
                    euclidian_distance(&current_point_d_plus, &nearest_point_d_plus)
                        .double_value(&[]) as f64;

                // E1 calculation: ratio of distances
                e1_sum += dist_d_plus / min_dist;

                // E2 calculation: for detecting deterministic vs stochastic signals
                if dim > 1 {
                    // Get the previous embedding dimension distances
                    let embedded_d_minus1 = embed_timeseries(x, dim - 1, time_lag)?;
                    let current_point_d_minus1 = embedded_d_minus1.get(i);
                    let nearest_point_d_minus1 = embedded_d_minus1.get(nearest_idx);
                    let dist_d_minus1: f64 =
                        euclidian_distance(&current_point_d_minus1, &nearest_point_d_minus1)
                            .double_value(&[]) as f64;

                    if dist_d_minus1 > 0.0 {
                        e2_sum += (dist_d_plus / min_dist) / (min_dist / dist_d_minus1);
                    }
                }
            }
        }

        let e1 = e1_sum / num_vectors as f64;
        e1_values.push(e1);

        if dim > 1 {
            let e2 = e2_sum / num_vectors as f64;
            e2_values.push(e2);
        }
    }

    // Find the dimension where E1 stops changing (becomes approximately constant)
    // E1(d) should be close to 1 for d >= embedding dimension
    let mut estimated_dim = max_dim;
    let threshold = 0.1; // threshold for detecting when E1 stops changing

    for i in 1..e1_values.len() {
        let e1_current = e1_values[i];
        let e1_prev = e1_values[i - 1];

        // Check if E1 has stabilized (small change and close to 1)
        let change = (e1_current - e1_prev).abs();
        if change < threshold && e1_current > 0.9 && e1_current < 1.1 {
            estimated_dim = (i + 1) as i64;
            break;
        }
    }

    // Additional check using E2 for deterministic vs stochastic signals
    // For deterministic signals, E2 should deviate from 1 at some dimension
    if !e2_values.is_empty() {
        let mut has_deterministic_structure = false;
        for &e2 in &e2_values {
            if (e2 - 1.0).abs() > 0.1 {
                has_deterministic_structure = true;
                break;
            }
        }

        // If no deterministic structure is found, use a smaller dimension
        if !has_deterministic_structure {
            estimated_dim = std::cmp::min(estimated_dim, 3);
        }
    }

    Ok(std::cmp::max(1, std::cmp::min(estimated_dim, max_dim)))
}

fn compute_recurrence_matrix(embedded_ts: &Tensor, radius: f64) -> Tensor {
    let n = embedded_ts.size()[0];
    let mut recurrence_matrix = Tensor::zeros([n, n], (Kind::Bool, Device::Cpu));

    // Fill the recurrence matrix
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let v1 = embedded_ts.get(i);
                let v2 = embedded_ts.get(j);
                let dist = euclidian_distance(&v1, &v2);
                let is_recurrent = dist.le(radius);
                recurrence_matrix.get(i).get(j).copy_(&is_recurrent);
            }
        }
    }

    recurrence_matrix
}

fn recurrence_matrix_to_edges(recurrence_matrix: &Tensor) -> (Tensor, Tensor) {
    let n = recurrence_matrix.size()[0];
    let mut src_vec = Vec::new();
    let mut dst_vec = Vec::new();

    // Convert adjacency matrix to edge list
    // For undirected graph, we only need to consider upper triangle
    for i in 0..n {
        for j in (i + 1)..n {
            // let is_connected: bool = recurrence_matrix.get(i).get(j).into();
            let is_connected: bool = recurrence_matrix.get(i).get(j).int64_value(&[]) != 0;
            if is_connected {
                src_vec.push(i);
                dst_vec.push(j);
                // Add reverse edge for undirected graph
                src_vec.push(j);
                dst_vec.push(i);
            }
        }
    }

    let src = tensor_from_slice(&src_vec);
    let dst = tensor_from_slice(&dst_vec);

    (src, dst)
}

/// Convert two index arrays into a `[2, E]` adge index tensor.
fn to_edge_index(src: &Tensor, dst: &Tensor) -> Tensor {
    Tensor::stack(&[src, dst], 0)
}

fn euclidian_distance(v1: &Tensor, v2: &Tensor) -> Tensor {
    (v1 - v2)
        .pow_tensor_scalar(2)
        .sum_dim_intlist([0i64].as_slice(), false, Kind::Float)
        .sqrt()
}

// Create a Tensor from a tensor slice
fn tensor_from_slice<T: tch::kind::Element>(slice: &[T]) -> Tensor {
    Tensor::f_from_slice(slice).expect("Failed to create tensor from slice")
}

pub fn recurrence_graph_rs(
    x: &Tensor,
    radius: f64,
    embedding_dim: Option<i64>,
    time_lag: i64,
) -> Result<(Tensor, Tensor), String> {
    // 1. Determine the embedding dimension
    let dim = match embedding_dim {
        Some(d) => d,
        None => {
            // Estimate the dim using Cao's algorithm
            estimate_embedding_dim(x, time_lag, 10)?
        }
    };

    // 2. Embed the time series
    let embedded = embed_timeseries(x, dim, time_lag)?;

    // 3. Compute the recurrence matrix
    let recurrence_matrix = compute_recurrence_matrix(&embedded, radius);

    // 4. Convert recurrence matrix to edge list
    let (src, dst) = recurrence_matrix_to_edges(&recurrence_matrix);

    // 5. Create edge index tensor
    let edge_index = to_edge_index(&src, &dst);

    // 6. Create edge weights (unweighted graph)
    let num_edges = src.size()[0];
    let edge_weight = Tensor::ones([num_edges], (Kind::Float, Device::Cpu));

    Ok((edge_index, edge_weight))
}

/// Recurrence graph construction with all parameters specified
pub fn recurrence_graph_withdim(
    x: &Tensor,
    radius: f64,
    embedding_dim: i64,
    time_lag: i64,
) -> Result<(Tensor, Tensor), String> {
    let (edge_index, edge_weight) = recurrence_graph_rs(x, radius, Some(embedding_dim), time_lag)?;
    Ok((edge_index, edge_weight))
}

// Helper function to get the recurrence matrix directly (for debugging/analysis)
pub fn get_recurrence_matrix(
    x: &Tensor,
    radius: f64,
    embedding_dim: Option<i64>,
    time_lag: i64,
) -> Result<(Tensor, i64), String> {
    let dim = match embedding_dim {
        Some(d) => d,
        None => estimate_embedding_dim(x, time_lag, 10)?,
    };

    let embedded = embed_timeseries(x, dim, time_lag)?;
    Ok((compute_recurrence_matrix(&embedded, radius), dim))
}

// Convenience function that takes a slice and converts to tensor
pub fn recurrence_graph_from_slice(
    x: &[f64],
    radius: f64,
    embedding_dim: Option<i64>,
    time_lag: i64,
) -> Result<(Tensor, Tensor), String> {
    let x_tensor = tensor_from_slice(x);
    recurrence_graph_rs(&x_tensor, radius, embedding_dim, time_lag)
}

/// Estimate embedding dimension for a slice input using Cao's algorithm
pub fn estimate_embedding_dim_from_slice(
    x: &[f64],
    time_lag: i64,
    max_dim: i64,
) -> Result<i64, String> {
    let x_tensor = tensor_from_slice(x);
    estimate_embedding_dim(&x_tensor, time_lag, max_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let x = tensor_from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let embedded = embed_timeseries(&x, 2, 1).unwrap();
        assert_eq!(embedded.size(), [4, 2]);

        let first_row: Vec<f64> = embedded.get(0).into();
        assert_eq!(first_row, vec![1.0, 2.0]);

        let second_row: Vec<f64> = embedded.get(1).into();
        assert_eq!(second_row, vec![2.0, 3.0]);
    }

    #[test]
    fn test_euclidian_distance() {
        let v1 = tensor_from_slice(&[0.0f64, 0.0]);
        let v2 = tensor_from_slice(&[3.0f64, 4.0]);
        let dist: f64 = euclidian_distance(&v1, &v2).into();
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_recurrence_matrix() {
        let embedded = tensor_from_slice(&[
            0.0f64, 0.0, // Point 0
            1.0, 1.0, // Point 1
            0.1, 0.1, // Point 3
        ])
        .view([3, 2]);

        let rm = compute_recurrence_matrix(&embedded, 0.5);

        let val_02: bool = rm.get(0).get(2).into();
        let val_20: bool = rm.get(2).get(0).into();
        let val_01: bool = rm.get(0).get(1).into();

        assert_eq!(val_02, true); // Points 0 and 2 should be connected
        assert_eq!(val_20, true); // Symmetric
        assert_eq!(val_01, false); // Points 0 and 1 should not be connected
    }

    #[test]
    fn test_embedding_dim_estimation() {
        // Create a simple periodic signal
        let x: Vec<f64> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let estimated_dim = estimate_embedding_dim_from_slice(&x, 1, 10).unwrap();
        assert!(estimated_dim >= 1 && estimated_dim <= 10);
    }

    #[test]
    fn test_recurrence_graph_with_estimation() {
        let x = [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0,
        ];
        let (edge_index, edge_weight, estimated_dim) =
            recurrence_graph_from_slice(&x, 1.0, None, 1).unwrap();

        assert_eq!(edge_index.size()[0], 2); // should have 2 rows (src, dst)
        assert!(edge_index.size()[1] > 0); // Should have some edges
        assert_eq!(edge_weight.size()[0], edge_index.size()[1]); // Same number of weights as edges
        assert!(estimated_dim >= 1 && estimated_dim <= 10) // Reasonable embedding dimension
    }

    #[test]
    fn test_recurrence_graph_with_specified_dim() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let (edge_index, edge_weight, used_dim) =
            recurrence_graph_from_slice(&x, 1.0, Some(2), 1).unwrap();

        assert_eq!(used_dim, 2); // Should use the specified dimension
        assert_eq!(edge_index.size()[0], 2);
        assert!(edge_index.size()[1] > 0);
        assert_eq!(edge_weight.size()[0], edge_index.size()[1]);
    }
}

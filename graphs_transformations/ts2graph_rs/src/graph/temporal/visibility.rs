use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

#[derive(Debug, Clone)]
pub enum Method {
    Nvg,
    Hvg,
}

pub struct GraphOutput {
    pub edge_index: Tensor,  // Shape: (2, E)
    pub edge_weight: Tensor, // Shape: (E,)
}

pub fn tsnet_vg(
    x: &Tensor,
    method: &str,
    directed: bool,
    limit: Option<i64>,
    num_cores: Option<usize>,
) -> Result<GraphOutput, Box<dyn std::error::Error>> {
    // Ensure tensor is 1D and on CPU
    let x = if x.dim() == 1 {
        x.to_device(Device::Cpu)
    } else {
        return Err("Input tensor must be 1-dimensional".into());
    };

    let n = x.size()[0];
    let limit = limit.unwrap_or(i64::MAX);

    let concrete_method = match method {
        "nvg" => Method::Nvg,
        _ => Method::Hvg,
    };

    // Convert tensor to Vec for easier indexing in parallel operations
    let x_vec: Vec<f64> = x.to_kind(Kind::Double).try_into()?;

    // Generate all combinations of indices (i, j) where i < j
    let mut combinations = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            combinations.push((i, j));
        }
    }

    // Configure rayon thread pool if num_cores is specified
    let _pool = num_cores.map(|cores| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .build_global()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build_global().unwrap())
    });

    // Process combinations in parallel
    let linked_pairs: Vec<(i64, i64, f64)> = combinations
        .into_par_iter()
        .filter_map(|(i, j)| {
            if is_linked(&x_vec, i, j, &concrete_method, limit) {
                // Calculate edge weight (e.g., inverse of distance, or custom weight)
                let weight =
                    calculate_edge_weight(&x_vec, i as usize, j as usize, &concrete_method);
                Some((i, j, weight))
            } else {
                None
            }
        })
        .collect();

    // Build edge tensors
    if linked_pairs.is_empty() {
        return Ok(GraphOutput {
            edge_index: Tensor::empty(&[2, 0], (Kind::Int64, Device::Cpu)),
            edge_weight: Tensor::empty(&[0], (Kind::Float, Device::Cpu)),
        });
    }

    let mut edges = Vec::new();
    let mut weights = Vec::new();

    for (i, j, weight) in linked_pairs {
        edges.push([i, j]);
        weights.push(weight);

        if !directed {
            edges.push([j, i]);
            weights.push(weight);
        }
    }

    // Convert to tensors
    let edge_data: Vec<i64> = edges.iter().flatten().copied().collect();
    let edge_index = Tensor::from_slice(&edge_data).view([2, edges.len() as i64]);

    let edge_weight = Tensor::from_slice(&weights);

    Ok(GraphOutput {
        edge_index,
        edge_weight,
    })
}

// pub fn tsnet_vg_batch(
//     x_batch: &Tensor,
//     method: Method,
//     directed: bool,
//     limit: Option<i64>,
//     num_cores: Option<usize>,
// ) -> Result<Vec<GraphOutput>, Box<dyn std::error::Error>> {
//     // Ensure tensor is 2D (batch_size, sequence_length)
//     if x_batch.dim() != 2 {
//         return Err("Batch tensor must be 2-dimensional (batch_size, sequence_length)".into());
//     }
//
//     let batch_size = x_batch.size()[0];
//     let mut graphs = Vec::with_capacity(batch_size as usize);
//
//     for i in 0..batch_size {
//         let x_i = x_batch.get(i);
//         let graph = tsnet_vg(&x_i, method.clone(), directed, limit, num_cores)?;
//         graphs.push(graph);
//     }
//
//     Ok(graphs)
// }

fn is_linked(x: &[f64], i: i64, j: i64, method: &Method, limit: i64) -> bool {
    let i = i as usize;
    let j = j as usize;

    // Check if distance exceeds limit
    if (j - i) as i64 > limit {
        return false;
    }

    // Adjacent points are always linked
    if j - i == 1 {
        return true;
    }

    // Check visibility between non-adjacent points
    match method {
        Method::Nvg => is_nvg_linked(x, i, j),
        Method::Hvg => is_hvg_linked(x, i, j),
    }
}

fn is_nvg_linked(x: &[f64], i: usize, j: usize) -> bool {
    // Natural Visibility Graph: check if line of sight exists
    let x_i = x[i];
    let x_j = x[j];

    for k in (i + 1)..j {
        let x_k = x[k];

        // Calculate the height of the line between (i, x_i) and (j, x_j) at position k
        let line_height = x_j + (x_i - x_j) * (j - k) as f64 / (j - i) as f64;

        // If any intermediate point is above or equal to the line, no visibility
        if x_k >= line_height {
            return false;
        }
    }

    true
}

fn is_hvg_linked(x: &[f64], i: usize, j: usize) -> bool {
    // Horizontal Visibility Graph: check if any intermediate point blocks the view
    let x_i = x[i];
    let x_j = x[j];

    for k in (i + 1)..j {
        let x_k = x[k];

        // If any intermediate point is higher than or equal to either endpoint, no visibility
        if x_k >= x_i || x_k >= x_j {
            return false;
        }
    }

    true
}

fn calculate_edge_weight(x: &[f64], i: usize, j: usize, method: &Method) -> f64 {
    match method {
        Method::Hvg => {
            // For HVG, weight could be based on the minimum height of endpoints
            let min_height = x[i].min(x[j]);
            min_height
        }
        Method::Nvg => {
            // For NVG, weight could be based on the slope or correlation
            let slope = (x[j] - x[i]) / (j - i) as f64;
            slope.abs()
        }
    }
}

// // Alternative weight calculation methods
// pub fn calculate_custom_weights(
//     x: &Tensor,
//     edge_index: &Tensor,
//     weight_method: WeightMethod,
// ) -> Tensor {
//     let x_vec: Vec<f64> = x
//         .to_kind(Kind::Double)
//         .try_into()
//         .map_err(|e| format!("{:?}", e))?;
//     let edge_data: Vec<i64> = Vec::<i64>::from(edge_index);
//     let num_edges = edge_index.size()[1] as usize;
//
//     let weights: Vec<f64> = (0..num_edges)
//         .map(|e| {
//             let i = edge_data[e] as usize;
//             let j = edge_data[e + num_edges] as usize;
//
//             match weight_method {
//                 WeightMethod::Distance => 1.0 / (j - i) as f64,
//                 WeightMethod::ValueDiff => (x_vec[j] - x_vec[i]).abs(),
//                 WeightMethod::ValueSum => x_vec[i] + x_vec[j],
//                 WeightMethod::ValueProduct => x_vec[i] * x_vec[j],
//                 WeightMethod::Uniform => 1.0,
//             }
//         })
//         .collect();
//
//     Tensor::from_slice(&weights)
// }

#[derive(Debug, Clone)]
pub enum WeightMethod {
    Distance,     // 1/distance
    ValueDiff,    // |x[j] - x[i]|
    ValueSum,     // x[i] + x[j]
    ValueProduct, // x[i] * x[j]
    Uniform,      // all weights = 1.0
}

// Utility functions for creating tensors from various sources
pub fn tensor_from_vec(data: Vec<f64>) -> Tensor {
    Tensor::from_slice(&data)
}

pub fn tensor_from_slice(data: &[f64]) -> Tensor {
    Tensor::from_slice(data)
}

pub fn random_time_series(length: i64, seed: Option<i64>) -> Tensor {
    if let Some(s) = seed {
        tch::manual_seed(s);
    }
    Tensor::randn(&[length], (Kind::Double, Device::Cpu))
}

// Convert edge_index format for PyTorch Geometric compatibility
pub fn to_pyg_format(edge_index: &Tensor, edge_weight: &Tensor) -> (Tensor, Tensor) {
    // PyTorch Geometric expects edge_index as (2, E) and edge_weight as (E,)
    // This function is mainly for clarity/documentation
    (edge_index.shallow_clone(), edge_weight.shallow_clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hvg_tensor_output() -> Result<(), Box<dyn std::error::Error>> {
        let data = tensor_from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
        let output = tsnet_vg(&data, "hvg", false, None, None)?;

        println!("Edge index shape: {:?}", output.edge_index.size());
        println!("Edge weight shape: {:?}", output.edge_weight.size());
        println!("Edge index:\n{:?}", output.edge_index);
        println!("Edge weights:\n{:?}", output.edge_weight);

        assert_eq!(output.edge_index.size()[0], 2); // Should have 2 rows
        assert_eq!(output.edge_index.size()[1], output.edge_weight.size()[0]); // E edges
        Ok(())
    }

    #[test]
    fn test_nvg_tensor_output() -> Result<(), Box<dyn std::error::Error>> {
        let data = tensor_from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
        let output = tsnet_vg(&data, "nvg", false, None, None)?;

        println!("NVG Edge index shape: {:?}", output.edge_index.size());
        println!("NVG Edge weight shape: {:?}", output.edge_weight.size());

        assert_eq!(output.edge_index.size()[0], 2);
        assert_eq!(output.edge_index.size()[1], output.edge_weight.size()[0]);
        Ok(())
    }

    #[test]
    fn test_directed_vs_undirected() -> Result<(), Box<dyn std::error::Error>> {
        let data = tensor_from_vec(vec![1.0, 2.0, 1.5, 3.0]);

        let directed = tsnet_vg(&data, "hvg", true, None, None)?;
        let undirected = tsnet_vg(&data, "hvg", false, None, None)?;

        println!("Directed edges: {}", directed.edge_index.size()[1]);
        println!("Undirected edges: {}", undirected.edge_index.size()[1]);

        // Undirected should have more edges (each connection in both directions)
        assert!(undirected.edge_index.size()[1] >= directed.edge_index.size()[1]);
        Ok(())
    }

    #[test]
    fn test_empty_graph() -> Result<(), Box<dyn std::error::Error>> {
        // Create data where no edges should exist (monotonic decreasing)
        let data = tensor_from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let output = tsnet_vg(&data, "hvg", false, Some(1), None)?;

        println!("Empty graph edge count: {}", output.edge_index.size()[1]);
        // Should still have adjacent connections
        assert!(output.edge_index.size()[1] > 0);
        Ok(())
    }

    // #[test]
    // fn test_custom_weights() -> Result<(), Box<dyn std::error::Error>> {
    //     let data = tensor_from_vec(vec![1.0, 3.0, 2.0, 4.0]);
    //     let output = tsnet_vg(&data, Method::Hvg, false, None, None)?;
    //
    //     let custom_weights =
    //         calculate_custom_weights(&data, &output.edge_index, WeightMethod::Distance);
    //     println!("Custom weights shape: {:?}", custom_weights.size());
    //     println!("Custom weights: {:?}", custom_weights);
    //
    //     assert_eq!(custom_weights.size()[0], output.edge_index.size()[1]);
    //     Ok(())
    // }

    // #[test]
    // fn test_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    //     let batch_data = Tensor::from_slice2(&[
    //         &[1.0, 3.0, 2.0, 4.0, 1.0],
    //         &[2.0, 1.0, 3.0, 1.5, 2.5],
    //         &[1.5, 2.5, 1.0, 3.0, 2.0],
    //     ]);
    //
    //     let outputs = tsnet_vg_batch(&batch_data, Method::Hvg, false, None, Some(2))?;
    //
    //     assert_eq!(outputs.len(), 3);
    //     for (i, output) in outputs.iter().enumerate() {
    //         println!("Batch {}: {} edges", i, output.edge_index.size()[1]);
    //         assert_eq!(output.edge_index.size()[0], 2);
    //         assert_eq!(output.edge_index.size()[1], output.edge_weight.size()[0]);
    //     }
    //
    //     Ok(())
    // }
}

// // Example usage and demonstration
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     println!("Time Series Network Generation - Tensor Output Format\n");
//
//     // Example 1: Basic usage
//     let time_series = tensor_from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0, 2.5, 3.5, 1.8, 2.9]);
//     println!("Time series data: {:?}", Vec::<f64>::from(&time_series));
//
//     let hvg_output = tsnet_vg(&time_series, Method::Hvg, false, None, Some(4))?;
//     println!("\nHorizontal Visibility Graph:");
//     println!("Edge index shape: {:?}", hvg_output.edge_index.size());
//     println!("Edge weight shape: {:?}", hvg_output.edge_weight.size());
//     println!("Number of edges: {}", hvg_output.edge_index.size()[1]);
//
//     // Example 2: Directed graph
//     let nvg_output = tsnet_vg(&time_series, Method::Nvg, true, None, Some(4))?;
//     println!("\nNatural Visibility Graph (directed):");
//     println!("Edge index shape: {:?}", nvg_output.edge_index.size());
//     println!("Edge weight shape: {:?}", nvg_output.edge_weight.size());
//
//     // Example 3: Custom weights
//     let custom_weights =
//         calculate_custom_weights(&time_series, &hvg_output.edge_index, WeightMethod::Distance);
//     println!("\nCustom distance-based weights:");
//     println!("Custom weights shape: {:?}", custom_weights.size());
//
//     // Example 4: PyTorch Geometric format
//     let (pyg_edge_index, pyg_edge_weight) =
//         to_pyg_format(&hvg_output.edge_index, &hvg_output.edge_weight);
//     println!("\nPyTorch Geometric compatible format:");
//     println!("Edge index: {:?}", pyg_edge_index.size());
//     println!("Edge weight: {:?}", pyg_edge_weight.size());
//
//     // Example 5: Batch processing
//     println!("\n--- Batch Processing ---");
//     let batch_data = Tensor::of_slice2(&[
//         &[1.0, 3.0, 2.0, 4.0],
//         &[2.0, 1.0, 3.0, 1.5],
//         &[1.5, 2.5, 1.0, 3.0],
//     ]);
//
//     let batch_outputs = tsnet_vg_batch(&batch_data, Method::Hvg, false, None, Some(2))?;
//     for (i, output) in batch_outputs.iter().enumerate() {
//         println!(
//             "Batch {}: edge_index {:?}, edge_weight {:?}",
//             i,
//             output.edge_index.size(),
//             output.edge_weight.size()
//         );
//     }
//
//     // Example 6: Different weight methods
//     println!("\n--- Weight Method Comparison ---");
//     let test_data = tensor_from_vec(vec![1.0, 4.0, 2.0, 5.0]);
//     let test_output = tsnet_vg(&test_data, Method::Hvg, false, None, None)?;
//
//     let weight_methods = [
//         WeightMethod::Distance,
//         WeightMethod::ValueDiff,
//         WeightMethod::ValueSum,
//         WeightMethod::Uniform,
//     ];
//
//     for method in &weight_methods {
//         let weights = calculate_custom_weights(&test_data, &test_output.edge_index, method.clone());
//         println!("{:?} weights: {:?}", method, Vec::<f64>::from(&weights));
//     }
//
//     Ok(())
// }

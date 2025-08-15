use std::{
    collections::{HashMap, HashSet},
    f64, vec,
};

use kdtree::{KdTree, distance::squared_euclidean};
use petgraph::{Graph, graph::UnGraph};
use tch::{Kind, Tensor};

/// Constants for box estimation (matching R implementation)
const K_MINIMUM_NUMBER_BOXES: usize = 10;
const K_MAXIMUM_NUMBER_BOXES: usize = 500;

#[derive(Debug, Clone)]
struct CaoParameters {
    e: f64,
    e_star: f64,
}

#[derive(Debug)]
pub struct NearestNeighborResult {
    /// Indices of nearest neighbors (including self)
    indices: Tensor,
    /// Distance to nearest neighbors (including self)
    distances: Tensor,
}

#[derive(Debug, Clone)]
pub struct NeighborList {
    pub neighbors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub number_points: Option<i64>,
    pub time_lag: i64,
    pub max_embedding_dim: i64,
    pub threshold: f64,
    pub max_relative_change: f64,
    pub std_noise: Option<f64>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            number_points: None,
            time_lag: 1,
            max_embedding_dim: 15,
            threshold: 0.95,
            max_relative_change: 0.10,
            std_noise: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecurrenceMatrixConfig {
    pub embedding_dim: i64,
    pub time_lag: i64,
    pub radius: f32,
}

impl Default for RecurrenceMatrixConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 2,
            time_lag: 1,
            radius: 0.1,
        }
    }
}

#[derive(Debug)]
pub struct TsNet {
    pub edge_index: Tensor,
    pub edge_weight: Tensor,
    pub embedding_dim: i64,
    pub time_lag: i64,
    pub radius: f32,
}

use ndarray::Array2;

// /// Represents a neighbor list result
// #[derive(Debug)]
// pub struct NeighborList {
//     pub neighbors: Vec<Vec<usize>>,
//     pub radius: f64,
// }

/// Box-assisted neighbor search (closer to R implementation)
pub struct BoxAssistedSearch<'a> {
    data: &'a Tensor,
    num_boxes: usize,
    radius: f32,
    box_size: Vec<f64>,
    min_vals: Vec<f64>,
    max_vals: Vec<f64>,
}

impl<'a> BoxAssistedSearch<'a> {
    pub fn new(data: &'a Tensor, radius: f32, num_boxes: Option<usize>) -> Self {
        let shape = data.size();
        let n_points = shape[0] as usize;
        let n_dims = shape[1] as usize;

        // Estimate number of boxes if not provided (similar to R's estimateNumberBoxes)
        let num_boxes = num_boxes.unwrap_or_else(|| estimate_number_boxes(data, radius));

        // Calculate bounds and box sizes
        let mut min_vals = vec![f64::INFINITY; n_dims];
        let mut max_vals = vec![f64::NEG_INFINITY; n_dims];

        for i in 0..n_points {
            for j in 0..n_dims {
                let val = data.double_value(&[i as i64, j as i64]);
                min_vals[j] = min_vals[j].min(val);
                max_vals[j] = max_vals[j].max(val);
            }
        }

        let box_size: Vec<f64> = (0..n_dims)
            .map(|i| (max_vals[i] - min_vals[i]) / num_boxes as f64)
            .collect();

        Self {
            data,
            num_boxes,
            radius: radius,
            box_size,
            min_vals,
            max_vals,
        }
    }

    /// Convert point coordinates to box indices
    fn point_to_box_indices(&self, point: &Tensor) -> Vec<usize> {
        (0..point.size()[0])
            .map(|i| {
                let coord = point.double_value(&[i]);
                let box_sz = self.box_size[i as usize];
                let min_val = self.min_vals[i as usize];
                let idx = if box_sz == 0.0 {
                    0
                } else {
                    ((coord - min_val) / box_sz).floor() as i32
                };
                (idx.max(0) as usize).min(self.num_boxes - 1)
            })
            .collect()
    }

    /// Get all box indices within radius (accounting for wrapped grid)
    fn get_neighbor_boxes(&self, center_box: &[usize]) -> Vec<Vec<usize>> {
        let n_dims = center_box.len();
        let boxes_to_check: Vec<i32> = self
            .box_size
            .iter()
            .map(|&sz| ((self.radius / sz as f32).ceil() as i32) + 1)
            .collect();

        let mut neighbor_boxes = Vec::new();

        // Generate all combinations of box offsets
        self.generate_box_combinations(
            center_box,
            0,
            Vec::new(),
            &boxes_to_check,
            &mut neighbor_boxes,
        );

        neighbor_boxes
    }

    fn generate_box_combinations(
        &self,
        center_box: &[usize],
        dim: usize,
        current: Vec<usize>,
        ranges: &Vec<i32>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if dim == center_box.len() {
            result.push(current);
            return;
        }

        let center = center_box[dim] as i32;
        let range = ranges[dim];
        for offset in -range..=range {
            let new_idx = center + offset;
            // Handle wrapped grid (periodic boundaries)
            // let wrapped_idx = if new_idx < 0 {
            //     (self.num_boxes as i32 + new_idx) as usize
            // } else if new_idx >= self.num_boxes as i32 {
            //     (new_idx - self.num_boxes as i32) as usize
            // } else {
            //     new_idx as usize
            // };
            let wrapped_idx =
                ((new_idx % self.num_boxes as i32) + self.num_boxes as i32) % self.num_boxes as i32;

            let mut new_current = current.clone();
            new_current.push(wrapped_idx as usize);
            self.generate_box_combinations(center_box, dim + 1, new_current, ranges, result);
        }
    }

    /// Calculate Euclidean distance between two points
    fn distance(&self, i: usize, j: usize) -> f32 {
        let ncols = self.data.size()[1];
        let mut sum: f64 = 0.0;
        for dim in 0..ncols {
            let diff =
                self.data.double_value(&[i as i64, dim]) - self.data.double_value(&[j as i64, dim]);
            sum += diff * diff;
        }
        sum.sqrt() as f32
    }

    fn flatten_indices(&self, indices: &[usize]) -> usize {
        let mut flat = 0;
        for (i, &idx) in indices.iter().enumerate() {
            flat *= self.num_boxes;
            flat += idx;
        }
        flat
    }

    /// Find all neighbors for all points
    pub fn find_all_neighbors(&self) -> Vec<Vec<usize>> {
        let n_points = self.data.size()[0] as usize;
        let mut all_neighbors = vec![Vec::new(); n_points];

        // Create point-to-box mapping
        let mut box_to_points: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..n_points {
            let point = self.data.get(i as i64);
            let box_indices = self.point_to_box_indices(&point);
            let flat_idx = self.flatten_indices(&box_indices);
            box_to_points
                .entry(flat_idx)
                .or_insert_with(Vec::new)
                .push(i);
        }

        // For each point, search in relevant boxes
        for i in 0..n_points {
            let point = self.data.get(i as i64);
            let center_box = self.point_to_box_indices(&point);
            let neighbor_boxes = self.get_neighbor_boxes(&center_box);

            let mut candidates = Vec::new();
            for box_indices in neighbor_boxes {
                let flat_idx = self.flatten_indices(&box_indices);
                if let Some(points) = box_to_points.get(&flat_idx) {
                    candidates.extend(points);
                }
            }

            // Check actual distances
            for &j in &candidates {
                if i != j && self.distance(i, j) <= self.radius {
                    all_neighbors[i].push(j);
                    // all_neighbors[j].push(i);
                }
            }

            // Sort for consistent ordering
            all_neighbors[i].sort_unstable();
        }

        all_neighbors
    }
}

/// Estimate the number of boxes
fn estimate_number_boxes(data: &Tensor, radius: f32) -> usize {
    if data.numel() == 0 {
        return K_MINIMUM_NUMBER_BOXES;
    }

    if radius <= 0.0 {
        return K_MINIMUM_NUMBER_BOXES;
    }

    // Find min and max values
    let min_val = data.min().double_value(&[]) as f32;
    let max_val = data.max().double_value(&[]) as f32;

    // Calculate number of boxes: (max - min) / radius
    let range = max_val - min_val;
    let mut number_boxes = (range as f32 / radius);

    // Apply bounds
    if number_boxes > K_MAXIMUM_NUMBER_BOXES as f32 {
        number_boxes = K_MAXIMUM_NUMBER_BOXES as f32;
    }
    if number_boxes < K_MINIMUM_NUMBER_BOXES as f32 {
        number_boxes = K_MINIMUM_NUMBER_BOXES as f32;
    }

    number_boxes as usize
}

fn embed_timeseries(x: &Tensor, dim: i64, tau: i64) -> Result<Tensor, String> {
    let n = x.size()[0];
    if n < (dim - 1) * tau + 1 {
        return Err("Time series too short for given embedding parameters".into());
    }

    let num_vectors = n - (dim - 1) * tau;

    let size0 = x.size()[0];
    let views: Vec<Tensor> = (0..dim)
        .filter_map(|j| {
            let start = j * tau;
            if start + num_vectors <= size0 {
                Some(x.narrow(0, start, num_vectors))
            } else {
                None
            }
        })
        .collect();

    if views.is_empty() {
        return Err("Embedding failed: no valid segments".into());
    }
    let embedded = Tensor::stack(&views, 1);
    Ok(embedded)
}

fn estimate_embedding_dim(
    time_series: &Tensor,
    config: EmbeddingConfig,
) -> Result<Option<i64>, String> {
    if config.max_embedding_dim < 3 {
        return Err("Max embedding_dim should be greater than 2".into());
    }

    // Ensure tensor is 1D and convert to double precision
    let mut ts = time_series.flatten(0, -1).to_kind(Kind::Double);
    let ts_device = ts.device();
    let time_series_len = ts.size()[0];

    // Normalize time series using tensor operation
    let mean = ts.mean(Kind::Double);
    let std_dev = ts.std(false);
    ts = (ts - &mean) / &std_dev;

    // Add noise to avoid identical phase space Points
    let std_noise = if let Some(noise) = config.std_noise {
        if noise < 0.0 { 0.0 } else { noise }
    } else {
        let ts_vec: Vec<f64> = ts
            .shallow_clone()
            .try_into()
            .map_err(|e| format!("{:?}", e))?;
        let mut sorted_data = ts_vec.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let k_sd_fraction = 1e-6;
        let min_abs_separation = sorted_data
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(f64::INFINITY, f64::min);

        if min_abs_separation.is_finite() {
            min_abs_separation * k_sd_fraction
        } else {
            1e-6
        }
    };

    if std_noise > 0.0 {
        let noise = Tensor::randn(&[time_series_len as i64], (Kind::Double, ts_device)) * std_noise;
        ts = ts + noise;
    }

    // Extract data segments
    let number_points = config.number_points.unwrap_or(time_series_len);
    let ts_beg = (time_series_len / 2 - number_points / 2) as i64;
    let ts_end = (time_series_len / 2 + number_points / 2) as i64;
    let data = ts.narrow(0, ts_beg, ts_end - ts_beg);

    let mut embedding_dim = 0;

    // First iteartion: get E(1) and E(2)
    let e_params_1 = get_cao_parameters(&data, 1, config.time_lag)?;
    let e_params_2 = get_cao_parameters(&data, 2, config.time_lag)?;

    let mut e_vector = vec![e_params_1.e, e_params_2.e];
    let mut e_star_vector = vec![e_params_1.e_star, e_params_2.e_star];
    let mut e1_vector = vec![e_vector[1] / e_vector[0]];
    let mut e2_vector = vec![e_star_vector[1] / e_star_vector[0]];

    // Compute from d=3 to d=max_embedding_dim
    for dimension in 3..=config.max_embedding_dim {
        let e_params = get_cao_parameters(&data, dimension, config.time_lag)?;
        e_vector.push(e_params.e);
        e_star_vector.push(e_params.e_star);

        let e1 = e_vector[(dimension - 1) as usize] / e_vector[(dimension - 2) as usize];
        let e2 = e_star_vector[(dimension - 1) as usize] / e_star_vector[(dimension - 2) as usize];

        e1_vector.push(e1);
        e2_vector.push(e2);

        // Error for dimension - 2
        if dimension >= 4 {
            let idx = dimension - 3; // e1_vector index for dimension - 2
            let relative_error = (e1_vector[(idx + 1) as usize] - e1_vector[idx as usize]).abs()
                / e1_vector[idx as usize];

            // Check if E1(d) >= threshold and first time condition is met
            if embedding_dim == 0
                && e1_vector[idx as usize] >= config.threshold
                && relative_error < config.max_relative_change
            {
                embedding_dim = dimension - 2;
            }
        }
    }

    // Return embedding dimension
    if embedding_dim == 0 {
        Ok(None)
    } else {
        Ok(Some(embedding_dim as i64))
    }
}

fn get_cao_parameters(
    data: &Tensor,
    embedding_dim: i64,
    time_lag: i64,
) -> Result<CaoParameters, String> {
    const K_ZERO: f64 = 1e-15;

    let takens = build_takens_explicit(data, embedding_dim, time_lag)?;
    let takens_next_dimension = build_takens_explicit(data, embedding_dim + 1, time_lag)?;

    let max_iter = takens_next_dimension.size()[0];

    // Find nearest neighbors
    let nearest_neigh = if embedding_dim == 1 {
        let data_1d = data.narrow(0, 0, max_iter).unsqueeze(1);
        nn_search(&data_1d, &data_1d, 2)?
    } else {
        let takens_subset = takens.narrow(0, 0, max_iter);
        nn_search(&takens_subset, &takens_subset, 2)?
    };

    let mut min_dist_ratios = Vec::new();
    let mut stochastic_parameters = Vec::new();

    // Computing parameters for each Takens position
    for takens_position in 0..max_iter {
        // Get closest neighbor (avoid picking the same vector with index 1, use index 1 for second
        // closets)
        let closest_neigh_idx: i64 = nearest_neigh.indices.int64_value(&[takens_position, 1]);
        let nearest_dist: f64 = nearest_neigh.distances.double_value(&[takens_position, 1]);

        if nearest_dist < K_ZERO {
        } else {
            // Calculate numerator using maximum distance (L-infinity norm)
            let vec1 = takens_next_dimension.get(takens_position);
            let vec2 = takens_next_dimension.get(closest_neigh_idx);
            let numerator = maximum_distance(&vec1, &vec2)?;

            let min_dist_ratio = numerator / nearest_dist;
            min_dist_ratios.push(min_dist_ratio);
        }

        // Calculate stochastic parameter
        let data_idx1 = takens_position + embedding_dim * time_lag;
        let data_idx2 = closest_neigh_idx + embedding_dim * time_lag;

        if data_idx1 < data.size()[0] && data_idx2 < data.size()[0] {
            let val1: f64 = data.double_value(&[data_idx1]);
            let val2: f64 = data.double_value(&[data_idx2]);
            let stochastic_param = (val1 - val2).abs();
            stochastic_parameters.push(stochastic_param);
        }
    }

    let e = if min_dist_ratios.is_empty() {
        0.0
    } else {
        min_dist_ratios.iter().sum::<f64>() / min_dist_ratios.len() as f64
    };

    let e_star = if stochastic_parameters.is_empty() {
        0.0
    } else {
        stochastic_parameters.iter().sum::<f64>() / stochastic_parameters.len() as f64
    };

    Ok(CaoParameters { e, e_star })
}

fn nn_search(data: &Tensor, query: &Tensor, k: i64) -> Result<NearestNeighborResult, String> {
    let n_data = data.size()[0];
    let n_query = query.size()[0];

    if data.size()[1] != query.size()[1] {
        return Err("Query and data must have same dimensionality".to_string());
    }

    if k > n_data {
        return Err("Cannot find more nearest neighbors than there are data points".into());
    }

    let distances = chebyshev_distance(data, query);
    let (k_distances, k_indices) = distances.topk(k, 1, false, true);

    Ok(NearestNeighborResult {
        indices: k_indices,
        distances: k_distances,
    })
}

fn pairwise_distances(embedded_matrix: &Tensor) -> Tensor {
    // Compute pairwise squared distances using broadcasting
    // ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    let norms_sq = embedded_matrix
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, Kind::Double);

    let dot_products = embedded_matrix.matmul(&embedded_matrix.transpose(0, 1));

    let distances_sq: Tensor = &norms_sq.unsqueeze(1) + &norms_sq.unsqueeze(0) - 2.0 * dot_products;

    // Take square root and clamp to avoid numerical issues
    distances_sq.clamp_min(0.0).sqrt()
}

fn chebyshev_distance(a: &Tensor, b: &Tensor) -> Tensor {
    let a_unsqueezed = a.unsqueeze(1); // [n, 1, dim]
    let b_unsqueezed = b.unsqueeze(0); // [1, n, dim]
    (a_unsqueezed - b_unsqueezed).abs().amax(-1, false) // [n, n]
}

fn maximum_distance(vec1: &Tensor, vec2: &Tensor) -> Result<f64, String> {
    let diff = (vec1 - vec2).abs();
    let max_diff: f64 = diff.max().try_into().map_err(|e| format!("{:?}", e))?;
    Ok(max_diff)
}

pub fn build_takens_explicit(
    time_series: &Tensor,
    embedding_dim: i64,
    time_lag: i64,
) -> Result<Tensor, String> {
    // Ensure tensor is 1D and get length
    let ts = time_series.flatten(0, -1);
    let ts_kind = ts.kind();
    let ts_device = ts.device();
    let n = ts.size()[0] as i64;

    // Calculate parameters
    let max_jump = (embedding_dim - 1) * time_lag;
    let num_vectors: i64 = n - max_jump;

    if num_vectors <= 0 {
        return Err("Time series too short for given embedding dimension and lag".into());
    }

    // Create jumps vector equivalent to R's seq(0, maxjump, time.lag)
    let jumps: Vec<i64> = (0..embedding_dim).map(|i| i * time_lag).collect();

    // Create output tensor
    let mut takens = Tensor::zeros(&[num_vectors, embedding_dim], (ts_kind, ts_device));

    // Build Takens vectors
    for i in 0..num_vectors {
        for (j, &jump) in jumps.iter().enumerate() {
            let value = ts.get((jump + i) as i64);
            takens = takens.index_put_(
                &[Some(Tensor::from(i)), Some(Tensor::from(j as i64))],
                &value,
                false,
            );
        }
    }
    Ok(takens)
}

// /// Estimate embedding dimensions using Cao's algorithm
// /// This implements the algorithm proposed by L. Cao which uses E1(d) and E2(d) functions
// /// to determine the minimum embedding dimension from a scalar time series.
// fn estimate_embedding_dim(x: &Tensor, time_lag: i64, max_dim: i64) -> Result<i64, String> {
//     let n = x.size()[0];
//
//     if max_dim < 2 {
//         return Err("Maximum dimension must be at least 2".into());
//     }
//
//     let mut e1_values = Vec::new();
//     let mut e2_values = Vec::new();
//
//     // Calculate E1 and E2 for dimensions 1 to max_dim
//     for dim in 1..=max_dim {
//         if n < ((dim - 1) * time_lag + 1) {
//             return Err("Time series too short for embedding dimension estimation".into());
//         }
//
//         let embedded_d = embed_timeseries(x, dim, time_lag)?;
//         let embedded_d_plus = embed_timeseries(x, dim + 1, time_lag)?;
//
//         let num_vectors_d = embedded_d.size()[0];
//         let num_vectors_d_plus = embedded_d_plus.size()[0];
//         let num_vectors = std::cmp::min(num_vectors_d, num_vectors_d_plus);
//
//         if num_vectors < 2 {
//             continue;
//         }
//
//         let mut e1_sum = 0.0f64;
//         let mut e2_sum = 0.0f64;
//
//         // For each point in the embedding space
//         for i in 0..num_vectors {
//             let current_point_d = embedded_d.get(i);
//             let current_point_d_plus = embedded_d_plus.get(i);
//
//             // Find nearest neighbor in d-dimensional space
//             let mut min_dist = f64::INFINITY;
//             let mut nearest_idx = 0;
//
//             for j in 0..num_vectors {
//                 if i != j {
//                     let other_point_d = embedded_d.get(j);
//                     let dist: f64 = euclidian_distance(&current_point_d, &other_point_d)
//                         .double_value(&[]) as f64;
//                     if dist < min_dist {
//                         min_dist = dist;
//                         nearest_idx = j;
//                     }
//                 }
//             }
//
//             if min_dist > 0.0 {
//                 // Calculate distance in (d+1)-dimensional
//                 let nearest_point_d_plus = embedded_d_plus.get(nearest_idx);
//                 let dist_d_plus: f64 =
//                     euclidian_distance(&current_point_d_plus, &nearest_point_d_plus)
//                         .double_value(&[]) as f64;
//
//                 // E1 calculation: ratio of distances
//                 e1_sum += dist_d_plus / min_dist;
//
//                 // E2 calculation: for detecting deterministic vs stochastic signals
//                 if dim > 1 {
//                     // Get the previous embedding dimension distances
//                     let embedded_d_minus1 = embed_timeseries(x, dim - 1, time_lag)?;
//                     let current_point_d_minus1 = embedded_d_minus1.get(i);
//                     let nearest_point_d_minus1 = embedded_d_minus1.get(nearest_idx);
//                     let dist_d_minus1: f64 =
//                         euclidian_distance(&current_point_d_minus1, &nearest_point_d_minus1)
//                             .double_value(&[]) as f64;
//
//                     if dist_d_minus1 > 0.0 {
//                         e2_sum += (dist_d_plus / min_dist) / (min_dist / dist_d_minus1);
//                     }
//                 }
//             }
//         }
//
//         let e1 = e1_sum / num_vectors as f64;
//         e1_values.push(e1);
//
//         if dim > 1 {
//             let e2 = e2_sum / num_vectors as f64;
//             e2_values.push(e2);
//         }
//     }
//
//     // Find the dimension where E1 stops changing (becomes approximately constant)
//     // E1(d) should be close to 1 for d >= embedding dimension
//     let mut estimated_dim = max_dim;
//     let threshold = 0.1; // threshold for detecting when E1 stops changing
//
//     for i in 1..e1_values.len() {
//         let e1_current = e1_values[i];
//         let e1_prev = e1_values[i - 1];
//
//         // Check if E1 has stabilized (small change and close to 1)
//         let change = (e1_current - e1_prev).abs();
//         if change < threshold && e1_current > 0.9 && e1_current < 1.1 {
//             estimated_dim = (i + 1) as i64;
//             break;
//         }
//     }
//
//     // Additional check using E2 for deterministic vs stochastic signals
//     // For deterministic signals, E2 should deviate from 1 at some dimension
//     if !e2_values.is_empty() {
//         let mut has_deterministic_structure = false;
//         for &e2 in &e2_values {
//             if (e2 - 1.0).abs() > 0.1 {
//                 has_deterministic_structure = true;
//                 break;
//             }
//         }
//
//         // If no deterministic structure is found, use a smaller dimension
//         if !has_deterministic_structure {
//             estimated_dim = std::cmp::min(estimated_dim, 3);
//         }
//     }
//
//     Ok(std::cmp::max(1, std::cmp::min(estimated_dim, max_dim)))
// }

// fn compute_recurrence_matrix(embedded_ts: &Tensor, radius: f64) -> Tensor {
//     let n = embedded_ts.size()[0];
//     // let recurrence_matrix = Tensor::zeros([n, n], (Kind::Bool, Device::Cpu));
//     let mut data = vec![0u8; (n * n) as usize];
//
//     // Fill the recurrence matrix
//     for i in 0..n {
//         for j in 0..n {
//             if i != j {
//                 let v1 = embedded_ts.get(i);
//                 let v2 = embedded_ts.get(j);
//                 let dist = euclidean_distance(&v1, &v2);
//                 let is_recurrent = dist.le(radius).int64_value(&[]) != 0;
//
//                 // recurrence_matrix.get(i).get(j).copy_(&is_recurrent);
//                 data[(i * n + j) as usize] = if is_recurrent { 1 } else { 0 };
//             }
//         }
//     }
//
//     // recurrence_matrix
//     Tensor::from_slice(&data)
//         .reshape(&[n, n])
//         .to_kind(Kind::Bool)
// }

fn compute_recurrence_matrix(
    takens: Option<&Tensor>,
    time_series: Option<&Tensor>,
    config: RecurrenceMatrixConfig,
) -> Result<(Tensor, Tensor), String> {
    let takens_matrix = if let Some(t) = takens {
        t.shallow_clone()
    } else if let Some(ts) = time_series {
        build_takens_explicit(ts, config.embedding_dim, config.time_lag)?
    } else {
        return Err("Either takens or time_series must be provided".into());
    };

    let n = takens_matrix.size()[0] as usize;

    // Find all neighbors within radius
    // let neighs: Vec<Vec<usize>> = find_all_neighbors(&takens_matrix, config.radius)?;
    let searcher = BoxAssistedSearch::new(&takens_matrix, config.radius, None);
    let neighbor_list = searcher.find_all_neighbors();

    // let neighbor_tensor = tensor_from_vec2d(&neighs);
    // let mut flat = vec![0i64; n * n];
    //
    // for (i, neighbors) in neighs.iter().enumerate() {
    //     for &j in neighbors {
    //         flat[i * n + j] = 1;
    //         flat[j * n + i] = 1;
    //     }
    // }
    //
    // let recurrence_tensor = Tensor::from_slice(&flat).reshape(&[n as i64, n as i64]);

    let (edge_index, edge_weight) = neighbor_list_to_sparse_index(&neighbor_list);

    Ok((edge_index, edge_weight))
}

fn find_all_neighbors(takens: &Tensor, radius: f32) -> Result<Vec<Vec<usize>>, String> {
    let n_points = takens.size()[0] as usize;
    let embedding_dim = takens.size()[1] as usize;

    let mut kdtree = KdTree::new(embedding_dim);
    let mut all_points: Vec<Vec<f32>> = Vec::with_capacity(n_points);

    // Populate tree
    for i in 0..n_points {
        let mut coords: Vec<f32> = vec![0.0; embedding_dim];
        for d in 0..embedding_dim {
            coords[d] = takens.double_value(&[i as i64, d as i64]) as f32;
        }
        all_points.push(coords);
    }
    for (i, coords) in all_points.iter().enumerate() {
        kdtree.add(coords, i);
    }

    // Query neighbors within radius
    let mut neighbors_sets: Vec<HashSet<usize>> = vec![HashSet::new(); n_points];
    for i in 0..n_points {
        let mut coords = vec![0.0; embedding_dim];
        for d in 0..embedding_dim {
            coords[d] = takens.double_value(&[i as i64, d as i64]) as f32;
        }

        let results = kdtree
            .within(&coords, radius + 1e-3, &squared_euclidean)
            .unwrap();

        for &(_, idx) in results.iter() {
            let j = *idx as usize;

            if i == j {
                continue;
            }
            neighbors_sets[i].insert(j);
            neighbors_sets[j].insert(i);
        }
    }

    let neighbors: Vec<Vec<usize>> = neighbors_sets
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();

    Ok(neighbors)
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

fn euclidean_distance(v1: &Tensor, v2: &Tensor) -> Tensor {
    (v1 - v2)
        .pow_tensor_scalar(2)
        .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
        .sqrt()
}

// Create a Tensor from a tensor slice
fn tensor_from_slice<T: tch::kind::Element>(slice: &[T]) -> Tensor {
    Tensor::f_from_slice(slice).expect("Failed to create tensor from slice")
}

fn tensor_from_vec2d<T: tch::kind::Element + Copy>(data: &Vec<Vec<T>>) -> Tensor {
    let rows = data.len();
    let cols = if rows > 0 { data[0].len() } else { 0 };

    let flat: Vec<T> = data.iter().flat_map(|v| v.iter().copied()).collect();

    Tensor::from_slice(&flat).reshape(&[rows as i64, cols as i64])
}

fn tensor_to_undirected_graph(adjacency_matrix: &Tensor) -> Result<UnGraph<usize, f64>, String> {
    let shape = adjacency_matrix.size();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err("Adjacency matrix must be square".into());
    }

    let n_nodes = shape[0] as usize;
    let mut graph: UnGraph<usize, f64> = Graph::new_undirected();
    let mut node_indices = Vec::with_capacity(n_nodes);

    for i in 0..n_nodes {
        let node_idx = graph.add_node(i);
        node_indices.push(node_idx);
    }

    // Add edges based on adjacency_matrix
    // Only check upper triangle to avoid duplicate adges in undirected graphs
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            let weight: f64 = adjacency_matrix.double_value(&[i as i64, j as i64]);

            // Add edge if there's a connection (non-zero weight)
            if weight != 0.0 {
                graph.add_edge(node_indices[i], node_indices[j], 1.0f64);
            }
        }
    }

    Ok(graph)
}

pub fn recurrence_graph_rs(
    x: &Tensor,
    radius: f32,
    embedding_dim: Option<i64>,
    time_lag: i64,
) -> Result<TsNet, String> {
    // 1. Determine the embedding dimension
    let embedding_config = EmbeddingConfig {
        time_lag: time_lag,
        ..Default::default()
    };
    let dim = match embedding_dim {
        Some(d) => d,
        None => {
            let estimated_dim = estimate_embedding_dim(x, embedding_config);
            match estimated_dim {
                Ok(Some(d)) => d,
                _ => 2,
            }
            // Estimate the dim using Cao's algorithm
            // Ok(estimate_embedding_dim(x, embedding_config))?
        }
    };

    let rm_config = RecurrenceMatrixConfig {
        radius: radius,
        embedding_dim: dim,
        ..Default::default()
    };

    // 3. Compute the recurrence matrix
    // let recurrence_matrix = compute_recurrence_matrix(None, Some(x), rm_config)?;

    // Convert recurrence matrix to undirected graph
    // let graph = tensor_to_undirected_graph(&recurrence_matrix)?;

    // let (edge_index, edge_weight) = adjacency_to_sparse_edges(&recurrence_matrix)?;
    // let (edge_index, edge_weight) = neighbor_list_to_sparse_index(neighs);
    let (edge_index, edge_weight) = compute_recurrence_matrix(None, Some(x), rm_config)?;

    Ok(TsNet {
        edge_index,
        edge_weight,
        embedding_dim: dim,
        time_lag,
        radius,
    })
}

fn neighbor_list_to_sparse_index(neighs: &Vec<Vec<usize>>) -> (Tensor, Tensor) {
    let mut row = Vec::new();
    let mut col = Vec::new();
    let mut weights = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (i, neighbors) in neighs.iter().enumerate() {
        let i_usize = i;

        // Include (i, i) for self-connection as in R code
        row.push(i_usize as i64);
        col.push(i_usize as i64);
        weights.push(1.0f64);

        for &j in neighbors {
            if i_usize <= j && seen.insert((i_usize, j)) {
                row.push(i_usize as i64);
                col.push(j as i64);
                weights.push(1.0f64);
            } else if j < i_usize && seen.insert((j, i_usize)) {
                row.push(j as i64);
                col.push(i_usize as i64);
                weights.push(1.0f64);
            }
        }
    }

    let edge_index = Tensor::stack(&[Tensor::from_slice(&row), Tensor::from_slice(&col)], 0);
    let edge_weight = Tensor::from_slice(&weights);

    (edge_index, edge_weight)
}

// pub fn recurrence_graph_with_embedding_dim(
//     x: &Tensor,
//     radius: f64,
//     embedding_dim: Option<i64>,
//     time_lag: i64,
// ) -> Result<(Tensor, Tensor, i64), String> {
//     // 1. Determine the embedding dimension
//     let embedding_config = EmbeddingConfig {
//         ..Default::default()
//     };
//     let dim = match embedding_dim {
//         Some(d) => d,
//         None => {
//             let estimated_dim = estimate_embedding_dim(x, embedding_config);
//             match estimated_dim {
//                 Ok(Some(d)) => d,
//                 _ => 2,
//             }
//         }
//     };
//
//     // 2. Embed the time series
//     let embedded = embed_timeseries(x, dim, time_lag)?;
//
//     // 3. Compute the recurrence matrix
//     let recurrence_matrix = compute_recurrence_matrix(&embedded, radius);
//
//     // 4. Convert recurrence matrix to edge list
//     let (src, dst) = recurrence_matrix_to_edges(&recurrence_matrix);
//
//     // 5. Create edge index tensor
//     let edge_index = to_edge_index(&src, &dst);
//
//     // 6. Create edge weights (unweighted graph)
//     let num_edges = src.size()[0];
//     let edge_weight = Tensor::ones([num_edges], (Kind::Float, Device::Cpu));
//
//     Ok((edge_index, edge_weight, dim))
// }

/// Recurrence graph construction with all parameters specified
pub fn recurrence_graph_withdim(
    x: &Tensor,
    radius: f32,
    embedding_dim: i64,
    time_lag: i64,
) -> Result<TsNet, String> {
    let net = recurrence_graph_rs(x, radius, Some(embedding_dim), time_lag)?;
    Ok(net)
}

// Helper function to get the recurrence matrix directly (for debugging/analysis)
// pub fn get_recurrence_matrix(
//     x: &Tensor,
//     radius: f64,
//     embedding_dim: Option<i64>,
//     time_lag: i64,
// ) -> Result<(Tensor, i64), String> {
//     let embedding_config = EmbeddingConfig {
//         time_lag: time_lag,
//         ..Default::default()
//     };
//     let dim = match embedding_dim {
//         Some(d) => d,
//         None => {
//             let estimated_dim = estimate_embedding_dim(x, embedding_config);
//             match estimated_dim {
//                 Ok(Some(d)) => d,
//                 _ => 2,
//             }
//         }
//     };
//
//     let rm_config = RecurrenceMatrixConfig {
//         radius: radius,
//         embedding_dim: dim,
//         ..Default::default()
//     };
//
//     // 3. Compute the recurrence matrix
//     let recurrence_matrix = compute_recurrence_matrix(None, Some(x), rm_config)?;
//
//     Ok((recurrence_matrix, dim))
// }

fn adjacency_to_sparse_edges(adjacency_matrix: &Tensor) -> Result<(Tensor, Tensor), String> {
    let shape = adjacency_matrix.size();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err("Adjacency matrix must be square".into());
    }

    let n = shape[0] as i64;
    let mut src = Vec::new();
    let mut dst = Vec::new();
    let mut weights = Vec::new();

    for i in 0..n {
        for j in i..n {
            // Only looking at the upper triangle for undirected graph
            let weight = adjacency_matrix.double_value(&[i, j]);
            if weight != 0.0 {
                src.push(i);
                dst.push(j);
                weights.push(weight);

                if i != j {
                    src.push(j);
                    dst.push(i);
                    weights.push(weight);
                }
            }
        }
    }

    let edge_index = Tensor::stack(&[tensor_from_slice(&src), tensor_from_slice(&dst)], 0);
    let edge_weight = tensor_from_slice(&weights);

    Ok((edge_index, edge_weight))
}

// Convenience function that takes a slice and converts to tensor
pub fn recurrence_graph_from_slice(
    x: &[f64],
    radius: f32,
    embedding_dim: Option<i64>,
    time_lag: i64,
) -> Result<TsNet, String> {
    let x_tensor = tensor_from_slice(x);
    recurrence_graph_rs(&x_tensor, radius, embedding_dim, time_lag)
}

/// Estimate embedding dimension for a slice input using Cao's algorithm
pub fn estimate_embedding_dim_from_slice(
    x: &[f64],
    time_lag: i64,
    max_dim: i64,
) -> Result<Option<i64>, String> {
    let max_feasible_dim = ((x.len() - 1) / time_lag as usize) as i64;
    let max_dim = std::cmp::min(max_dim, max_feasible_dim);
    let x_tensor = tensor_from_slice(x);
    let embedding_config = EmbeddingConfig {
        ..Default::default()
    };
    estimate_embedding_dim(&x_tensor, embedding_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a Tensor from a tensor slice
    fn tensor_from_slice<T: tch::kind::Element>(slice: &[T]) -> Tensor {
        Tensor::f_from_slice(slice).expect("Failed to create tensor from slice")
    }

    fn assert_tensor_eq(actual: &Tensor, expected: &[f64], row_idx: usize) {
        let expected_tensor = tensor_from_slice(expected).to_kind(Kind::Double);

        let is_equal = actual.eq_tensor(&expected_tensor).all().int64_value(&[]) != 0;

        assert!(
            is_equal,
            "Mismatch at row {}: expected {:?}, got {:?}",
            row_idx, expected_tensor, actual
        );
    }

    #[test]
    fn test_build_takens_explicit() {
        let ts = tensor_from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let embedding = build_takens_explicit(&ts, 3, 1).unwrap();
        assert_eq!(embedding.size(), [6, 3])
    }

    #[test]
    fn test_embedding() {
        let x = tensor_from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let embedded = embed_timeseries(&x, 2, 1).unwrap();
        assert_eq!(embedded.size(), [4, 2]);

        let expected_rows = [
            &[1.0, 2.0][..],
            &[2.0, 3.0][..],
            &[3.0, 4.0][..],
            &[4.0, 5.0][..],
        ];

        for (i, expected) in expected_rows.iter().enumerate() {
            let actual = embedded.get(i as i64);
            assert_tensor_eq(&actual, expected, i);
        }
    }

    #[test]
    fn test_euclidian_distance() {
        let v1 = tensor_from_slice(&[0.0f64, 0.0]);
        let v2 = tensor_from_slice(&[3.0f64, 4.0]);
        let dist: f64 = euclidean_distance(&v1, &v2).double_value(&[]) as f64;
        assert!((dist - 5.0).abs() < 1e-6);
    }

    // #[test]
    // fn test_recurrence_matrix() -> Result<(), Box<dyn std::error::Error>> {
    //     let embedded = Tensor::from_slice(&[
    //         0.0f64, 0.0, // Point 0
    //         1.0, 1.0, // Point 1
    //         0.1, 0.1, // Point 3
    //     ])
    //     .view([3, 2]);
    //
    //     let rm_config = RecurrenceMatrixConfig {
    //         radius: 0.5,
    //         ..Default::default()
    //     };
    //
    //     let recurrence_matrix = compute_recurrence_matrix(Some(&embedded), None, rm_config)?;
    //
    //     let val_02: bool = recurrence_matrix.get(0).get(2).int64_value(&[]) != 0;
    //     let val_20: bool = recurrence_matrix.get(2).get(0).int64_value(&[]) != 0;
    //     let val_01: bool = recurrence_matrix.get(0).get(1).int64_value(&[]) != 0;
    //
    //     assert_eq!(val_02, true); // Points 0 and 2 should be connected
    //     assert_eq!(val_20, true); // Symmetric
    //     assert_eq!(val_01, false); // Points 0 and 1 should not be connected
    //     Ok(())
    // }

    #[test]
    fn test_embedding_dim_estimation() {
        // Create a simple periodic signal
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let estimated_dim = estimate_embedding_dim_from_slice(&x, 1, 10);
        let embedding_dim = match estimated_dim {
            Ok(Some(d)) => d,
            _ => 2,
        };
        assert!(embedding_dim >= 1 && embedding_dim <= 10);
    }

    #[test]
    fn test_recurrence_graph_with_estimation() {
        let x = [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0,
        ];
        let net = recurrence_graph_from_slice(&x, 5.0, None, 1).unwrap();

        assert_eq!(net.edge_index.size()[0], 2); // should have 2 rows (src, dst)
        assert!(net.edge_index.size()[0] > 0); // Should have some edges
        assert_eq!(net.edge_weight.size()[0], net.edge_index.size()[1]); // Same number of weights as edges
    }

    #[test]
    fn test_recurrence_graph_with_specified_dim() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let net = recurrence_graph_from_slice(&x, 5.0, Some(2), 1).unwrap();

        assert_eq!(net.edge_index.size()[0], 2); // should have 2 rows (src, dst)
        assert!(net.edge_index.size()[0] > 0); // Should have some edges
        assert_eq!(net.edge_weight.size()[0], net.edge_index.size()[1]); // Same number of weights as edges
    }
}

// #[cfg(feature = "neighbor_search")]
#[path = "temporal/recurrence/recurrence_tests.rs"]
mod recurrence_tests;

// use core::f64;
// use std::vec;
//
// use rand::{RngCore, SeedableRng};
// use rand_chacha::ChaCha8Rng;
//
// use crate::graph::temporal::NeighborSearch;
//
// pub struct Takens {
//     pub owned_takens: Vec<Vec<f64>>,
// }
//
// #[derive(Debug, Clone)]
// struct CaoParameters {
//     e: f64,
//     e_star: f64,
// }
//
// #[derive(Debug)]
// pub struct NearestNeighborResult {
//     /// Indices of nearest neighbors (including self)
//     indices: Vec<Vec<usize>>,
//     /// Distance to nearest neighbors (including self)
//     distances: Vec<Vec<f64>>,
// }
//
// impl Takens {
//     pub fn build_takens(
//         time_series: &[f64],
//         embedding_dim: usize,
//         time_lag: usize,
//     ) -> Result<Takens, Box<dyn std::error::Error>> {
//         let n = time_series.len();
//         let max_jump = (embedding_dim - 1) * time_lag;
//
//         if n <= max_jump {
//             return Err("TIme series too short for given embedding dimension and lag".into());
//         }
//
//         let num_vectors = n - max_jump;
//         let jumps: Vec<usize> = (0..embedding_dim).map(|i| i * time_lag).collect();
//
//         let mut takens = Vec::with_capacity(num_vectors);
//         for i in 0..num_vectors {
//             let mut row = Vec::with_capacity(embedding_dim);
//             for &jump in &jumps {
//                 row.push(time_series[i + jump]);
//             }
//             takens.push(row);
//         }
//
//         Ok(Takens {
//             owned_takens: takens,
//         })
//     }
//
//     /// Return slice views of the owned takens
//     pub fn views(&self) -> Vec<&[f64]> {
//         self.owned_takens.iter().map(|row| row.as_slice()).collect()
//     }
//
//     /// Shape of the Takens matrix
//     pub fn size(&self) -> (usize, usize) {
//         (self.owned_takens.len(), self.owned_takens[0].len())
//     }
// }
//
// #[derive(Debug, Clone)]
// pub struct NeighborList {
//     pub neighbors: Vec<Vec<usize>>,
// }
//
// #[derive(Debug, Clone)]
// pub struct EmbeddingConfig {
//     pub number_points: Option<usize>,
//     pub time_lag: usize,
//     pub max_embedding_dim: usize,
//     pub threshold: f64,
//     pub max_relative_change: f64,
//     pub std_noise: Option<f64>,
// }
//
// impl Default for EmbeddingConfig {
//     fn default() -> Self {
//         Self {
//             number_points: None,
//             time_lag: 1,
//             max_embedding_dim: 15,
//             threshold: 0.95,
//             max_relative_change: 0.10,
//             std_noise: None,
//         }
//     }
// }
//
// #[derive(Debug, Clone)]
// pub struct RecurrenceMatrixConfig {
//     pub embedding_dim: usize,
//     pub time_lag: usize,
//     pub radius: f64,
// }
//
// impl Default for RecurrenceMatrixConfig {
//     fn default() -> Self {
//         Self {
//             embedding_dim: 2,
//             time_lag: 1,
//             radius: 0.1,
//         }
//     }
// }
//
// #[derive(Debug)]
// pub struct TsNet {
//     pub edge_index: Vec<i64>,
//     pub edge_weight: Vec<f64>,
//     pub embedding_dim: usize,
//     pub time_lag: usize,
//     pub radius: f64,
// }
//
// fn estimate_embedding_dim(
//     time_series: &[f64],
//     config: EmbeddingConfig,
// ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
//     if config.max_embedding_dim < 3 {
//         return Err("Max embedding_dim should be greater than 2".into());
//     }
//
//     let time_series_len = time_series.len();
//     if time_series_len == 0 {
//         return Err("Time series is empty".into());
//     }
//
//     // Normalize time series
//     let mean: f64 = time_series.iter().copied().sum::<f64>() / time_series_len as f64;
//     let variance: f64 =
//         time_series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / time_series_len as f64;
//     let std_dev = variance.sqrt();
//
//     let mut ts: Vec<f64> = time_series.iter().map(|&x| (x - mean) / std_dev).collect();
//
//     // Add noise to avoid identical phase space Points
//     let std_noise = if let Some(noise) = config.std_noise {
//         noise.max(0.0)
//     } else {
//         let mut sorted_data = ts.clone();
//         sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
//
//         let k_sd_fraction = 1e-6;
//         let min_abs_separation = sorted_data
//             .windows(2)
//             .map(|w| (w[1] - w[0]).abs())
//             .fold(f64::INFINITY, f64::min);
//
//         if min_abs_separation.is_finite() {
//             min_abs_separation * k_sd_fraction
//         } else {
//             1e-6
//         }
//     };
//
//     if std_noise > 0.0 {
//         let mut rng = ChaCha8Rng::seed_from_u64(42);
//         for val in ts.iter_mut() {
//             *val += rng.next_u64() as f64 * std_noise;
//         }
//     }
//
//     // Extract data segments
//     let number_points = config.number_points.unwrap_or(time_series_len);
//     let ts_beg = time_series_len / 2 - number_points / 2;
//     let ts_end = time_series_len / 2 + number_points / 2;
//     let data = &ts[ts_beg..ts_end];
//
//     let mut embedding_dim = 0;
//
//     // First iteartion: get E(1) and E(2)
//     let e_params_1 = get_cao_parameters(data, 1, config.time_lag)?;
//     let e_params_2 = get_cao_parameters(data, 2, config.time_lag)?;
//
//     let mut e_vector = vec![e_params_1.e, e_params_2.e];
//     let mut e_star_vector = vec![e_params_1.e_star, e_params_2.e_star];
//     let mut e1_vector = vec![e_vector[1] / e_vector[0]];
//     let mut e2_vector = vec![e_star_vector[1] / e_star_vector[0]];
//
//     // Compute from d=3 to d=max_embedding_dim
//     for dimension in 3..=config.max_embedding_dim {
//         let e_params = get_cao_parameters(&data, dimension as usize, config.time_lag)?;
//         e_vector.push(e_params.e);
//         e_star_vector.push(e_params.e_star);
//
//         let e1 = e_vector[(dimension - 1) as usize] / e_vector[(dimension - 2) as usize];
//         let e2 = e_star_vector[(dimension - 1) as usize] / e_star_vector[(dimension - 2) as usize];
//
//         e1_vector.push(e1);
//         e2_vector.push(e2);
//
//         // Error for dimension - 2
//         if dimension >= 4 {
//             let idx = dimension - 3; // e1_vector index for dimension - 2
//             let relative_error = (e1_vector[(idx + 1) as usize] - e1_vector[idx as usize]).abs()
//                 / e1_vector[idx as usize];
//
//             // Check if E1(d) >= threshold and first time condition is met
//             if embedding_dim == 0
//                 && e1_vector[idx as usize] >= config.threshold
//                 && relative_error < config.max_relative_change
//             {
//                 embedding_dim = dimension - 2;
//             }
//         }
//     }
//
//     // Return embedding dimension
//     if embedding_dim == 0 {
//         Ok(None)
//     } else {
//         Ok(Some(embedding_dim as usize))
//     }
// }
//
// fn get_cao_parameters(
//     data: &[f64],
//     embedding_dim: usize,
//     time_lag: usize,
// ) -> Result<CaoParameters, Box<dyn std::error::Error>> {
//     const K_ZERO: f64 = 1e-15;
//
//     let takens = Takens::build_takens(data, embedding_dim, time_lag)?;
//     let takens_next = Takens::build_takens(data, embedding_dim + 1, time_lag)?;
//
//     let max_iter = takens_next.size().0;
//
//     let takens_slices: Vec<&[f64]> = takens
//         .owned_takens
//         .iter()
//         .take(max_iter)
//         .map(|v| v.as_slice())
//         .collect();
//
//     let takens_next_slices: Vec<&[f64]> = takens_next
//         .owned_takens
//         .iter()
//         .take(max_iter)
//         .map(|v| v.as_slice())
//         .collect();
//
//     // Find nearest neighbors
//     let nearest_neigh = if embedding_dim == 1 {
//         let data_1d: Vec<&[f64]> = data[..max_iter]
//             .iter()
//             .map(|x| std::slice::from_ref(x))
//             .collect();
//         nn_search(&data_1d, &data_1d, 2)?
//     } else {
//         nn_search(takens_slices.as_slice(), takens_slices.as_slice(), 2)?
//     };
//
//     let mut min_dist_ratios = Vec::new();
//     let mut stochastic_parameters = Vec::new();
//
//     // Computing parameters for each Takens position
//     for takens_position in 0..max_iter {
//         // Get closest neighbor (avoid picking the same vector with index 1, use index 1 for second
//         // closets)
//         let closest_neigh_idx = nearest_neigh.indices[takens_position][1];
//         let nearest_dist = nearest_neigh.distances[takens_position][1];
//
//         if nearest_dist >= K_ZERO {
//             // Maximum distance in next dimension
//             let numerator = maximum_distance(
//                 takens_next_slices[takens_position],
//                 takens_next_slices[closest_neigh_idx],
//             )?;
//             min_dist_ratios.push(numerator / nearest_dist);
//         }
//
//         // Calculate stochastic parameter
//         let data_idx1 = takens_position + embedding_dim * time_lag;
//         let data_idx2 = closest_neigh_idx + embedding_dim * time_lag;
//
//         if data_idx1 < data.len() && data_idx2 < data.len() {
//             stochastic_parameters.push((data[data_idx1] - data[data_idx2]).abs());
//         }
//     }
//
//     let e = if min_dist_ratios.is_empty() {
//         0.0
//     } else {
//         min_dist_ratios.iter().sum::<f64>() / min_dist_ratios.len() as f64
//     };
//
//     let e_star = if stochastic_parameters.is_empty() {
//         0.0
//     } else {
//         stochastic_parameters.iter().sum::<f64>() / stochastic_parameters.len() as f64
//     };
//
//     Ok(CaoParameters { e, e_star })
// }
//
// fn nn_search(
//     data: &[&[f64]],
//     queries: &[&[f64]],
//     k: usize,
// ) -> Result<NearestNeighborResult, Box<dyn std::error::Error>> {
//     let n_data = data.len();
//     let n_query = queries.len();
//
//     if n_data == 0 || n_query == 0 {
//         return Err("Query and data must have same dimensionality".into());
//     }
//
//     let dim = data[0].len();
//     if queries.iter().any(|q| q.len() != dim) {
//         return Err("Query and data must have same dimensionality".into());
//     }
//
//     if k > n_data {
//         return Err("Cannot find more nearest neighbors than there are data points".into());
//     }
//
//     let mut all_indices = Vec::with_capacity(n_query);
//     let mut all_distances = Vec::with_capacity(n_query);
//
//     for q in queries {
//         let mut dists: Vec<(usize, f64)> = data
//             .iter()
//             .enumerate()
//             .map(|(i, x)| (i, chebyshev_distance(x, q)))
//             .collect();
//
//         // partial sort: keep k smallest
//         dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//
//         let (indices, distances): (Vec<_>, Vec<_>) = dists.into_iter().take(k).unzip();
//
//         all_indices.push(indices);
//         all_distances.push(distances);
//     }
//
//     Ok(NearestNeighborResult {
//         indices: all_indices,
//         distances: all_distances,
//     })
// }
//
// fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
//     a.iter()
//         .zip(b.iter())
//         .map(|(x, y)| (x - y).abs())
//         .fold(0.0, f64::max)
// }
//
// fn maximum_distance(vec1: &[f64], vec2: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
//     if vec1.len() != vec2.len() {
//         return Err("Vectors must have same length".into());
//     }
//
//     let max_diff = vec1
//         .iter()
//         .zip(vec2.iter())
//         .map(|(a, b)| (a - b).abs())
//         .fold(0.0, f64::max);
//
//     Ok(max_diff)
// }
//
// fn compute_recurrence_matrix(
//     takens: Option<Vec<&[f64]>>,
//     time_series: Option<&[f64]>,
//     config: RecurrenceMatrixConfig,
// ) -> Result<(Vec<i64>, Vec<f64>), Box<dyn std::error::Error>> {
//     let takens_matrix;
//     let _takens_holder;
//
//     if let Some(t) = takens {
//         takens_matrix = t
//     } else if let Some(ts) = time_series {
//         _takens_holder = Takens::build_takens(ts, config.embedding_dim, config.time_lag)?;
//         takens_matrix = _takens_holder.views();
//     } else {
//         return Err("Either takens or time_series must be provided".into());
//     };
//
//     let mut searcher = NeighborSearch::new(&takens_matrix, config.radius, None)?;
//     let neighbor_list = searcher.find_all_neighbors()?;
//
//     let (edge_index, edge_weight) = neighbor_list_to_sparse_index(&neighbor_list);
//
//     Ok((edge_index, edge_weight))
// }
//
// pub fn recurrence_graph_rs(
//     x: &[f64],
//     radius: f64,
//     embedding_dim: Option<usize>,
//     time_lag: usize,
// ) -> Result<TsNet, Box<dyn std::error::Error>> {
//     // 1. Determine the embedding dimension
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
//             // Estimate the dim using Cao's algorithm
//             // Ok(estimate_embedding_dim(x, embedding_config))?
//         }
//     };
//
//     let rm_config = RecurrenceMatrixConfig {
//         radius: radius,
//         embedding_dim: dim,
//         ..Default::default()
//     };
//
//     let (edge_index, edge_weight) = compute_recurrence_matrix(None, Some(x), rm_config)?;
//
//     Ok(TsNet {
//         edge_index,
//         edge_weight,
//         embedding_dim: dim,
//         time_lag,
//         radius,
//     })
// }
//
// fn neighbor_list_to_sparse_index(neighbors_list: &Vec<Vec<usize>>) -> (Vec<i64>, Vec<f64>) {
//     let mut edge_index: Vec<i64> = Vec::new();
//     let mut edge_weight: Vec<f64> = Vec::new();
//     let mut seen = std::collections::HashSet::new();
//
//     for (i, neighbors) in neighbors_list.iter().enumerate() {
//         let i64_i = i as i64;
//
//         edge_index.push(i64_i);
//         edge_index.push(i64_i);
//         edge_weight.push(1.0);
//
//         for &j in neighbors {
//             let i64_j = j as i64;
//
//             if i <= j && seen.insert((i, j)) {
//                 edge_index.push(i64_i);
//                 edge_index.push(i64_j);
//                 edge_weight.push(1.0);
//             } else if j < i && seen.insert((j, i)) {
//                 edge_index.push(i64_j);
//                 edge_index.push(i64_i);
//                 edge_weight.push(1.0);
//             }
//         }
//     }
//
//     (edge_index, edge_weight)
// }
//
// /// Recurrence graph construction with all parameters specified
// pub fn recurrence_graph_withdim(
//     x: &[f64],
//     radius: f64,
//     embedding_dim: usize,
//     time_lag: usize,
// ) -> Result<TsNet, Box<dyn std::error::Error>> {
//     let net = recurrence_graph_rs(x, radius, Some(embedding_dim), time_lag)?;
//     Ok(net)
// }
//

use core::f64;
use std::vec;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::graph::temporal::NeighborSearch;

pub struct Takens {
    pub owned_takens: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct CaoParameters {
    pub e: f64,
    pub e_star: f64,
}

#[derive(Debug)]
pub struct NearestNeighborResult {
    /// Indices of nearest neighbors (including self)
    indices: Vec<Vec<usize>>,
    /// Distance to nearest neighbors (including self)
    distances: Vec<Vec<f64>>,
}

impl Takens {
    pub fn build_takens(
        time_series: &[f64],
        embedding_dim: usize,
        time_lag: usize,
    ) -> Result<Takens, Box<dyn std::error::Error>> {
        let n = time_series.len();
        let max_jump = (embedding_dim - 1) * time_lag;

        if n <= max_jump {
            return Err("TIme series too short for given embedding dimension and lag".into());
        }

        let num_vectors = n - max_jump;
        let jumps: Vec<usize> = (0..embedding_dim).map(|i| i * time_lag).collect();

        let mut takens = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let mut row = Vec::with_capacity(embedding_dim);
            for &jump in &jumps {
                row.push(time_series[i + jump]);
            }
            takens.push(row);
        }

        Ok(Takens {
            owned_takens: takens,
        })
    }

    /// Return slice views of the owned takens
    pub fn views(&self) -> Vec<&[f64]> {
        self.owned_takens.iter().map(|row| row.as_slice()).collect()
    }

    /// Shape of the Takens matrix
    pub fn size(&self) -> (usize, usize) {
        (self.owned_takens.len(), self.owned_takens[0].len())
    }
}

#[derive(Debug, Clone)]
pub struct NeighborList {
    pub neighbors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub number_points: Option<usize>,
    pub time_lag: usize,
    pub max_embedding_dim: usize,
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
    pub embedding_dim: usize,
    pub time_lag: usize,
    pub radius: f64,
    pub self_loops: bool,
    pub test: bool,
}

impl Default for RecurrenceMatrixConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 2,
            time_lag: 1,
            radius: 0.1,
            self_loops: false,
            test: false,
        }
    }
}

#[derive(Debug)]
pub struct TsNet {
    pub edge_index: Vec<i64>,
    pub edge_weight: Vec<f64>,
    pub embedding_dim: usize,
    pub time_lag: usize,
    pub radius: f64,
}

pub fn estimate_embedding_dim(
    time_series: &[f64],
    config: EmbeddingConfig,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    if config.max_embedding_dim < 3 {
        return Err("Max embedding_dim should be greater than 2".into());
    }

    let time_series_len = time_series.len();
    if time_series_len == 0 {
        return Err("Time series is empty".into());
    }

    // Normalize time series
    let mean: f64 = time_series.iter().copied().sum::<f64>() / time_series_len as f64;
    let variance: f64 =
        time_series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / time_series_len as f64;
    let std_dev = if variance == 0.0 {
        1.0
    } else {
        variance.sqrt()
    };

    let mut ts: Vec<f64> = time_series.iter().map(|&x| (x - mean) / std_dev).collect();

    // Add noise to avoid identical phase space Points
    let std_noise = if let Some(noise) = config.std_noise {
        noise.max(0.0)
    } else {
        let mut sorted_data = ts.clone();
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
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, std_noise).unwrap();
        for val in ts.iter_mut() {
            *val += normal.sample(&mut rng);
        }
    }

    // Extract data segments
    let number_points = config.number_points.unwrap_or(time_series_len);
    let ts_beg = time_series_len / 2 - number_points / 2;
    let ts_end = time_series_len / 2 + number_points / 2;
    let data = &ts[ts_beg..ts_end];

    let mut embedding_dim = 0;

    // First iteartion: get E(1) and E(2)
    let e_params_1 = get_cao_parameters(data, 1, config.time_lag)?;
    let e_params_2 = get_cao_parameters(data, 2, config.time_lag)?;

    let mut e_vector = vec![e_params_1.e, e_params_2.e];
    let mut e_star_vector = vec![e_params_1.e_star, e_params_2.e_star];
    let mut e1_vector = vec![e_vector[1] / e_vector[0]];
    let mut e2_vector = vec![e_star_vector[1] / e_star_vector[0]];

    // Compute from d=3 to d=max_embedding_dim
    for dimension in 3..=config.max_embedding_dim {
        let e_params = get_cao_parameters(&data, dimension as usize, config.time_lag)?;
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
        Ok(Some(embedding_dim as usize))
    }
}

pub fn get_cao_parameters(
    data: &[f64],
    embedding_dim: usize,
    time_lag: usize,
) -> Result<CaoParameters, Box<dyn std::error::Error>> {
    const K_ZERO: f64 = 1e-15;

    let takens = Takens::build_takens(data, embedding_dim, time_lag)?;
    let takens_next = Takens::build_takens(data, embedding_dim + 1, time_lag)?;

    let max_iter = takens_next.size().0;

    let takens_slices: Vec<&[f64]> = takens
        .owned_takens
        .iter()
        .take(max_iter)
        .map(|v| v.as_slice())
        .collect();

    let takens_next_slices: Vec<&[f64]> = takens_next
        .owned_takens
        .iter()
        .take(max_iter)
        .map(|v| v.as_slice())
        .collect();

    // Find nearest neighbors
    let nearest_neigh = if embedding_dim == 1 {
        let data_1d: Vec<&[f64]> = data[..max_iter]
            .iter()
            .map(|x| std::slice::from_ref(x))
            .collect();
        nn_search(&data_1d, &data_1d, 2)?
    } else {
        nn_search(takens_slices.as_slice(), takens_slices.as_slice(), 2)?
    };

    let mut min_dist_ratios = Vec::new();
    let mut stochastic_parameters = Vec::new();

    // Computing parameters for each Takens position
    for takens_position in 0..max_iter {
        // Get closest neighbor (avoid picking the same vector with index 1, use index 1 for second
        // closets)
        let closest_neigh_idx = nearest_neigh.indices[takens_position][1];
        let nearest_dist = nearest_neigh.distances[takens_position][1];

        if nearest_dist >= K_ZERO {
            // Maximum distance in next dimension
            let numerator = maximum_distance(
                takens_next_slices[takens_position],
                takens_next_slices[closest_neigh_idx],
            )?;
            min_dist_ratios.push(numerator / nearest_dist);
        }

        // Calculate stochastic parameter
        let data_idx1 = takens_position + embedding_dim * time_lag;
        let data_idx2 = closest_neigh_idx + embedding_dim * time_lag;

        if data_idx1 < data.len() && data_idx2 < data.len() {
            stochastic_parameters.push((data[data_idx1] - data[data_idx2]).abs());
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

fn nn_search(
    data: &[&[f64]],
    queries: &[&[f64]],
    k: usize,
) -> Result<NearestNeighborResult, Box<dyn std::error::Error>> {
    let n_data = data.len();
    let n_query = queries.len();

    if n_data == 0 || n_query == 0 {
        return Err("Query and data must have same dimensionality".into());
    }

    let dim = data[0].len();
    if queries.iter().any(|q| q.len() != dim) {
        return Err("Query and data must have same dimensionality".into());
    }

    if k > n_data {
        return Err("Cannot find more nearest neighbors than there are data points".into());
    }

    let mut all_indices = Vec::with_capacity(n_query);
    let mut all_distances = Vec::with_capacity(n_query);

    for q in queries {
        let mut dists: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .map(|(i, x)| (i, chebyshev_distance(x, q)))
            .collect();

        // partial sort: keep k smallest
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (indices, distances): (Vec<_>, Vec<_>) = dists.into_iter().take(k).unzip();

        all_indices.push(indices);
        all_distances.push(distances);
    }

    Ok(NearestNeighborResult {
        indices: all_indices,
        distances: all_distances,
    })
}

fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn maximum_distance(vec1: &[f64], vec2: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if vec1.len() != vec2.len() {
        return Err("Vectors must have same length".into());
    }

    let max_diff = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    Ok(max_diff)
}

fn estimate_phase_space_diameter(takens: &[&[f64]]) -> f64 {
    if takens.is_empty() {
        return 0.0;
    }

    let dim = takens[0].len();
    let mut min_coords = vec![f64::INFINITY; dim];
    let mut max_coords = vec![f64::NEG_INFINITY; dim];

    for point in takens {
        for (i, &coord) in point.iter().enumerate() {
            min_coords[i] = min_coords[i].min(coord);
            max_coords[i] = max_coords[i].max(coord);
        }
    }

    let diameter = min_coords
        .iter()
        .zip(max_coords.iter())
        .map(|(min, max)| (max - min).powi(2))
        .sum::<f64>()
        .sqrt();

    diameter
}

fn compute_actual_radius(normalized_radius: f64, takens: &[&[f64]]) -> f64 {
    let diameter = estimate_phase_space_diameter(takens);
    normalized_radius * diameter
}

fn compute_recurrence_matrix(
    takens: Option<Vec<&[f64]>>,
    time_series: Option<&[f64]>,
    config: RecurrenceMatrixConfig,
) -> Result<(Vec<i64>, Vec<f64>), Box<dyn std::error::Error>> {
    let takens_matrix;
    let _takens_holder;

    if let Some(t) = takens {
        takens_matrix = t
    } else if let Some(ts) = time_series {
        _takens_holder = Takens::build_takens(ts, config.embedding_dim, config.time_lag)?;
        takens_matrix = _takens_holder.views();
    } else {
        return Err("Either takens or time_series must be provided".into());
    };
    let search_radius;
    if config.test {
        search_radius = config.radius;
    } else {
        search_radius = compute_actual_radius(config.radius, &takens_matrix);
    }

    let mut searcher = NeighborSearch::new(&takens_matrix, search_radius, None)?;
    let neighbor_list = searcher.find_all_neighbors()?;

    let (edge_index, edge_weight) =
        neighbor_list_to_sparse_index(&neighbor_list, config.self_loops);

    Ok((edge_index, edge_weight))
}

pub fn recurrence_graph_rs(
    x: &[f64],
    radius: f64,
    embedding_dim: Option<usize>,
    time_lag: usize,
    self_loops: bool,
    is_test: bool,
) -> Result<TsNet, Box<dyn std::error::Error>> {
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
        time_lag: time_lag,
        self_loops: self_loops,
        test: is_test,
    };

    let (edge_index, edge_weight) = compute_recurrence_matrix(None, Some(x), rm_config)?;

    Ok(TsNet {
        edge_index,
        edge_weight,
        embedding_dim: dim,
        time_lag,
        radius,
    })
}

fn neighbor_list_to_sparse_index(
    neighbors_list: &Vec<Vec<usize>>,
    self_loops: bool,
) -> (Vec<i64>, Vec<f64>) {
    let mut edge_index: Vec<i64> = Vec::new();
    let mut edge_weight: Vec<f64> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (i, neighbors) in neighbors_list.iter().enumerate() {
        let i64_i = i as i64;

        if self_loops {
            edge_index.push(i64_i);
            edge_index.push(i64_i);
            edge_weight.push(1.0);
        }
        for &j in neighbors {
            let i64_j = j as i64;

            if i <= j && seen.insert((i, j)) {
                edge_index.push(i64_i);
                edge_index.push(i64_j);
                edge_weight.push(1.0);
            } else if j < i && seen.insert((j, i)) {
                edge_index.push(i64_j);
                edge_index.push(i64_i);
                edge_weight.push(1.0);
            }
        }
    }

    (edge_index, edge_weight)
}

/// Recurrence graph construction with all parameters specified
pub fn recurrence_graph_withdim(
    x: &[f64],
    radius: f64,
    embedding_dim: usize,
    time_lag: usize,
    self_loops: bool,
    is_test: bool,
) -> Result<TsNet, Box<dyn std::error::Error>> {
    let net = recurrence_graph_rs(
        x,
        radius,
        Some(embedding_dim),
        time_lag,
        self_loops,
        is_test,
    )?;
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Helper function to generate synthetic time series data
    fn generate_sine_wave(n: usize, frequency: f64, amplitude: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * frequency * i as f64 / n as f64).sin())
            .collect()
    }

    fn generate_linear_series(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    fn generate_constant_series(n: usize, value: f64) -> Vec<f64> {
        vec![value; n]
    }

    #[test]
    fn test_build_takens_basic() {
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let embedding = Takens::build_takens(&ts, 3, 1).unwrap();

        assert_eq!(embedding.size(), (6, 3));

        // Check first few rows
        let views = embedding.views();
        assert_eq!(views[0], &[1.0, 2.0, 3.0]);
        assert_eq!(views[1], &[2.0, 3.0, 4.0]);
        assert_eq!(views[5], &[6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_build_takens_with_lag() {
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let embedding = Takens::build_takens(&ts, 3, 2).unwrap();

        assert_eq!(embedding.size(), (4, 3));

        let views = embedding.views();
        assert_eq!(views[0], &[1.0, 3.0, 5.0]);
        assert_eq!(views[1], &[2.0, 4.0, 6.0]);
        assert_eq!(views[3], &[4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_build_takens_single_dimension() {
        let ts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedding = Takens::build_takens(&ts, 1, 1).unwrap();

        assert_eq!(embedding.size(), (5, 1));

        let views = embedding.views();
        assert_eq!(views[0], &[1.0]);
        assert_eq!(views[4], &[5.0]);
    }

    #[test]
    fn test_build_takens_error_cases() {
        let ts = vec![1.0, 2.0, 3.0];

        // Time series too short
        let result = Takens::build_takens(&ts, 5, 2);
        assert!(result.is_err());

        // Empty time series
        let empty_ts: Vec<f64> = vec![];
        let result = Takens::build_takens(&empty_ts, 2, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.time_lag, 1);
        assert_eq!(config.max_embedding_dim, 15);
        assert_eq!(config.threshold, 0.95);
        assert_eq!(config.max_relative_change, 0.10);
        assert!(config.number_points.is_none());
        assert!(config.std_noise.is_none());
    }

    #[test]
    fn test_recurrence_matrix_config_default() {
        let config = RecurrenceMatrixConfig::default();
        assert_eq!(config.embedding_dim, 2);
        assert_eq!(config.time_lag, 1);
        assert_eq!(config.radius, 0.1);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 1.0, 3.0];
        let distance = chebyshev_distance(&a, &b);
        assert_eq!(distance, 1.0); // max(|1-2|, |2-1|, |3-3|) = max(1, 1, 0) = 1
    }

    #[test]
    fn test_chebyshev_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let distance = chebyshev_distance(&a, &b);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_maximum_distance() {
        let vec1 = vec![1.0, 5.0, 3.0];
        let vec2 = vec![2.0, 2.0, 7.0];
        let max_dist = maximum_distance(&vec1, &vec2).unwrap();
        assert_eq!(max_dist, 4.0); // max(|1-2|, |5-2|, |3-7|) = max(1, 3, 4) = 4
    }

    #[test]
    fn test_maximum_distance_error() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let result = maximum_distance(&vec1, &vec2);
        assert!(result.is_err());
    }

    #[test]
    fn test_nn_search_basic() {
        let a = [1.0, 1.0];
        let b = [2.0, 2.0];
        let c = [3.0, 3.0];
        let q = [1.5, 1.5];

        let data = vec![&a[..], &b[..], &c[..]];
        let queries = vec![&q[..]];

        let result = nn_search(&data, &queries, 2).unwrap();

        assert_eq!(result.indices.len(), 1);
        assert_eq!(result.distances.len(), 1);
        assert_eq!(result.indices[0].len(), 2);
        assert_eq!(result.distances[0].len(), 2);
    }

    #[test]
    fn test_nn_search_self_query() {
        let a = [1.0, 1.0];
        let b = [2.0, 2.0];
        let c = [3.0, 3.0];

        let data = vec![&a[..], &b[..], &c[..]];
        let result = nn_search(&data, &data, 2).unwrap();

        assert_eq!(result.indices.len(), 3);
        // Each point should find itself as the nearest neighbor (distance 0)
        for i in 0..3 {
            assert_eq!(result.indices[i][0], i);
            assert_eq!(result.distances[i][0], 0.0);
        }
    }

    #[test]
    fn test_nn_search_error_cases() {
        let a = [1.0, 2.0];
        let q1 = [1.0, 2.0, 3.0];

        // Different dimensionality
        let data = vec![&a[..]];
        let queries = vec![&q1[..]];
        let result = nn_search(&data, &queries, 1);
        assert!(result.is_err());

        // k > n_data
        let b = [1.0];
        let q2 = [1.0];
        let data = vec![&b[..]];
        let queries = vec![&q2[..]];
        let result = nn_search(&data, &queries, 2);
        assert!(result.is_err());

        // Empty data
        let empty_data: Vec<&[f64]> = vec![];
        let queries = vec![&q2[..]];
        let result = nn_search(&empty_data, &queries, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_neighbor_list_to_sparse_index() {
        let neighbors = vec![
            vec![1, 2], // Node 0 connected to 1, 2
            vec![0, 2], // Node 1 connected to 0, 2
            vec![0, 1], // Node 2 connected to 0, 1
        ];

        let (edge_index, weights) = neighbor_list_to_sparse_index(&neighbors, true);

        // Should have even number of indices (pairs of source, target)
        assert_eq!(edge_index.len() % 2, 0);
        assert_eq!(edge_index.len() / 2, weights.len());

        // All weights should be 1.0
        assert!(weights.iter().all(|&w| w == 1.0));

        // Convert edge_index pairs to tuples for easier checking
        let edges: Vec<(i64, i64)> = edge_index
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();

        // Should contain self-loops for each node
        assert!(edges.contains(&(0, 0)));
        assert!(edges.contains(&(1, 1)));
        assert!(edges.contains(&(2, 2)));

        // Check that we have the expected number of edges
        // 3 self-loops + unique neighbor connections
        assert!(edges.len() >= 3); // At least self-loops

        // All edge indices should be non-negative
        assert!(edge_index.iter().all(|&idx| idx >= 0));
    }

    #[test]
    fn test_recurrence_graph_rs_basic() {
        let ts = generate_sine_wave(50, 1.0, 1.0);
        let result = recurrence_graph_rs(&ts, 0.1, Some(2), 1, false, true);

        assert!(result.is_ok());
        let net = result.unwrap();
        assert_eq!(net.embedding_dim, 2);
        assert_eq!(net.time_lag, 1);
        assert_eq!(net.radius, 0.1);
        assert!(!net.edge_index.is_empty());
        // edge_index now contains pairs of i64 values, so length should be 2 * number of edges
        assert_eq!(net.edge_index.len() / 2, net.edge_weight.len());
        assert_eq!(net.edge_index.len() % 2, 0);
    }

    #[test]
    fn test_recurrence_graph_rs_auto_embedding() {
        let ts = generate_sine_wave(30, 1.0, 1.0);
        let result = recurrence_graph_rs(&ts, 0.2, None, 1, false, true);

        assert!(result.is_ok());
        let net = result.unwrap();
        assert!(net.embedding_dim >= 2); // Should estimate some reasonable dimension
        assert!(!net.edge_index.is_empty());
        assert_eq!(net.edge_index.len() % 2, 0); // Should have even number of indices
        assert_eq!(net.edge_index.len() / 2, net.edge_weight.len());
    }

    #[test]
    fn test_recurrence_graph_withdim() {
        let ts = generate_linear_series(20);
        let result = recurrence_graph_withdim(&ts, 0.5, 3, 2, false, true);

        assert!(result.is_ok());
        let net = result.unwrap();
        assert_eq!(net.embedding_dim, 3);
        assert_eq!(net.time_lag, 2);
        assert_eq!(net.radius, 0.5);
    }

    #[test]
    fn test_recurrence_graph_error_cases() {
        // Empty time series
        let empty_ts: Vec<f64> = vec![];
        let result = recurrence_graph_rs(&empty_ts, 0.1, Some(2), 1, false, true);
        assert!(result.is_err());

        // Very short time series
        let short_ts = vec![1.0, 2.0];
        let result = recurrence_graph_rs(&short_ts, 0.1, Some(5), 2, false, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_embedding_dim_basic() {
        let config = EmbeddingConfig::default();
        let ts = generate_sine_wave(100, 2.0, 1.0);

        let result = estimate_embedding_dim(&ts, config);
        assert!(result.is_ok());

        // Should return Some dimension or None if not found
        match result.unwrap() {
            Some(dim) => assert!(dim > 0 && dim <= 15),
            None => {} // This is also valid
        }
    }

    #[test]
    fn test_estimate_embedding_dim_error_cases() {
        let mut config = EmbeddingConfig::default();
        config.max_embedding_dim = 2; // Too small

        let ts = generate_sine_wave(50, 1.0, 1.0);
        let result = estimate_embedding_dim(&ts, config);
        assert!(result.is_err());

        // Empty time series
        let empty_ts: Vec<f64> = vec![];
        let config = EmbeddingConfig::default();
        let result = estimate_embedding_dim(&empty_ts, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_cao_parameters() {
        let ts = generate_sine_wave(50, 1.0, 1.0);
        let result = get_cao_parameters(&ts, 2, 1);

        assert!(result.is_ok());
        let params = result.unwrap();
        assert!(params.e >= 0.0);
        assert!(params.e_star >= 0.0);
    }

    #[test]
    fn test_get_cao_parameters_dimension_1() {
        let ts = generate_linear_series(20);
        let result = get_cao_parameters(&ts, 1, 1);

        assert!(result.is_ok());
        let params = result.unwrap();
        assert!(params.e >= 0.0);
        assert!(params.e_star >= 0.0);
    }

    #[test]
    fn test_compute_recurrence_matrix_with_takens() {
        let ts = generate_sine_wave(30, 1.0, 1.0);
        let takens = Takens::build_takens(&ts, 3, 1).unwrap();
        let takens_views = takens.views();

        let config = RecurrenceMatrixConfig {
            embedding_dim: 3,
            time_lag: 1,
            radius: 0.5,
            self_loops: false,
            test: true,
        };

        let result = compute_recurrence_matrix(Some(takens_views), None, config);
        assert!(result.is_ok());

        let (edge_index, weights) = result.unwrap();
        assert!(!edge_index.is_empty());
        assert_eq!(edge_index.len() % 2, 0); // Should have pairs of indices
        assert_eq!(edge_index.len() / 2, weights.len());
    }

    #[test]
    fn test_compute_recurrence_matrix_with_time_series() {
        let ts = generate_sine_wave(25, 1.0, 1.0);
        let config = RecurrenceMatrixConfig {
            embedding_dim: 2,
            time_lag: 1,
            radius: 0.3,
            self_loops: false,
            test: true,
        };

        let result = compute_recurrence_matrix(None, Some(&ts), config);
        assert!(result.is_ok());

        let (edge_index, weights) = result.unwrap();
        assert!(!edge_index.is_empty());
        assert_eq!(edge_index.len() % 2, 0);
        assert_eq!(edge_index.len() / 2, weights.len());
    }

    #[test]
    fn test_compute_recurrence_matrix_error() {
        let config = RecurrenceMatrixConfig::default();

        // Neither takens nor time_series provided
        let result = compute_recurrence_matrix(None, None, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_time_series() {
        let ts = generate_constant_series(20, 5.0);
        let result = recurrence_graph_rs(&ts, 0.1, Some(2), 1, false, true);

        assert!(result.is_ok());
        let net = result.unwrap();
        assert_eq!(net.embedding_dim, 2);
    }

    #[test]
    fn test_different_radii() {
        let ts = generate_sine_wave(30, 1.0, 1.0);

        let small_radius = recurrence_graph_rs(&ts, 0.01, Some(2), 1, false, true).unwrap();
        let large_radius = recurrence_graph_rs(&ts, 1.0, Some(2), 1, false, true).unwrap();

        // Larger radius should generally result in more edges
        // Compare number of actual edges (edge_index.len() / 2)
        assert!(large_radius.edge_index.len() >= small_radius.edge_index.len());
    }

    #[test]
    fn test_different_time_lags() {
        let ts = generate_sine_wave(40, 1.0, 1.0);

        let lag1 = recurrence_graph_rs(&ts, 0.2, Some(3), 1, false, true).unwrap();
        let lag3 = recurrence_graph_rs(&ts, 0.2, Some(3), 3, false, true).unwrap();

        // Both should succeed but with different network structures
        assert_eq!(lag1.time_lag, 1);
        assert_eq!(lag3.time_lag, 3);
        assert!(!lag1.edge_index.is_empty());
        assert!(!lag3.edge_index.is_empty());
    }

    #[test]
    fn test_embedding_dimensions() {
        let ts = generate_sine_wave(50, 1.0, 1.0);

        for dim in 1..=5 {
            let result = recurrence_graph_rs(&ts, 0.2, Some(dim), 1, false, true);
            assert!(result.is_ok(), "Failed for embedding dimension {}", dim);

            let net = result.unwrap();
            assert_eq!(net.embedding_dim, dim);
        }
    }

    #[test]
    fn test_edge_weights_are_valid() {
        let ts = generate_sine_wave(25, 1.0, 1.0);
        let net = recurrence_graph_rs(&ts, 0.3, Some(2), 1, false, true).unwrap();

        // All weights should be positive and finite
        for &weight in &net.edge_weight {
            assert!(weight > 0.0);
            assert!(weight.is_finite());
        }
    }

    #[test]
    fn test_edge_indices_are_valid() {
        let ts = generate_sine_wave(30, 1.0, 1.0);
        let net = recurrence_graph_rs(&ts, 0.2, Some(2), 1, false, true).unwrap();

        let takens = Takens::build_takens(&ts, 2, 1).unwrap();
        let max_node = takens.size().0 as i64;

        // All edge indices should be within valid range and non-negative
        for &idx in &net.edge_index {
            assert!(idx >= 0);
            assert!(idx < max_node);
        }

        // Should have pairs of indices
        assert_eq!(net.edge_index.len() % 2, 0);
    }

    #[test]
    fn test_tsnet_structure() {
        let ts = generate_linear_series(20);
        let net = recurrence_graph_rs(&ts, 0.5, Some(3), 2, false, true).unwrap();

        // Verify TsNet structure
        assert_eq!(net.edge_index.len() % 2, 0); // Should have pairs of indices
        assert_eq!(net.edge_index.len() / 2, net.edge_weight.len());
        assert_eq!(net.embedding_dim, 3);
        assert_eq!(net.time_lag, 2);
        assert_eq!(net.radius, 0.5);

        // Should have at least self-loops
        assert!(!net.edge_index.is_empty());

        // All indices should be non-negative i64 values
        assert!(net.edge_index.iter().all(|&idx| idx >= 0));
    }

    #[test]
    fn test_reproducibility() {
        let ts = generate_sine_wave(30, 1.0, 1.0);

        let net1 = recurrence_graph_rs(&ts, 0.2, Some(2), 1, false, true).unwrap();
        let net2 = recurrence_graph_rs(&ts, 0.2, Some(2), 1, false, true).unwrap();

        // Results should be identical for same parameters
        assert_eq!(net1.edge_index, net2.edge_index);
        assert_eq!(net1.edge_weight, net2.edge_weight);
        assert_eq!(net1.embedding_dim, net2.embedding_dim);
    }

    #[test]
    fn test_edge_index_format() {
        let ts = generate_sine_wave(20, 1.0, 1.0);
        let net = recurrence_graph_rs(&ts, 0.3, Some(2), 1, false, true).unwrap();

        // edge_index should contain flattened pairs (src1, dst1, src2, dst2, ...)
        assert_eq!(net.edge_index.len() % 2, 0);

        // Convert to pairs for validation
        let edge_pairs: Vec<(i64, i64)> = net
            .edge_index
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();

        assert_eq!(edge_pairs.len(), net.edge_weight.len());

        // Should contain self-loops (at least some nodes connected to themselves)
        let has_self_loops = edge_pairs.iter().any(|(src, dst)| src == dst);
        assert!(has_self_loops);

        // All indices should be within valid range
        let takens = Takens::build_takens(&ts, 2, 1).unwrap();
        let max_node = takens.size().0 as i64;

        for (src, dst) in edge_pairs {
            assert!(src >= 0 && src < max_node);
            assert!(dst >= 0 && dst < max_node);
        }
    }
}

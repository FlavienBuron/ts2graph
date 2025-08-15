use tch::{IndexOp, Tensor};

// Test configuratiion structure
#[derive(Debug, Clone)]
struct TestConfig {
    name: String,
    data: Vec<f64>,
    radius: f32,
    embedding_dim: Option<i64>,
    time_lag: i64,
    expected_min_edges: usize,
    expected_max_edges: Option<usize>,
    tolerance: f32,
}

impl TestConfig {
    fn new(name: &str, data: Vec<f64>, radius: f32) -> Self {
        Self {
            name: name.to_string(),
            data: data,
            radius: radius,
            embedding_dim: None,
            time_lag: 1,
            expected_min_edges: 0,
            expected_max_edges: None,
            tolerance: 1e-6,
        }
    }

    fn with_embedding_dim(mut self, dim: i64) -> Self {
        self.embedding_dim = Some(dim);
        self
    }

    fn with_time_lag(mut self, lag: i64) -> Self {
        self.time_lag = lag;
        self
    }

    fn with_expected_edges(mut self, min: usize, max: Option<usize>) -> Self {
        self.expected_min_edges = min;
        self.expected_max_edges = max;
        self
    }

    fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Test result structure for detailed analysis
#[derive(Debug)]
struct TestResults {
    config: TestConfig,
    rust_edges: usize,
    rust_embedding_dim: i64,
    r_edges: Option<usize>,
    r_embedding_dim: Option<i64>,
    edges_match: Option<bool>,
    embedding_dim_match: Option<bool>,
    execution_time_rust: std::time::Duration,
    execution_time_r: Option<std::time::Duration>,
}

/// Generate standard test datasets
mod test_data {

    pub fn linear_trend() -> Vec<f64> {
        (0..20).map(|i| i as f64).collect()
    }

    pub fn sine_wave(length: usize, frequency: f64, amplitude: f64) -> Vec<f64> {
        (0..length)
            .map(|i| {
                amplitude
                    * (2.0 * std::f64::consts::PI * frequency * i as f64 / length as f64).sin()
            })
            .collect()
    }

    pub fn noisy_sine_wave(
        length: usize,
        frequency: f64,
        amplitude: f64,
        noise_level: f64,
    ) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::rng();
        (0..length)
            .map(|i| {
                let signal = amplitude
                    * (2.0 * std::f64::consts::PI * frequency * i as f64 / length as f64).sin();
                let noise = noise_level * rng.random_range(-1.0..1.0);
                signal + noise
            })
            .collect()
    }

    pub fn logistic_map(length: usize, r: f64, x0: f64) -> Vec<f64> {
        let mut result = Vec::with_capacity(length);
        let mut x = x0;
        for _ in 0..length {
            x = r * x * (1.0 - x);
            result.push(x);
        }
        result
    }

    pub fn constant_series(length: usize, value: f64) -> Vec<f64> {
        vec![value; length]
    }

    pub fn random_walk(length: usize, step_size: f64) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::rng();
        let mut result = Vec::with_capacity(length);
        let mut current = 0.0;

        for _ in 0..length {
            current += step_size * rng.random_range(-1.0..1.0);
            result.push(current);
        }
        result
    }

    pub fn multi_frequency_signal(length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| {
                let t = i as f64 / length as f64;
                (2.0 * std::f64::consts::PI * t).sin()
                    + 0.5 * (4.0 * std::f64::consts::PI * t).sin()
                    + 0.25 * (8.0 * std::f64::consts::PI * t).sin()
            })
            .collect()
    }
}

/// R Comparison Tests
#[cfg(feature = "integration")]
mod r_comparison_test {
    use extendr_api::prelude::Scalar;
    use extendr_api::scalar::{Rfloat, Rint};
    use extendr_api::{
        Doubles, Integers, Operators, R, RMatrix, Rinternals, Robj, matrix, robj::Types,
    };
    use rand::rand_core::le;
    use std::cmp::{max, min};
    use std::collections::HashSet;

    use super::*;
    use ts2graph_rs::graph::temporal::recurrence::recurrence_graph_from_slice;

    fn call_r_tsnet_vg(
        x: &[f64],
        radius: f32,
        embedding_dim: Option<i64>,
        time_lag: i64,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>, f64), String> {
        extendr_engine::start_r();
        println!("Calling R");

        let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let x_r = Doubles::from_values(&x_f64);

        let result = if let Some(dim) = embedding_dim {
            R!("
                    library(ts2net)
                    library(igraph)

                    net <- tsnet_rn(x = {{x_r}}, radius={{radius}}, embedding.dim = {{dim}}, time.lag = {{time_lag}})

                    if (is.null(net$embedding_dim)) {
                        embedding_dim <- 0
                    } else {
                        embedding_dim <- net$embedding_dim
                    }

                    edge_list <- get.edgelist(net)
                    if (nrow(edge_list) == 0) {
                        edge_list <- matrix(integer(0), ncol = 2)
                    }
                    edge_weights <- E(net)$weight
                    if (is.null(edge_weights)) {
                        edge_weights <- rep(1.0, ecount(net))
                    }

                    list(
                        edge_index = apply(edge_list),
                        edge_weight = edge_weights,
                        embedding_dim = embedding_dim,
                        num_vertices = vcount(net),
                        num_edges = ecount(net)
                    )
                ")?
        } else {
            R!("
                    library(ts2net)
                    library(igraph)

                    net <- tsnet_rn(x = {{x_r}}, radius = {{radius}}, time.lag = {{time_lag}})

                    if (is.null(net$embedding_dim)) {
                        embedding_dim <- 0
                    } else {
                        embedding_dim <- net$embedding_dim
                    }

                    edge_list <- get.edgelist(net)
                    if (nrow(edge_list) == 0) {
                        edge_list <- matrix(integer(0), ncol = 2)
                    }
                    edge_weights <- E(net)$weight
                    if (is.null(edge_weights)) {
                        edge_weights <- rep(1.0, ecount(net))
                    }

                    list(
                        edge_index = edge_list,
                        edge_weight = edge_weights,
                        embedding_dim = embedding_dim,
                        num_vertices = vcount(net),
                        num_edges = ecount(net)
                    )
                ")?
        };
        let edge_list = result.dollar("edge_index")?;
        let edge_weight = result.dollar("edge_weight")?;
        let embedding_dim = result.dollar("embedding_dim")?;

        let edge_index_rust = extract_edge_index_from_r(edge_list)?;
        let edge_weight_rust: Vec<f64> = Doubles::try_from(edge_weight)?
            .iter()
            .map(|r| r.inner())
            .collect();
        let embedding_dim_rust: f64 = Doubles::try_from(embedding_dim)?
            .iter()
            .next()
            .unwrap_or_else(|| 0.into())
            .inner() as f64;
        //
        // let edge_index_rust = vec![vec![0 as i64]];
        // let edge_weight_rust = vec![0.0];
        // let embedding_dim_rust = 0 as i64;
        Ok((edge_index_rust, edge_weight_rust, embedding_dim_rust as f64))
    }

    fn call_r_tsnet_rn(
        x: &[f64],
        radius: f32,
        embedding_dim: Option<i64>,
        time_lag: i64,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>, f64), String> {
        extendr_engine::start_r();
        println!("Calling R");

        let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let x_r = Doubles::from_values(&x_f64);

        let result = if let Some(dim) = embedding_dim {
            R!("
                    library(ts2net)
                    library(igraph)

                    net <- tsnet_rn(x = {{x_r}}, radius={{radius}}, embedding.dim = {{dim}}, time.lag = {{time_lag}})

                    if (is.null(net$embedding_dim)) {
                        embedding_dim <- 0
                    } else {
                        embedding_dim <- net$embedding_dim
                    }

                    edge_list <- get.edgelist(net)
                    if (nrow(edge_list) == 0) {
                        edge_list <- matrix(integer(0), ncol = 2)
                    }
                    edge_weights <- E(net)$weight
                    if (is.null(edge_weights)) {
                        edge_weights <- rep(1.0, ecount(net))
                    }

                    list(
                        edge_index = apply(edge_list),
                        edge_weight = edge_weights,
                        embedding_dim = embedding_dim,
                        num_vertices = vcount(net),
                        num_edges = ecount(net)
                    )
                ")?
        } else {
            R!("
                    library(ts2net)
                    library(igraph)

                    net <- tsnet_rn(x = {{x_r}}, radius = {{radius}}, time.lag = {{time_lag}})

                    if (is.null(net$embedding_dim)) {
                        embedding_dim <- 0
                    } else {
                        embedding_dim <- net$embedding_dim
                    }

                    edge_list <- get.edgelist(net)
                    if (nrow(edge_list) == 0) {
                        edge_list <- matrix(integer(0), ncol = 2)
                    }
                    edge_weights <- E(net)$weight
                    if (is.null(edge_weights)) {
                        edge_weights <- rep(1.0, ecount(net))
                    }

                    list(
                        edge_index = edge_list,
                        edge_weight = edge_weights,
                        embedding_dim = embedding_dim,
                        num_vertices = vcount(net),
                        num_edges = ecount(net)
                    )
                ")?
        };
        let edge_list = result.dollar("edge_index")?;
        let edge_weight = result.dollar("edge_weight")?;
        let embedding_dim = result.dollar("embedding_dim")?;

        let edge_index_rust = extract_edge_index_from_r(edge_list)?;
        let edge_weight_rust: Vec<f64> = Doubles::try_from(edge_weight)?
            .iter()
            .map(|r| r.inner())
            .collect();
        let embedding_dim_rust: f64 = Doubles::try_from(embedding_dim)?
            .iter()
            .next()
            .unwrap_or_else(|| 0.into())
            .inner() as f64;
        //
        // let edge_index_rust = vec![vec![0 as i64]];
        // let edge_weight_rust = vec![0.0];
        // let embedding_dim_rust = 0 as i64;
        Ok((edge_index_rust, edge_weight_rust, embedding_dim_rust as f64))
    }

    fn extract_edge_index_from_r(edge_list: Robj) -> Result<Vec<Vec<f64>>, String> {
        let matrix: RMatrix<Rfloat> = RMatrix::try_from(edge_list)
            .map_err(|e| format!("Failed to convert edge list: {:?}", e))?;
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();

        if ncols != 2 {
            return Err("Expected edge list to have 2 columns".to_string());
        }

        if nrows == 0 {
            return Ok(vec![vec![], vec![]]);
        }

        let mut src_vec = Vec::with_capacity(nrows);
        let mut dst_vec = Vec::with_capacity(nrows);

        for i in 0..nrows {
            let src: f64 = matrix[[i, 0]].inner() as f64 - 1.0f64;

            let dst: f64 = matrix[[i, 1]].inner() as f64 - 1.0f64;

            src_vec.push(src);
            dst_vec.push(dst);
        }

        Ok(vec![src_vec, dst_vec])
    }

    fn compare_edge_sets(rust_edges: &[Vec<i64>], r_edges: &Vec<Vec<f64>>) -> bool {
        if rust_edges.len() != 2 || r_edges.len() != 2 {
            return false;
        }

        let rust_edges_pairs: HashSet<(i64, i64)> = rust_edges[0]
            .iter()
            .zip(rust_edges[1].iter())
            .map(|(&src, &dst)| (src.min(dst), src.max(dst)))
            .collect();

        let r_edges_pairs: HashSet<(i64, i64)> = r_edges[0]
            .iter()
            .zip(r_edges[1].iter())
            .map(|(&src, &dst)| {
                let (min, max) = (src.min(dst), src.max(dst));
                (min as i64, max as i64)
            })
            .collect();
        let mut rust_edges_sorted: Vec<_> = rust_edges_pairs.iter().collect();
        let mut r_edges_sorted: Vec<_> = r_edges_pairs.iter().collect();
        rust_edges_sorted.sort();
        r_edges_sorted.sort();

        let intersection: HashSet<_> = rust_edges_pairs.intersection(&r_edges_pairs).collect();
        let union: HashSet<_> = rust_edges_pairs.union(&r_edges_pairs).collect();

        let jaccard = intersection.len() as f64 / union.len() as f64;
        println!("Jaccard similarity: {:.4}", jaccard);

        // for (r, s) in rust_edges_sorted.iter().zip(r_edges_sorted.iter()) {
        //     println!("Rust: {:?}, R: {:?}", r, s);
        // }
        // println!("{:?}", rust_edges_pairs);
        // println!("{:?}", r_edges_pairs);

        rust_edges_pairs == r_edges_pairs
    }

    fn tensor_to_matrix(tensor: &Tensor) -> Vec<Vec<i64>> {
        let shape = tensor.size();
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                let val: i64 = tensor.i((i as i64, j as i64)).int64_value(&[]);
                row.push(val);
            }
            result.push(row)
        }
        result
    }

    #[test]
    fn test_r_vs_rust_comparison() {
        let test_configs = vec![
            TestConfig::new("simple_sine", test_data::sine_wave(500, 1.0, 1.0), 0.5),
            // .with_embedding_dim(5),
            TestConfig::new(
                "logistic_chaos",
                test_data::logistic_map(500, 3.7, 0.4),
                0.15,
            ),
            // .with_embedding_dim(5),
            TestConfig::new("linear_trend", test_data::linear_trend(), 1.5), //.with_embedding_dim(5),
        ];

        for config in test_configs {
            println!("Comparing R vs Rust for: {}", config.name);

            // Get Rust result
            let rust_start = std::time::Instant::now();
            let rust_result = recurrence_graph_from_slice(
                &config.data,
                config.radius,
                config.embedding_dim,
                config.time_lag,
            )
            .unwrap();
            let rust_time = rust_start.elapsed();

            // Get R result
            let r_start = std::time::Instant::now();
            let r_result = call_r_tsnet_rn(
                &config.data,
                config.radius,
                config.embedding_dim,
                config.time_lag,
            )
            .unwrap();
            let r_time = r_start.elapsed();

            println!("Rust time {:?} R time {:?}", rust_time, r_time);

            // Compare embedding dimensions
            let dim_match = rust_result.embedding_dim as f64 == r_result.2;
            assert!(
                dim_match,
                "Embedding dimension mismatch for {}: Rust={}, R={}",
                config.name, rust_result.embedding_dim, r_result.2
            );
            println!(
                "Embedding dim match {:?} == {:?}",
                rust_result.embedding_dim, r_result.2
            );

            // Compare edge counts
            let rust_edge_count = rust_result.edge_index.size()[1] as usize;
            let r_edge_count = r_result.1.len();

            // Allow for small difference due to numerical precision
            let edge_count_diff = (rust_edge_count as i64 - r_edge_count as i64).abs();
            println!("Edges rust {:?} vs R {:?}", rust_edge_count, r_edge_count);
            // assert!(
            //     edge_count_diff <= 2,
            //     "Large edge count difference for {}: Rust={}, R={}",
            //     config.name,
            //     rust_edge_count,
            //     r_edge_count
            // );

            // Compare edge structures (if counts are similar)
            if edge_count_diff == 0 {
                let rust_edges = tensor_to_matrix(&rust_result.edge_index);
                let edges_match = compare_edge_sets(&rust_edges, &r_result.0);
                // assert!(
                //     edges_match,
                //     "Edge structure don't match for {} with {:?}",
                //     config.name, edges_match
                // );
                println!("{edges_match}")
            }
        }
    }
}

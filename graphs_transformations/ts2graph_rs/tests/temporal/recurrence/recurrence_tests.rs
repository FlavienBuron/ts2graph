mod recurrence_tests {
    use extendr_api::{
        Attributes, R, RMatrix, Robj,
        scalar::{Rfloat, Rint, Scalar},
    };
    use std::collections::HashSet;
    use ts2graph_rs::graph::temporal::recurrence::{
        EmbeddingConfig, TsNet, estimate_embedding_dim, get_cao_parameters, recurrence_graph_rs,
    };

    // Test structure to hold both R and Rust results
    #[derive(Debug)]
    struct RecurrenceComparisonResult {
        r_edges: Vec<(usize, usize)>,
        rust_edges: Vec<(usize, usize)>,
        r_weights: Vec<f64>,
        rust_weights: Vec<f64>,
        edge_match_count: usize,
        total_r_edges: usize,
        total_rust_edges: usize,
        edge_differences: EdgeDifferences,
        weight_statistics: WeightStatistics,
        parameters_match: ParametersMatch,
    }

    #[derive(Debug)]
    struct EdgeDifferences {
        missing_in_rust: Vec<(usize, usize)>,
        extra_in_rust: Vec<(usize, usize)>,
    }

    #[derive(Debug)]
    struct WeightStatistics {
        weight_correlation: f64,
        weight_mae: f64,  // Mean Absolute Error
        weight_rmse: f64, // Root Mean Square Error
        matching_edges_weight_diff: Vec<f64>,
    }

    #[derive(Debug)]
    struct ParametersMatch {
        embedding_dim: bool,
        time_lag: bool,
        radius: bool,
    }

    // Helper function to load and execute R script with ts2net functionality
    fn setup_r_environment() -> Result<(), Box<dyn std::error::Error>> {
        extendr_engine::start_r();
        R!("
            # Load required libraries
            if (!require('ts2net', quietly = TRUE)) {
                stop('ts2net package not found')
            } 
            if (!require('nonlinearTseries', quietly = TRUE)) {
                stop('nonlinearTseries package not found')
            }
        ")?;
        Ok(())
    }

    // Helper function to call R tsnet_rn function and extract results
    fn call_r_tsnet(
        x: &[f64],
        radius: f64,
        embedding_dim: Option<usize>,
        time_lag: usize,
    ) -> Result<(Vec<(usize, usize)>, Vec<f64>, usize, usize, f64), Box<dyn std::error::Error>>
    {
        if x.is_empty() {
            return Err("Time series data is empty".into());
        }

        // Convert Rust data to R format
        let x_r = extendr_api::Doubles::try_from(x.iter().copied().collect::<Vec<_>>())?;

        // Prepare R call parameters
        let result = if let Some(dim) = embedding_dim {
            R!("ts2net::tsnet_rn({{x_r}}, {{radius}}, {{dim}}, {{time_lag}})")?
        } else {
            R!("ts2net::tsnet_rn({{x_r}}, {{radius}}, time.lag={{time_lag}})")?
        };

        // Extract edge list
        let edge_list_result = R!("as_edgelist({{result.clone()}}, names=FALSE)")?;

        let edge_matrix: RMatrix<Rfloat> = RMatrix::try_from(edge_list_result)?;

        let dims = edge_matrix.dim();
        let nrows = dims[0];
        let ncols = dims[1];

        if ncols != 2 {
            return Err("Edge list must have 2 columns".into());
        }

        // Convert to Vec<(usize, usize)>
        let mut edges = Vec::with_capacity(nrows);
        for i in 0..nrows {
            // R is 1-based, not 0-based
            let from = edge_matrix[[i, 0]].inner() as usize - 1;
            let to = edge_matrix[[i, 1]].inner() as usize - 1;
            edges.push((from.min(to) as usize, from.max(to) as usize));
        }
        edges.sort_unstable();
        edges.dedup();

        // Extract edge weights - handle case where graph might have no edges
        let weights = if !edges.is_empty() {
            let weights_result = R!("E({{result.clone()}})$weight")?;
            if let Some(weight_vec) = weights_result.as_real_vector() {
                weight_vec.iter().copied().collect()
            } else {
                // If no explicit weights, igraph typically uses 1.0 as default
                vec![1.0; edges.len()]
            }
        } else {
            Vec::new()
        };

        // Extract parameters
        let embedding_dim_result = R!("{{result.clone()}}$embedding_dim")?;
        let actual_embedding_dim = embedding_dim_result
            .as_integer_vector()
            .and_then(|v| v.first().copied())
            .unwrap_or(2) as usize;

        let time_lag_result = R!("{{result.clone()}}$time_lag")?;
        let actual_time_lag = time_lag_result
            .as_integer_vector()
            .and_then(|v| v.first().copied())
            .unwrap_or(1) as usize;

        let radius_result = R!("{{result.clone()}}$radius")?;
        let actual_radius = radius_result
            .as_real_vector()
            .and_then(|v| v.first().copied())
            .unwrap_or(radius);

        Ok((
            edges,
            weights,
            actual_embedding_dim,
            actual_time_lag,
            actual_radius,
        ))
    }

    // Helper: R call for estimate_embedding_dim
    fn call_r_estimate_embedding_dim(
        time_series: &[f64],
        time_lag: usize,
        max_embedding_dim: usize,
        threshold: f64,
        max_relative_change: f64,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        // Convert Rust array to R vector
        let ts_r = extendr_api::Doubles::try_from(time_series.to_vec())?;

        // Call the private R function using interpolate macro
        let robj = R!(r#"
            nonlinearTseries::estimateEmbeddingDim(
                time.series = {{ts_r}},
                time.lag = {{time_lag}},
                max.embedding.dim = {{max_embedding_dim}},
                threshold = {{threshold}},
                max.relative.change = {{max_relative_change}}
            )
        "#)?;

        Ok(robj.as_real().map(|x| x as usize))
    }

    // Helper: R call for get_cao_parameters (Cao's E and E1)
    fn call_r_cao_parameters(
        data: &[f64],
        embedding_dim: usize,
        time_lag: usize,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let data_r = extendr_api::Doubles::try_from(data.iter().copied().collect::<Vec<_>>())?;
        let robj = R!("nonlinearTseries:::getCaoParameters(
            data={{data_r}},
            m={{embedding_dim}},
            time.lag={{time_lag}}
        )")?;
        // The R function returns a list with $E and $E1, both vectors (length m+1)
        let e = R!("{{robj.clone()}}$E")?.as_real().unwrap_or(0.0);
        let e1 = R!("{{robj}}$E.star")?.as_real().unwrap_or(0.0);
        Ok((e, e1))
    }

    // Helper function to generate test time series data
    fn generate_test_timeseries(n_points: usize, series_type: &str, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        match series_type {
            "random" => (0..n_points).map(|_| rng.random_range(-1.0..1.0)).collect(),
            "sine" => (0..n_points)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
                .collect(),
            "lorenz_x" => {
                // Simplified Lorenz-like chaotic series (just x component approximation)
                let mut series = vec![0.1]; // Initial condition
                for i in 1..n_points {
                    let prev = series[i - 1];
                    let next = prev + 0.1 * (10.0 * (0.5 - prev) + rng.random_range(-0.1..0.1));
                    series.push(next);
                }
                series
            }
            "ar1" => {
                // AR(1) process: x_t = 0.7 * x_{t-1} + noise
                let mut series = vec![rng.random_range(-1.0..1.0)];
                for i in 1..n_points {
                    let noise = rng.random_range(-0.5..0.5);
                    let next = 0.7 * series[i - 1] + noise;
                    series.push(next);
                }
                series
            }
            _ => generate_test_timeseries(n_points, "random", seed),
        }
    }

    // Function to compare recurrence graphs
    fn compare_recurrence_graphs(
        r_edges: Vec<(usize, usize)>,
        r_weights: Vec<f64>,
        r_embedding_dim: usize,
        r_time_lag: usize,
        r_radius: f64,
        rust_result: &TsNet,
        expected_embedding_dim: Option<usize>,
        expected_time_lag: usize,
        expected_radius: f64,
    ) -> RecurrenceComparisonResult {
        // Convert Rust edge_index (flat Vec<i64>) into normalized edges
        let rust_edges: Vec<(usize, usize)> = rust_result
            .edge_index
            .chunks_exact(2)
            .map(|pair| {
                let from = pair[0] as usize;
                let to = pair[1] as usize;
                (from.min(to), from.max(to))
            })
            .collect();

        let rust_weights = rust_result.edge_weight.clone();

        // Deduplicate edges using HashSet
        let r_edge_set: HashSet<(usize, usize)> = r_edges.iter().cloned().collect();
        let rust_edge_set: HashSet<(usize, usize)> = rust_edges.iter().cloned().collect();

        // Missing and extra edges
        let missing_in_rust: Vec<(usize, usize)> =
            r_edge_set.difference(&rust_edge_set).cloned().collect();
        let extra_in_rust: Vec<(usize, usize)> =
            rust_edge_set.difference(&r_edge_set).cloned().collect();

        // Common edges (intersection)
        let common_edges: Vec<(usize, usize)> =
            r_edge_set.intersection(&rust_edge_set).cloned().collect();
        let edge_match_count = common_edges.len();

        // Weight comparison for common edges
        let mut weight_correlation = 0.0;
        let mut weight_mae = 0.0;
        let mut weight_rmse = 0.0;
        let mut matching_edges_weight_diff = Vec::new();

        if !common_edges.is_empty()
            && r_weights.len() == r_edges.len()
            && rust_weights.len() == rust_edges.len()
        {
            let mut r_common_weights = Vec::with_capacity(common_edges.len());
            let mut rust_common_weights = Vec::with_capacity(common_edges.len());

            for edge in &common_edges {
                let r_idx = r_edges.iter().position(|e| e == edge).unwrap();
                let rust_idx = rust_edges.iter().position(|e| e == edge).unwrap();
                r_common_weights.push(r_weights[r_idx]);
                rust_common_weights.push(rust_weights[rust_idx]);
                matching_edges_weight_diff.push((r_weights[r_idx] - rust_weights[rust_idx]).abs());
            }

            // Compute correlation
            let r_mean: f64 = r_common_weights.iter().sum::<f64>() / r_common_weights.len() as f64;
            let rust_mean: f64 =
                rust_common_weights.iter().sum::<f64>() / rust_common_weights.len() as f64;

            let numerator: f64 = r_common_weights
                .iter()
                .zip(rust_common_weights.iter())
                .map(|(&r, &rust)| (r - r_mean) * (rust - rust_mean))
                .sum();

            let r_ss: f64 = r_common_weights.iter().map(|&r| (r - r_mean).powi(2)).sum();
            let rust_ss: f64 = rust_common_weights
                .iter()
                .map(|&rust| (rust - rust_mean).powi(2))
                .sum();

            if r_ss > 0.0 && rust_ss > 0.0 {
                weight_correlation = numerator / (r_ss * rust_ss).sqrt();
            }

            // MAE and RMSE
            weight_mae = matching_edges_weight_diff.iter().sum::<f64>()
                / matching_edges_weight_diff.len() as f64;
            weight_rmse = (matching_edges_weight_diff
                .iter()
                .map(|&d| d.powi(2))
                .sum::<f64>()
                / matching_edges_weight_diff.len() as f64)
                .sqrt();
        }

        // Compare parameters
        let expected_dim = expected_embedding_dim.unwrap_or(r_embedding_dim);
        let parameters_match = ParametersMatch {
            embedding_dim: rust_result.embedding_dim == expected_dim,
            time_lag: rust_result.time_lag == expected_time_lag,
            radius: (rust_result.radius - expected_radius).abs() < 1e-10,
        };

        RecurrenceComparisonResult {
            r_edges,
            rust_edges,
            r_weights,
            rust_weights,
            edge_match_count,
            total_r_edges: r_edge_set.len(),
            total_rust_edges: rust_edge_set.len(),
            edge_differences: EdgeDifferences {
                missing_in_rust,
                extra_in_rust,
            },
            weight_statistics: WeightStatistics {
                weight_correlation,
                weight_mae,
                weight_rmse,
                matching_edges_weight_diff,
            },
            parameters_match,
        }
    }

    #[test]
    fn test_setup_r_env() {
        println!("Setting up R environment...");
        match setup_r_environment() {
            Ok(_) => println!("R environment setup successful"),
            Err(e) => panic!("R environment setup failed: {}", e),
        }
    }

    #[test]
    fn test_get_cao_parameters_vs_r() {
        setup_r_environment().unwrap();
        let ts = generate_test_timeseries(100, "lorenz_x", 42);
        let embedding_dim = 2;
        let time_lag = 1;

        // Rust computation (deterministic, no noise added)
        let rust = get_cao_parameters(&ts, embedding_dim, time_lag).unwrap();

        // R computation
        let r = call_r_cao_parameters(&ts, embedding_dim, time_lag).unwrap();

        println!("Rust CaoParameters: e={}, e_star={}", rust.e, rust.e_star);
        println!("R CaoParameters:   e={}, e_star={}", r.0, r.1);

        // Use small tolerance for numerical comparison
        let tol = 1e-3;
        assert!(
            (rust.e - r.0).abs() < tol,
            "E mismatch: rust={} R={}",
            rust.e,
            r.0
        );
        assert!(
            (rust.e_star - r.1).abs() < tol,
            "E* mismatch: rust={} R={}",
            rust.e_star,
            r.1
        );
    }

    #[test]
    fn test_estimate_embedding_dim_vs_r() {
        setup_r_environment().unwrap();
        let ts = generate_test_timeseries(100, "lorenz_x", 42);
        let config = EmbeddingConfig {
            time_lag: 1,
            max_embedding_dim: 10,
            threshold: 0.95,
            max_relative_change: 0.10,
            std_noise: None,
            number_points: Some(100),
        };
        let rust_dim = estimate_embedding_dim(&ts, config.clone()).unwrap();
        let r_dim = call_r_estimate_embedding_dim(
            &ts,
            config.time_lag,
            config.max_embedding_dim,
            config.threshold,
            config.max_relative_change,
        )
        .unwrap();
        println!("Rust Dim = {:?}", rust_dim);
        println!("R Dim = {:?}", r_dim);
        assert_eq!(rust_dim, r_dim, "Rust and R embedding dimension differ");
    }

    #[test]
    fn test_small_timeseries_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        // Generate small test time series
        let x = generate_test_timeseries(50, "sine", 42);
        let radius = 0.3;
        let embedding_dim = Some(3);
        let time_lag = 1;

        // Call R function
        let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
            call_r_tsnet(&x, radius, embedding_dim, time_lag)?;

        // Call Rust function
        let rust_result = recurrence_graph_rs(&x, radius, embedding_dim, time_lag, false)?;

        // Compare results
        let comparison = compare_recurrence_graphs(
            r_edges,
            r_weights,
            r_embedding_dim,
            r_time_lag,
            r_radius,
            &rust_result,
            embedding_dim,
            time_lag,
            radius,
        );

        println!("Small time series test results:");
        println!(
            "  R edges: {}, Rust edges: {}",
            comparison.total_r_edges, comparison.total_rust_edges
        );
        println!("  Matching edges: {}", comparison.edge_match_count);
        println!(
            "  Edge match rate: {:.2}%",
            100.0 * comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64
        );
        println!(
            "  Missing in Rust: {}, Extra in Rust: {}",
            comparison.edge_differences.missing_in_rust.len(),
            comparison.edge_differences.extra_in_rust.len()
        );
        println!(
            "  Weight correlation: {:.4}",
            comparison.weight_statistics.weight_correlation
        );
        println!(
            "  Weight MAE: {:.6}",
            comparison.weight_statistics.weight_mae
        );
        println!(
            "  Parameters match - Dim: {}, Lag: {}, Radius: {}",
            comparison.parameters_match.embedding_dim,
            comparison.parameters_match.time_lag,
            comparison.parameters_match.radius
        );

        // Assert reasonable match rate
        let edge_match_rate =
            comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64;
        assert!(
            edge_match_rate >= 0.90,
            "Edge match rate too low: {:.2}%",
            edge_match_rate * 100.0
        );
        assert!(
            comparison.parameters_match.embedding_dim,
            "Embedding dimension mismatch"
        );
        assert!(comparison.parameters_match.time_lag, "Time lag mismatch");
        assert!(comparison.parameters_match.radius, "Radius mismatch");

        Ok(())
    }

    #[test]
    fn test_different_series_types() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let series_types = vec!["random", "sine", "ar1"];
        let radius = 0.25;
        let embedding_dim = Some(2);
        let time_lag = 1;

        for series_type in series_types {
            println!("Testing series type: {}", series_type);

            let x = generate_test_timeseries(80, series_type, 123);

            // Call R function
            let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
                call_r_tsnet(&x, radius, embedding_dim, time_lag)?;

            // Call Rust function
            let rust_result = recurrence_graph_rs(&x, radius, embedding_dim, time_lag, false)?;

            // Compare results
            let comparison = compare_recurrence_graphs(
                r_edges,
                r_weights,
                r_embedding_dim,
                r_time_lag,
                r_radius,
                &rust_result,
                embedding_dim,
                time_lag,
                radius,
            );

            let edge_match_rate =
                comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64;
            println!("  Edge match rate: {:.2}%", edge_match_rate * 100.0);
            println!(
                "  Weight correlation: {:.4}",
                comparison.weight_statistics.weight_correlation
            );

            assert!(
                edge_match_rate >= 0.85,
                "Edge match rate too low for {}: {:.2}%",
                series_type,
                edge_match_rate * 100.0
            );
        }

        Ok(())
    }

    #[test]
    fn test_different_parameters() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let x = generate_test_timeseries(100, "lorenz_x", 456);

        // Test different radii
        let radii = vec![0.1, 0.2, 0.4];
        for radius in radii {
            println!("Testing radius: {}", radius);

            let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
                call_r_tsnet(&x, radius, Some(3), 1)?;
            let rust_result = recurrence_graph_rs(&x, radius, Some(3), 1, false)?;

            let comparison = compare_recurrence_graphs(
                r_edges,
                r_weights,
                r_embedding_dim,
                r_time_lag,
                r_radius,
                &rust_result,
                Some(3),
                1,
                radius,
            );

            let edge_match_rate =
                comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64;
            println!("  Edge match rate: {:.2}%", edge_match_rate * 100.0);
            assert!(
                edge_match_rate >= 0.80,
                "Match rate too low for radius {}: {:.2}%",
                radius,
                edge_match_rate * 100.0
            );
        }

        // Test different embedding dimensions
        let dims = vec![2, 3, 4];
        for dim in dims {
            println!("Testing embedding dimension: {}", dim);

            let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
                call_r_tsnet(&x, 0.3, Some(dim), 1)?;
            let rust_result = recurrence_graph_rs(&x, 0.3, Some(dim), 1, false)?;

            let comparison = compare_recurrence_graphs(
                r_edges,
                r_weights,
                r_embedding_dim,
                r_time_lag,
                r_radius,
                &rust_result,
                Some(dim),
                1,
                0.3,
            );

            let edge_match_rate =
                comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64;
            println!("  Edge match rate: {:.2}%", edge_match_rate * 100.0);
            assert!(
                edge_match_rate >= 0.80,
                "Match rate too low for dim {}: {:.2}%",
                dim,
                edge_match_rate * 100.0
            );
        }

        Ok(())
    }

    #[test]
    fn test_automatic_embedding_dimension() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let x = generate_test_timeseries(100, "sine", 789);
        let radius = 0.3;
        let time_lag = 1;

        // Test with automatic embedding dimension estimation
        let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
            call_r_tsnet(&x, radius, None, time_lag)?;
        let rust_result = recurrence_graph_rs(&x, radius, None, time_lag, false)?;

        let comparison = compare_recurrence_graphs(
            r_edges,
            r_weights,
            r_embedding_dim,
            r_time_lag,
            r_radius,
            &rust_result,
            None,
            time_lag,
            radius,
        );

        println!("Automatic embedding dimension test:");
        println!(
            "  R estimated dim: {}, Rust estimated dim: {}",
            r_embedding_dim, rust_result.embedding_dim
        );
        println!(
            "  Edge match rate: {:.2}%",
            100.0 * comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64
        );
        println!(
            "  Parameters match - Dim: {}",
            comparison.parameters_match.embedding_dim
        );

        // The embedding dimensions should match (or be reasonable if they differ)
        if !comparison.parameters_match.embedding_dim {
            println!(
                "  WARNING: Embedding dimensions differ - R: {}, Rust: {}",
                r_embedding_dim, rust_result.embedding_dim
            );
        }

        Ok(())
    }

    #[test]
    fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let x = generate_test_timeseries(200, "ar1", 999);
        let radius = 0.25;
        let embedding_dim = Some(3);
        let time_lag = 1;

        // Time R function
        let start = std::time::Instant::now();
        let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
            call_r_tsnet(&x, radius, embedding_dim, time_lag)?;
        let r_duration = start.elapsed();

        // Time Rust function
        let start = std::time::Instant::now();
        let rust_result = recurrence_graph_rs(&x, radius, embedding_dim, time_lag, false)?;
        let rust_duration = start.elapsed();

        let comparison = compare_recurrence_graphs(
            r_edges,
            r_weights,
            r_embedding_dim,
            r_time_lag,
            r_radius,
            &rust_result,
            embedding_dim,
            time_lag,
            radius,
        );

        println!("Performance comparison for n=200:");
        println!("  R execution time: {:?}", r_duration);
        println!("  Rust execution time: {:?}", rust_duration);
        println!(
            "  Speedup: {:.2}x",
            r_duration.as_nanos() as f64 / rust_duration.as_nanos().max(1) as f64
        );
        println!(
            "  Edge match rate: {:.2}%",
            100.0 * comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64
        );
        println!(
            "  Weight correlation: {:.4}",
            comparison.weight_statistics.weight_correlation
        );

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        // Test case 1: Very small time series
        let x_small = generate_test_timeseries(20, "random", 111);
        let rust_result = recurrence_graph_rs(&x_small, 0.5, Some(2), 1, false)?;
        println!(
            "Small series (n=20) produced {} edges",
            rust_result.edge_index.len() / 2
        );

        // Test case 2: Very sparse graph (small radius)
        let x = generate_test_timeseries(50, "random", 222);
        let rust_result = recurrence_graph_rs(&x, 0.01, Some(2), 1, false)?;
        println!(
            "Sparse graph (r=0.01) produced {} edges",
            rust_result.edge_index.len() / 2
        );

        // Test case 3: Very dense graph (large radius)
        let rust_result = recurrence_graph_rs(&x, 2.0, Some(2), 1, false)?;
        println!(
            "Dense graph (r=2.0) produced {} edges",
            rust_result.edge_index.len() / 2
        );

        Ok(())
    }

    #[test]
    fn test_scalability_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let test_cases = vec![(500, "sine", 0.25), (1000, "ar1", 0.2)];

        for (size, series_type, radius) in test_cases {
            println!(
                "\n=== Testing scalability: size={}, type={}, radius={:.2} ===",
                size, series_type, radius
            );

            let x = generate_test_timeseries(size, series_type, 333);
            let embedding_dim = Some(3);
            let time_lag = 1;

            // Time R execution
            let r_start = std::time::Instant::now();
            let (r_edges, r_weights, r_embedding_dim, r_time_lag, r_radius) =
                call_r_tsnet(&x, radius, embedding_dim, time_lag)?;
            let r_duration = r_start.elapsed();

            // Time Rust execution
            let rust_start = std::time::Instant::now();
            let rust_result = recurrence_graph_rs(&x, radius, embedding_dim, time_lag, false)?;
            let rust_duration = rust_start.elapsed();

            let comparison = compare_recurrence_graphs(
                r_edges,
                r_weights,
                r_embedding_dim,
                r_time_lag,
                r_radius,
                &rust_result,
                embedding_dim,
                time_lag,
                radius,
            );

            println!("  R execution time: {:?}", r_duration);
            println!("  Rust execution time: {:?}", rust_duration);
            println!(
                "  Speedup: {:.2}x",
                r_duration.as_nanos() as f64 / rust_duration.as_nanos().max(1) as f64
            );
            println!(
                "  R edges: {}, Rust edges: {}",
                comparison.total_r_edges, comparison.total_rust_edges
            );
            println!(
                "  Edge match rate: {:.2}%",
                100.0 * comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64
            );
            println!(
                "  Weight correlation: {:.4}",
                comparison.weight_statistics.weight_correlation
            );

            let edge_match_rate =
                comparison.edge_match_count as f64 / comparison.total_r_edges.max(1) as f64;
            assert!(
                edge_match_rate >= 0.75,
                "Edge match rate too low for size {}: {:.2}%",
                size,
                edge_match_rate * 100.0
            );
        }

        Ok(())
    }
}


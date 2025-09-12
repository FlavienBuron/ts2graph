mod neighbor_search_tests {
    // use extendr_api::prelude::*;
    use extendr_api::{List, R, RMatrix};
    use ts2graph_rs::graph::temporal::Takens;
    use ts2graph_rs::graph::temporal::neighbor_search::NeighborSearch;

    // Test structure to hold both R and Rust results
    #[derive(Debug)]
    struct ComparisonResult {
        r_result: Vec<Vec<i32>>,
        rust_result: Vec<Vec<i32>>,
        match_count: usize,
        total_points: usize,
        differences: Vec<PointDifference>,
    }

    #[derive(Debug)]
    struct PointDifference {
        point_index: usize,
        r_neighbors: Vec<i32>,
        rust_neighbors: Vec<i32>,
        missing_in_rust: Vec<i32>,
        extra_in_rust: Vec<i32>,
    }

    // Helper function to load and execute R script
    fn setup_r_environment() -> Result<(), Box<dyn std::error::Error>> {
        extendr_engine::start_r();
        R!("
        # Load required libraries
        if (!require('nonlinearTseries', quietly = TRUE)) {
            stop('nonlinearTseries package not found')
        }
        ")?;
        Ok(())
    }

    // Helper function to call R function and extract results
    fn call_r_find_neighbors(
        takens: &[&[f64]],
        radius: f64,
        number_boxes: Option<usize>,
    ) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>> {
        if takens.is_empty() {
            return Err("Takens data is empty".into());
        }

        let n_rows = takens.len();
        let n_cols = takens[0].len();

        if !takens.iter().all(|rows| rows.len() == n_cols) {
            return Err("Takens data has inconsistent row lengths".into());
        }

        // Convert Rust data to R format
        let takens_r = RMatrix::new_matrix(n_rows, n_cols, |r, c| takens[r][c]);

        // Prepare R call parameters
        let result = if let Some(boxes) = number_boxes {
            R!("findAllNeighbours({{takens_r}}, {{radius}}, {{boxes}})")?
        } else {
            R!("findAllNeighbours({{takens_r}}, {{radius}})")?
        };

        // Convert R result back to Rust format
        let r_list: List = List::try_from(result).map_err(|_| "R result is not a list")?;
        let rust_result: Vec<Vec<i32>> = r_list
            .iter()
            .map(|(_, robj)| {
                robj.as_integer_vector()
                    .map(|ints| ints.iter().map(|&x| x - 1).collect())
                    .unwrap_or_else(Vec::new)
            })
            .collect();
        Ok(rust_result)
    }

    // Helper function to generate test data
    fn generate_test_takens(n_points: usize, dimension: usize, seed: u64) -> Takens {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut takens = Vec::new();

        for _ in 0..n_points {
            let mut point = Vec::new();
            for _ in 0..dimension {
                // Generate points in [0, 1] range
                point.push(rng.random_range(0.0..=1.0));
            }
            takens.push(point);
        }

        Takens {
            owned_takens: takens,
        }
    }

    // Function to compare two neighbor lists
    fn compare_neighbor_lists(
        r_result: &[Vec<i32>],
        rust_result: Vec<Vec<i32>>,
    ) -> ComparisonResult {
        let mut differences = Vec::new();
        let mut match_count = 0;
        let total_points = r_result.len();

        for (i, (r_neighbors, rust_neighbors)) in
            r_result.iter().zip(rust_result.iter()).enumerate()
        {
            let r_set: std::collections::HashSet<_> = r_neighbors.iter().collect();
            let rust_set: std::collections::HashSet<_> = rust_neighbors.iter().collect();

            if r_set == rust_set {
                match_count += 1;
            } else {
                let missing_in_rust: Vec<i32> = r_neighbors
                    .iter()
                    .filter(|&x| !rust_set.contains(x))
                    .cloned()
                    .collect();

                let extra_in_rust: Vec<i32> = rust_neighbors
                    .iter()
                    .filter(|&x| !r_set.contains(x))
                    .cloned()
                    .collect();

                differences.push(PointDifference {
                    point_index: i,
                    r_neighbors: r_neighbors.clone(),
                    rust_neighbors: rust_neighbors.clone(),
                    missing_in_rust,
                    extra_in_rust,
                });
            }
        }

        ComparisonResult {
            r_result: r_result.to_vec(),
            rust_result: rust_result.to_vec(),
            match_count,
            total_points,
            differences,
        }
    }

    #[test]
    fn test_setup_r_env() -> () {
        println!("run");
        _ = setup_r_environment();
        println!("Setup");
    }

    #[test]
    fn test_small_dataset_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        println!("test");

        // Generate small test dataset
        let takens = generate_test_takens(10, 2, 42);
        let takens_views = takens.views();
        let radius = 0.3;
        let number_boxes = None;

        // Call R function
        let r_result = call_r_find_neighbors(&takens.views(), radius, number_boxes)?;

        // Call Rust function
        let mut rust_finder = NeighborSearch::new(&takens_views, radius, number_boxes)?;
        let rust_result = rust_finder.find_all_neighbors()?;
        let rust_result_i32: Vec<Vec<i32>> = rust_result
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as i32).collect())
            .collect();

        // Compare results
        let comparison = compare_neighbor_lists(&r_result, rust_result_i32);

        println!("Small dataset test results:");
        println!(
            "  Matching points: {}/{}",
            comparison.match_count, comparison.total_points
        );
        println!(
            "  Match rate: {:.2}%",
            100.0 * comparison.match_count as f64 / comparison.total_points as f64
        );

        if !comparison.differences.is_empty() {
            println!("  First 5 differences:");
            for (_i, diff) in comparison.differences.iter().take(5).enumerate() {
                println!(
                    "    Point {}: R={:?}, Rust={:?}",
                    diff.point_index, diff.r_neighbors, diff.rust_neighbors
                );
            }
        }

        // Assert high match rate (adjust threshold as needed)
        let match_rate = comparison.match_count as f64 / comparison.total_points as f64;
        assert!(
            match_rate >= 0.95,
            "Match rate too low: {:.2}%",
            match_rate * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_medium_dataset_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        // Generate medium test dataset
        let takens = generate_test_takens(100, 3, 123);
        let takens_views = takens.views();
        let radius = 0.2;
        let number_boxes = Some(50);

        // Call R function
        let r_result = call_r_find_neighbors(&takens.views(), radius, number_boxes)?;

        // Call Rust function
        let mut rust_finder = NeighborSearch::new(&takens_views, radius, number_boxes)?;
        let rust_result = rust_finder.find_all_neighbors()?;
        let rust_result_i32: Vec<Vec<i32>> = rust_result
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as i32).collect())
            .collect();

        // Compare results
        let comparison = compare_neighbor_lists(&r_result, rust_result_i32);

        println!("Medium dataset test results:");
        println!(
            "  Matching points: {}/{}",
            comparison.match_count, comparison.total_points
        );
        println!(
            "  Match rate: {:.2}%",
            100.0 * comparison.match_count as f64 / comparison.total_points as f64
        );

        // Statistical analysis of differences
        if !comparison.differences.is_empty() {
            let total_missing: usize = comparison
                .differences
                .iter()
                .map(|d| d.missing_in_rust.len())
                .sum();
            let total_extra: usize = comparison
                .differences
                .iter()
                .map(|d| d.extra_in_rust.len())
                .sum();

            println!("  Total missing neighbors in Rust: {}", total_missing);
            println!("  Total extra neighbors in Rust: {}", total_extra);
        }

        let match_rate = comparison.match_count as f64 / comparison.total_points as f64;
        assert!(
            match_rate >= 0.90,
            "Match rate too low: {:.2}%",
            match_rate * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_different_radii_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let takens = generate_test_takens(50, 2, 456);
        let takens_views = takens.views();

        let radii = vec![0.1, 0.2, 0.3, 0.5];

        for radius in radii {
            println!("Testing radius: {}", radius);

            // Call R function
            let r_result = call_r_find_neighbors(&takens.views(), radius, None)?;

            // Call Rust function
            let mut rust_finder = NeighborSearch::new(&takens_views, radius, None)?;
            let rust_result = rust_finder.find_all_neighbors()?;
            let rust_result_i32: Vec<Vec<i32>> = rust_result
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as i32).collect())
                .collect();

            // Compare results
            let comparison = compare_neighbor_lists(&r_result, rust_result_i32);
            let match_rate = comparison.match_count as f64 / comparison.total_points as f64;

            println!("  Match rate: {:.2}%", match_rate * 100.0);
            assert!(
                match_rate >= 0.85,
                "Match rate too low for radius {}: {:.2}%",
                radius,
                match_rate * 100.0
            );
        }

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        // Test case 1: Very small radius (should have few/no neighbors)
        let takens = generate_test_takens(20, 2, 789);
        let takens_views = takens.views();
        let small_radius = 0.01;

        let r_result = call_r_find_neighbors(&takens.views(), small_radius, None)?;
        let mut rust_finder = NeighborSearch::new(&takens_views, small_radius, None)?;
        let rust_result = rust_finder.find_all_neighbors()?;
        let rust_result_i32: Vec<Vec<i32>> = rust_result
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as i32).collect())
            .collect();

        let comparison = compare_neighbor_lists(&r_result, rust_result_i32);
        println!(
            "Small radius test - Match rate: {:.2}%",
            100.0 * comparison.match_count as f64 / comparison.total_points as f64
        );

        // Test case 2: Very large radius (should have many neighbors)
        let large_radius = 2.0;

        let r_result = call_r_find_neighbors(&takens.views(), large_radius, None)?;
        let mut rust_finder = NeighborSearch::new(&takens_views, large_radius, None)?;
        let rust_result = rust_finder.find_all_neighbors()?;
        let rust_result_i32: Vec<Vec<i32>> = rust_result
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as i32).collect())
            .collect();

        let comparison = compare_neighbor_lists(&r_result, rust_result_i32);
        println!(
            "Large radius test - Match rate: {:.2}%",
            100.0 * comparison.match_count as f64 / comparison.total_points as f64
        );

        Ok(())
    }

    #[test]
    fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        let takens = generate_test_takens(200, 3, 999);
        let takens_views = takens.views();
        let radius = 0.25;

        // Time R function
        let start = std::time::Instant::now();
        let r_result = call_r_find_neighbors(&takens.views(), radius, None)?;
        let r_duration = start.elapsed();

        // Time Rust function
        let start = std::time::Instant::now();
        let mut rust_finder = NeighborSearch::new(&takens_views, radius, None)?;
        let rust_result = rust_finder.find_all_neighbors()?;
        let rust_duration = start.elapsed();
        let rust_result_i32: Vec<Vec<i32>> = rust_result
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as i32).collect())
            .collect();

        println!("Performance comparison for [200, 3]");
        println!("  R execution time: {:?}", r_duration);
        println!("  Rust execution time: {:?}", rust_duration);
        println!(
            "  Speedup: {:.2}x",
            r_duration.as_nanos() as f64 / rust_duration.as_nanos() as f64
        );

        // Verify results are comparable
        let comparison = compare_neighbor_lists(&r_result, rust_result_i32);
        let match_rate = comparison.match_count as f64 / comparison.total_points as f64;
        println!("  Match rate: {:.2}%", match_rate * 100.0);

        Ok(())
    }

    #[test]
    fn test_scalability_comparison() -> Result<(), Box<dyn std::error::Error>> {
        setup_r_environment()?;

        // Test configurations - (size, dimensions, radius)
        let test_cases = vec![
            (1_000, 3, 0.2),   // Medium dataset
            (5_000, 3, 0.2),   // Large dataset
            (10_000, 3, 0.15), // Very large dataset (smaller radius for sparsity)
        ];

        for (size, dim, radius) in test_cases {
            println!(
                "\n=== Running test case: size={}, dim={}, radius={:.2} ===",
                size, dim, radius
            );

            // Generate test dataset
            let takens = generate_test_takens(size, dim, 123);
            let takens_views = takens.views();
            let number_boxes = Some(50);

            // Time R execution
            let r_start = std::time::Instant::now();
            let r_result = call_r_find_neighbors(&takens.views(), radius, number_boxes)?;
            let r_duration = r_start.elapsed();

            // Time Rust execution
            let rust_start = std::time::Instant::now();
            let mut rust_finder = NeighborSearch::new(&takens_views, radius, number_boxes)?;
            let rust_result = rust_finder.find_all_neighbors()?;
            let rust_duration = rust_start.elapsed();
            let rust_result_i32: Vec<Vec<i32>> = rust_result
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as i32).collect())
                .collect();

            // Compare results
            let comparison = compare_neighbor_lists(&r_result, rust_result_i32);
            let match_rate = comparison.match_count as f64 / comparison.total_points as f64;

            // Output results
            println!(
                "Performance comparison for size={}, dim={}, radius={:.2} ===",
                size, dim, radius
            );
            println!("  R execution time: {:?}", r_duration);
            println!("  Rust execution time: {:?}", rust_duration);
            println!(
                "  Speedup: {:.2}x",
                r_duration.as_nanos() as f64 / rust_duration.as_nanos() as f64
            );

            if !comparison.differences.is_empty() {
                let total_missing: usize = comparison
                    .differences
                    .iter()
                    .map(|d| d.missing_in_rust.len())
                    .sum();
                let total_extra: usize = comparison
                    .differences
                    .iter()
                    .map(|d| d.extra_in_rust.len())
                    .sum();
                println!(
                    "  Discrepancies - Missing: {}, Extra: {}",
                    total_missing, total_extra
                );
            }

            // Validation
            assert!(
                match_rate >= 0.90,
                "Match rate too low for size {}: {:.2}%",
                size,
                match_rate * 100.0
            );
        }

        Ok(())
    }
}

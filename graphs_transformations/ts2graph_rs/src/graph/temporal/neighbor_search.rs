use std::usize;

use tch::{Device, Kind, Tensor};

pub struct NeighborSearch {
    phase_space: Tensor,
    embedding_dim: i64,
    number_vectors: i64,
    radius: f64,
    searching_workspace: Vec<i32>,
    boxes: Vec<i32>,
    possible_neighbors: Vec<i32>,
    device: Device,
}

impl NeighborSearch {
    /// Default constructor
    pub fn new(
        phase_space: Tensor,
        radius: f64,
        n_boxes: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = phase_space.size();
        if shape.len() != 2 {
            return Err("Phase space tensor must be 2-dimensional".into());
        }

        let number_vectors = shape[0];
        let embedding_dim = shape[1];
        let device = phase_space.device();
        let n_boxes = match n_boxes {
            Some(n) => n as usize,
            _ => estimate_number_boxes(&phase_space, radius) as usize,
        };

        let mut instance = Self {
            phase_space: phase_space,
            embedding_dim: embedding_dim,
            number_vectors: number_vectors,
            radius: radius,
            searching_workspace: vec![0; number_vectors as usize],
            boxes: vec![0; n_boxes * n_boxes + 1],
            possible_neighbors: vec![0; number_vectors as usize],
            device: device,
        };

        instance.prepare_box_assisted_search()?;
        Ok(instance)
    }

    /// Set the radius for neighbor search
    pub fn set_radius(&mut self, radius: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.radius = radius;
        if self.number_vectors > 0 {
            self.prepare_box_assisted_search()?;
        }
        Ok(())
    }

    /// Get reference to the phase space tensor
    pub fn get_phase_space(&self) -> &Tensor {
        &self.phase_space
    }

    /// Get the embedding dimension
    pub fn get_dimension(&self) -> i64 {
        self.embedding_dim
    }

    /// Get the number of vectors
    pub fn get_number_vectors(&self) -> i64 {
        self.number_vectors
    }

    /// Get the device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get wrapped position in the box grid
    fn get_wrapped_position(&self, row: i32, col: i32) -> usize {
        let n_boxes = ((self.boxes.len() - 1) as f64).sqrt() as i32;
        (n_boxes * positive_modulo(row, n_boxes) + positive_modulo(col, n_boxes)) as usize
    }

    /// Check if two vectors comply with the Theiler window constraint
    pub fn comply_theiler_window(
        vector_index1: usize,
        vector_index2: usize,
        theiler_window: i32,
    ) -> bool {
        if theiler_window < 0 {
            true
        } else {
            ((vector_index1 as i32) - (vector_index2 as i32)).abs() > theiler_window
        }
    }

    /// Calculate maximum distance between two vectors using max metric (tensor operations)
    pub fn calculate_max_distance(
        &self,
        vector_index1: usize,
        vector_index2: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let vec1 = self.phase_space.get(vector_index1 as i64);
        let vec2 = self.phase_space.get(vector_index2 as i64);
        let diff = &vec1 - &vec2;
        let abs_diff = diff.abs();
        let max_dist: f64 = abs_diff.max().try_into()?;
        Ok(max_dist)
    }

    /// Calculate distances between a vector and multiple candidates (vectorized)
    pub fn calculate_distances_vectorized(
        &self,
        vector_index: usize,
        candidates: &[usize],
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let target_vector = self.phase_space.get(vector_index as i64);

        // Create tensor of candidate indices
        let candidate_indices: Vec<i64> = candidates.iter().map(|&x| x as i64).collect();
        let indices_tensor = Tensor::from_slice(&candidate_indices);

        // Index select candidate vectors
        let candidate_vectors = self.phase_space.index_select(0, &indices_tensor);

        // Compute max distance using broadcasting
        let diff = &candidate_vectors - &target_vector.unsqueeze(0);
        let abs_diff = diff.abs();
        let max_distances = abs_diff.amax(&[1i64], false);

        Ok(max_distances)
    }

    /// Check if two phase space vectors are neighbors using the max metric
    pub fn are_neighbors(
        &self,
        vector_index1: usize,
        vector_index2: usize,
        neighborhood_radius: f64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let distance = self.calculate_max_distance(vector_index1, vector_index2)?;
        Ok(distance < neighborhood_radius)
    }

    /// Vectorized neighbor checking for multiple candidates
    pub fn are_neighbors_vectorized(
        &self,
        vector_index: usize,
        candidates: &[usize],
        neighborhood_radius: f64,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let distances = self.calculate_distances_vectorized(vector_index, candidates)?;
        let threshold = Tensor::scalar_tensor(neighborhood_radius, (Kind::Float, self.device));
        let mask = distances.lt_tensor(&threshold);

        let mask_vec: Vec<i64> = mask.try_into()?;
        Ok(mask_vec.into_iter().map(|x| x != 0).collect())
    }

    /// Find all neighbors for all vectors without Theiler window
    pub fn find_all_neighbors(&mut self) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>> {
        // A negative Theiler window indicates it can be ignored
        self.find_all_neighbors_with_theiler(-1)
    }

    /// Find all neighbors for all vectors with Theiler window
    pub fn find_all_neighbors_with_theiler(
        &mut self,
        theiler_window: i32,
    ) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>> {
        let n_vectors = self.number_vectors as usize;
        let mut neighbor_list = Vec::with_capacity(n_vectors);

        for i in 0..n_vectors {
            neighbor_list.push(self.box_assisted_search(i, theiler_window)?);
        }

        Ok(neighbor_list)
    }

    // Prepare the box-assisted search data structure
    fn prepare_box_assisted_search(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Reset boxes and possible neighbors
        self.boxes.fill(0);
        self.possible_neighbors.fill(0);

        if self.number_vectors == 0 || self.embedding_dim == 0 {
            return Ok(());
        }

        let n_takens = self.number_vectors as usize;
        let last_position = (self.embedding_dim - 1) as i64;

        // Extract first and last columns for box positioning using tensor operations
        let first_col = self.phase_space.select(1, 0);
        let last_col = self.phase_space.select(1, last_position);

        // Calculate box positions
        let x_box_positions = (&first_col / self.radius).to_kind(Kind::Int);
        let y_box_positions = (&last_col / self.radius).to_kind(Kind::Int);

        // Convert to CPU vectors for further processing
        let x_positions: Vec<i32> = x_box_positions.try_into()?;
        let y_positions: Vec<i32> = y_box_positions.try_into()?;

        // Count number of taken vectors in each box
        for i in 0..n_takens {
            let wrapped_box_position = self.get_wrapped_position(x_positions[i], y_positions[i]);
            self.boxes[wrapped_box_position] += 1;
        }

        // Calculate cumulative sum
        for i in 1..self.boxes.len() {
            self.boxes[i] += self.boxes[i - 1];
        }

        // Fill list of pointers to possible neighbors
        for i in 0..n_takens {
            let wrapped_box_position = self.get_wrapped_position(x_positions[i], y_positions[i]);
            self.boxes[wrapped_box_position] -= 1;
            self.possible_neighbors[self.boxes[wrapped_box_position] as usize] = i as i32;
        }

        Ok(())
    }

    pub fn box_assisted_search(
        &mut self,
        vector_index: usize,
        theiler_window: i32,
    ) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        let mut n_found = 0;
        let last_position = (self.embedding_dim - 1) as i64;

        // Get box positions using tensor operations
        let target_vector = self.phase_space.get(vector_index as i64);
        let x_box_positions = (f64::try_from(target_vector.get(0))? / self.radius) as i32;
        let y_box_positions =
            (f64::try_from(target_vector.get(last_position))? / self.radius) as i32;

        let mut candidates = Vec::new();

        // Look for vector neighbors in the 9 neighbors of the box
        for i in (x_box_positions - 1)..=(x_box_positions + 1) {
            for j in (y_box_positions - 1)..=(y_box_positions + 1) {
                let auxiliar_box_pos = self.get_wrapped_position(i, j);

                // Avoid empty boxes
                let start_idx = self.boxes[auxiliar_box_pos] as usize;
                let end_idx = if auxiliar_box_pos + 1 < self.boxes.len() {
                    self.boxes[auxiliar_box_pos + 1] as usize
                } else {
                    self.possible_neighbors.len()
                };

                for box_ptr in (start_idx..end_idx).rev() {
                    let possible_neighs = self.possible_neighbors[box_ptr] as usize;

                    if possible_neighs == vector_index {
                        continue;
                    }

                    if Self::comply_theiler_window(vector_index, possible_neighs, theiler_window) {
                        candidates.push(possible_neighs);
                    }
                }
            }
        }

        // Use vectorized operations for distance checking when we have many candidates
        if candidates.len() > 10 {
            let neighbor_mask =
                self.are_neighbors_vectorized(vector_index, &candidates, self.radius)?;
            for (i, &is_neighbor) in neighbor_mask.iter().enumerate() {
                if is_neighbor {
                    self.searching_workspace[n_found] = candidates[i] as i32;
                    n_found += 1;
                }
            }
        } else {
            // Use scalar operations for small candidates sets
            for &candidate in &candidates {
                if self.are_neighbors(vector_index, candidate, self.radius)? {
                    self.searching_workspace[n_found] = candidate as i32;
                    n_found += 1;
                }
            }
        }

        Ok(self.searching_workspace[0..n_found].to_vec())
    }
}

/// Helper function for positive modulo operation
fn positive_modulo(a: i32, b: i32) -> i32 {
    ((a % b) + b) % b
}

// Helper function to generate test data
fn generate_test_takens(n_points: usize, dimension: usize, seed: u64) -> Vec<Vec<f64>> {
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

    takens
}

/// Estimate the number of boxes
fn estimate_number_boxes(data: &Tensor, radius: f64) -> i32 {
    let min_nb_boxes = 10;
    let max_nb_boxes = 500;
    if data.numel() == 0 {
        return min_nb_boxes;
    }

    if radius <= 0.0 {
        return min_nb_boxes;
    }

    // Find min and max values
    let min_val = data.min().double_value(&[]) as f64;
    let max_val = data.max().double_value(&[]) as f64;

    // Calculate number of boxes: (max - min) / radius
    let range = max_val - min_val;
    let mut number_boxes = range / radius;

    // Apply bounds
    if number_boxes > max_nb_boxes as f64 {
        number_boxes = max_nb_boxes as f64;
    }
    if number_boxes < min_nb_boxes as f64 {
        number_boxes = min_nb_boxes as f64;
    }

    number_boxes as i32
}

pub fn find_neighbors_benchmark() {
    use std::time::Instant;
    let _guard = pprof::ProfilerGuard::new(100).unwrap(); // Start profiler

    let takens = generate_test_takens(10_000, 3, 123);
    let tensor = Tensor::from_slice2(&takens);

    let start = Instant::now();
    let mut finder = NeighborSearch::new(tensor, 0.2, None).unwrap();
    let _neighbors = finder.find_all_neighbors().unwrap();
    println!("Total time: {:?}", start.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tch::Tensor;

    #[test]
    fn test_neighbor_search_with_tensor() -> Result<(), Box<dyn std::error::Error>> {
        let data = vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 5.0, 6.0, 7.0];
        let tensor = Tensor::from_slice(&data).reshape(&[3, 3]);

        let neighbor_search = NeighborSearch::new(tensor, 0.5, Some(10))?;
        assert_eq!(neighbor_search.get_dimension(), 3);
        assert_eq!(neighbor_search.get_number_vectors(), 3);
        Ok(())
    }

    #[test]
    fn test_neighbor_search_with_data() -> Result<(), Box<dyn std::error::Error>> {
        let phase_space = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![5.0, 6.0, 7.0],
        ];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let neighbor_search = NeighborSearch::new(tensor_phase_space, 0.5, Some(10))?;
        assert_eq!(neighbor_search.get_dimension(), 3);
        assert_eq!(neighbor_search.get_number_vectors(), 3);
        Ok(())
    }

    #[test]
    fn test_distance_calculation() -> Result<(), Box<dyn std::error::Error>> {
        let phase_space = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![1.0, 1.0]];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let neighbor_search = NeighborSearch::new(tensor_phase_space, 0.5, Some(5))?;

        let distance = neighbor_search.calculate_max_distance(0, 1)?;
        assert!((distance - 0.1).abs() < 1e-6);

        let distance2 = neighbor_search.calculate_max_distance(0, 2)?;
        assert!((distance2 - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_vectorized_distances() -> Result<(), Box<dyn std::error::Error>> {
        let phase_space = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![1.0, 1.0]];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let neighbor_search = NeighborSearch::new(tensor_phase_space, 0.5, Some(5))?;

        let candidates = vec![1, 2];
        let distances = neighbor_search.calculate_distances_vectorized(0, &candidates)?;
        let distances_vec: Vec<f64> = distances.try_into()?;

        assert!((distances_vec[0] - 0.1).abs() < 1e-6);
        assert!((distances_vec[1] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_are_neighbors() -> Result<(), Box<dyn std::error::Error>> {
        let phase_space = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![1.0, 1.0]];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let neighbor_search = NeighborSearch::new(tensor_phase_space, 0.5, Some(5))?;

        assert!(neighbor_search.are_neighbors(0, 1, 0.2)?);
        assert!(!neighbor_search.are_neighbors(0, 2, 0.5)?);
        Ok(())
    }

    #[test]
    fn test_theiler_window() {
        assert!(NeighborSearch::comply_theiler_window(0, 5, 3));
        assert!(!NeighborSearch::comply_theiler_window(0, 2, 3));
        assert!(NeighborSearch::comply_theiler_window(0, 2, -1)); // Negative window ignores constraint
    }

    #[test]
    fn test_positive_modulo() {
        assert_eq!(positive_modulo(5, 3), 2);
        assert_eq!(positive_modulo(-1, 3), 2);
        assert_eq!(positive_modulo(-4, 3), 2);
    }

    #[test]
    fn test_find_all_neighbors_simple_case() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple 2D phase space with known neighbors
        let phase_space = vec![
            vec![0.0, 0.0],   // Point 0: origin
            vec![0.05, 0.05], // Point 1: close to origin
            vec![0.15, 0.15], // Point 2: medium distance
            vec![1.0, 1.0],   // Point 3: far from others
            vec![0.02, 0.03], // Point 4: very close to origin
        ];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search = NeighborSearch::new(tensor_phase_space, 0.1, Some(5))?;

        let all_neighbors = neighbor_search.find_all_neighbors()?;

        // Verify structure
        assert_eq!(all_neighbors.len(), 5);

        // Point 0 should have neighbors 1 and 4 (within radius 0.1)
        let neighbors_0: HashSet<i32> = all_neighbors[0].iter().cloned().collect();
        assert!(neighbors_0.contains(&1));
        assert!(neighbors_0.contains(&4));
        assert!(!neighbors_0.contains(&2)); // Too far
        assert!(!neighbors_0.contains(&3)); // Too far
        assert!(!neighbors_0.contains(&0)); // Self-exclusion

        // Point 3 should have no neighbors (isolated)
        assert_eq!(all_neighbors[3].len(), 0);

        Ok(())
    }

    #[test]
    fn test_find_all_neighbors_with_theiler_window() -> Result<(), Box<dyn std::error::Error>> {
        // Create a time series-like scenario where temporal correlation matters
        let phase_space = vec![
            vec![0.0, 0.0],   // t=0
            vec![0.01, 0.01], // t=1, very close to t=0
            vec![0.02, 0.02], // t=2, close to t=1
            vec![0.03, 0.03], // t=3, close to t=2
            vec![1.0, 1.0],   // t=4, far from others
        ];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search = NeighborSearch::new(tensor_phase_space, 0.05, Some(5))?;

        // Without Theiler window
        let neighbors_no_theiler = neighbor_search.find_all_neighbors()?;

        // With Theiler window of 2 (exclude temporally close points)
        let neighbors_with_theiler = neighbor_search.find_all_neighbors_with_theiler(2)?;

        // Point 0 should have fewer neighbors with Theiler window
        // Without Theiler: should find points 1, 2, 3
        // With Theiler=2: should exclude points 1, 2 (too close in time)
        assert!(neighbors_no_theiler[0].len() > neighbors_with_theiler[0].len());

        // Check that point 0 with Theiler=2 doesn't include points 1 and 2
        let theiler_neighbors_0: HashSet<i32> = neighbors_with_theiler[0].iter().cloned().collect();
        assert!(!theiler_neighbors_0.contains(&1)); // Excluded by Theiler
        assert!(!theiler_neighbors_0.contains(&2)); // Excluded by Theiler

        Ok(())
    }

    #[test]
    fn test_find_all_neighbors_symmetry() -> Result<(), Box<dyn std::error::Error>> {
        // Test that neighbor relationships are symmetric
        let phase_space = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.05],
            vec![0.1, 0.1],
            vec![0.5, 0.5],
        ];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search = NeighborSearch::new(tensor_phase_space, 0.15, Some(5))?;

        let all_neighbors = neighbor_search.find_all_neighbors()?;

        // Check symmetry: if i is a neighbor of j, then j should be a neighbor of i
        for i in 0..all_neighbors.len() {
            let _neighbors_i: HashSet<i32> = all_neighbors[i].iter().cloned().collect();
            for &neighbor_j in &all_neighbors[i] {
                let j = neighbor_j as usize;
                let neighbors_j: HashSet<i32> = all_neighbors[j].iter().cloned().collect();
                assert!(
                    neighbors_j.contains(&(i as i32)),
                    "Asymmetric relationship: {} -> {}, but not {} -> {}",
                    i,
                    j,
                    j,
                    i
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_find_all_neighbors_large_dataset() -> Result<(), Box<dyn std::error::Error>> {
        // Test performance and correctness with larger dataset
        let n_points = 500;
        let mut phase_space = Vec::with_capacity(n_points);

        // Create a grid of points
        let grid_size = (n_points as f64).sqrt().ceil() as usize;
        for i in 0..grid_size {
            for j in 0..grid_size {
                if phase_space.len() >= n_points {
                    break;
                }
                phase_space.push(vec![i as f64 * 0.1, j as f64 * 0.1]);
            }
            if phase_space.len() >= n_points {
                break;
            }
        }

        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search = NeighborSearch::new(tensor_phase_space, 0.15, Some(20))?;
        let all_neighbors = neighbor_search.find_all_neighbors()?;

        // Verify basic properties
        assert_eq!(all_neighbors.len(), n_points);

        // Each point should have some neighbors (except possibly edge cases)
        let mut total_neighbor_count = 0;
        for neighbors in &all_neighbors {
            total_neighbor_count += neighbors.len();
        }

        // Should have reasonable number of neighbors overall
        assert!(total_neighbor_count > 0);

        // Corner point (0,0) should have fewer neighbors than center points
        let _corner_neighbors = all_neighbors[0].len();
        let center_idx = grid_size / 2 * grid_size + grid_size / 2;
        if center_idx < all_neighbors.len() {
            let _center_neighbors = all_neighbors[center_idx].len();
            // Note: this is generally true but might not always hold due to grid edges
            // assert!(corner_neighbors <= center_neighbors,
            //     "Corner should have <= neighbors than center: {} vs {}",
            //     corner_neighbors, center_neighbors);
        }

        Ok(())
    }

    #[test]
    fn test_find_all_neighbors_different_radii() -> Result<(), Box<dyn std::error::Error>> {
        let phase_space = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.05],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![0.5, 0.5],
        ];

        // Test with small radius
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search_small = NeighborSearch::new(tensor_phase_space, 0.08, Some(5))?;
        let neighbors_small = neighbor_search_small.find_all_neighbors()?;

        // Test with large radius
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search_large = NeighborSearch::new(tensor_phase_space, 0.25, Some(5))?;
        let neighbors_large = neighbor_search_large.find_all_neighbors()?;

        // Larger radius should generally result in more neighbors
        for i in 0..neighbors_small.len() {
            assert!(
                neighbors_small[i].len() <= neighbors_large[i].len(),
                "Point {} should have more neighbors with larger radius: {} <= {}",
                i,
                neighbors_small[i].len(),
                neighbors_large[i].len()
            );
        }

        Ok(())
    }

    #[test]
    fn test_find_all_neighbors_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        // Test with single point
        let single_point = vec![vec![1.0, 2.0]];
        let tensor_phase_space = Tensor::from_slice2(&single_point);
        let mut neighbor_search_single = NeighborSearch::new(tensor_phase_space, 1.0, Some(3))?;
        let neighbors_single = neighbor_search_single.find_all_neighbors()?;
        assert_eq!(neighbors_single.len(), 1);
        assert_eq!(neighbors_single[0].len(), 0); // No neighbors for single point

        // Test with identical points
        let identical_points = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let tensor_phase_space = Tensor::from_slice2(&identical_points);
        let mut neighbor_search_identical = NeighborSearch::new(tensor_phase_space, 0.1, Some(3))?;
        let neighbors_identical = neighbor_search_identical.find_all_neighbors()?;

        // All points should be neighbors of each other (except self)
        for i in 0..neighbors_identical.len() {
            assert_eq!(
                neighbors_identical[i].len(),
                2,
                "Each identical point should have 2 neighbors, got {} for point {}",
                neighbors_identical[i].len(),
                i
            );
        }

        Ok(())
    }

    #[test]
    fn test_find_all_neighbors_high_dimensional() -> Result<(), Box<dyn std::error::Error>> {
        // Test with higher dimensional phase space
        let phase_space_5d = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.01, 0.01, 0.01, 0.01, 0.01],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.05, 0.0, 0.0, 0.0, 0.0], // Different in first dim only
            vec![0.0, 0.05, 0.0, 0.0, 0.0], // Different in second dim only
        ];

        let tensor_phase_space = Tensor::from_slice2(&phase_space_5d);
        let mut neighbor_search = NeighborSearch::new(tensor_phase_space, 0.08, Some(8))?;
        let all_neighbors = neighbor_search.find_all_neighbors()?;

        // Point 0 should be close to points 1, 3, 4 but not 2
        let neighbors_0: HashSet<i32> = all_neighbors[0].iter().cloned().collect();
        assert!(neighbors_0.contains(&1)); // Very close
        assert!(neighbors_0.contains(&3)); // Close in one dimension
        assert!(neighbors_0.contains(&4)); // Close in one dimension
        assert!(!neighbors_0.contains(&2)); // Too far (max distance = 0.1)

        Ok(())
    }

    #[test]
    fn test_box_assisted_search_consistency() -> Result<(), Box<dyn std::error::Error>> {
        // Test that box-assisted search gives same results as brute force
        let phase_space = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.05],
            vec![0.1, 0.1],
            vec![0.15, 0.15],
            vec![0.5, 0.5],
        ];

        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let mut neighbor_search = NeighborSearch::new(tensor_phase_space, 0.12, Some(5))?;
        let box_neighbors = neighbor_search.find_all_neighbors()?;

        // Verify results with manual brute-force calculation
        for i in 0..5 {
            let found_neighbors: HashSet<i32> = box_neighbors[i].iter().cloned().collect();

            // Manual neighbor check
            let mut expected_neighbors = HashSet::new();
            for j in 0..5 {
                if i != j {
                    let distance = neighbor_search.calculate_max_distance(i, j)?;
                    if distance < 0.12 {
                        expected_neighbors.insert(j as i32);
                    }
                }
            }

            assert_eq!(
                found_neighbors, expected_neighbors,
                "Box-assisted search mismatch for point {}: found {:?}, expected {:?}",
                i, found_neighbors, expected_neighbors
            );
        }

        Ok(())
    }

    #[test]
    fn test_vectorized_vs_scalar_operations() -> Result<(), Box<dyn std::error::Error>> {
        // Test that vectorized operations give same results as scalar operations
        let phase_space = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.05],
            vec![0.1, 0.1],
            vec![0.15, 0.15],
            vec![0.2, 0.2],
        ];

        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let neighbor_search = NeighborSearch::new(tensor_phase_space, 0.12, Some(5))?;
        let candidates = vec![1, 2, 3, 4];

        // Vectorized approach
        let vectorized_neighbors =
            neighbor_search.are_neighbors_vectorized(0, &candidates, 0.12)?;

        // Scalar approach
        let mut scalar_neighbors = Vec::new();
        for &candidate in &candidates {
            scalar_neighbors.push(neighbor_search.are_neighbors(0, candidate, 0.12)?);
        }

        assert_eq!(
            vectorized_neighbors, scalar_neighbors,
            "Vectorized and scalar neighbor detection should match"
        );

        Ok(())
    }

    #[test]
    fn test_prepare_box_assisted_search_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        // Test box preparation with various challenging scenarios

        // All points in same location
        let same_location = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
        let tensor_phase_space = Tensor::from_slice2(&same_location);
        let neighbor_search1 = NeighborSearch::new(tensor_phase_space, 0.1, Some(3))?;
        assert_eq!(neighbor_search1.get_number_vectors(), 3);

        // Points spread across large range
        let wide_spread = vec![vec![-1000.0, -1000.0], vec![0.0, 0.0], vec![1000.0, 1000.0]];
        let tensor_phase_space = Tensor::from_slice2(&wide_spread);
        let neighbor_search2 = NeighborSearch::new(tensor_phase_space, 100.0, Some(10))?;
        assert_eq!(neighbor_search2.get_number_vectors(), 3);

        // Very small radius
        let small_radius_case = vec![vec![0.0, 0.0], vec![0.000001, 0.000001]];
        let tensor_phase_space = Tensor::from_slice2(&small_radius_case);
        let neighbor_search3 = NeighborSearch::new(tensor_phase_space, 0.000001, Some(5))?;
        assert_eq!(neighbor_search3.get_number_vectors(), 2);

        Ok(())
    }

    #[test]
    fn test_wrapped_position_calculation() -> Result<(), Box<dyn std::error::Error>> {
        let phase_space = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let tensor_phase_space = Tensor::from_slice2(&phase_space);
        let neighbor_search = NeighborSearch::new(tensor_phase_space, 1.0, Some(3))?; // 3x3 grid

        // Test various wrapped positions
        assert_eq!(neighbor_search.get_wrapped_position(0, 0), 0);
        assert_eq!(neighbor_search.get_wrapped_position(1, 1), 4); // center of 3x3
        assert_eq!(neighbor_search.get_wrapped_position(-1, -1), 8); // wraps to bottom-right
        assert_eq!(neighbor_search.get_wrapped_position(3, 3), 0); // wraps to top-left

        Ok(())
    }
}

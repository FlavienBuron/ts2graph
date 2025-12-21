// build.rs
use std::path::Path;

// fn main() {
//     println!("cargo:rerun-if-changed=build.rs");
//
//     // Define the libtorch path
//     let libtorch_path =
//         "/home/flavien/ownCloud/MasterThesis/ts2graph/.venv/lib/python3.13/site-packages/torch/lib";
//
//     // Check if the path exists
//     if !Path::new(libtorch_path).exists() {
//         panic!(
//             "ERROR: libtorch path does not exist: {}\n\
//              Please check:\n\
//              1. Is the virtual environment activated?\n\
//              2. Is PyTorch installed in the virtual environment?\n\
//              3. Does the path exist?",
//             libtorch_path
//         );
//     }
//
//     // Check for required libraries
//     let required_libs = ["libtorch.so", "libtorch_cpu.so", "libc10.so"];
//     let mut missing_libs = Vec::new();
//
//     for lib in &required_libs {
//         let lib_path = Path::new(libtorch_path).join(lib);
//         if !lib_path.exists() {
//             missing_libs.push(lib);
//         }
//     }
//
//     if !missing_libs.is_empty() {
//         println!("cargo:warning=Missing libraries: {:?}", missing_libs);
//         println!("cargo:warning=Files in {}:", libtorch_path);
//         if let Ok(entries) = std::fs::read_dir(libtorch_path) {
//             for entry in entries.flatten() {
//                 if let Ok(file_name) = entry.file_name().into_string() {
//                     if file_name.ends_with(".so") {
//                         println!("cargo:warning=  - {}", file_name);
//                     }
//                 }
//             }
//         }
//     }
//
//     // Add library search path
//     println!("cargo:rustc-link-search=native={}", libtorch_path);
//
//     // Link libraries
//     println!("cargo:rustc-link-lib=dylib=torch");
//     println!("cargo:rustc-link-lib=dylib=torch_cpu");
//     println!("cargo:rustc-link-lib=dylib=c10");
//
//     // Link C++ standard library
//     if cfg!(target_os = "macos") {
//         println!("cargo:rustc-link-lib=c++");
//     } else {
//         println!("cargo:rustc-link-lib=stdc++");
//     }
//
//     // Success message
//     println!(
//         "cargo:warning=✓ Successfully configured libtorch from: {}",
//         libtorch_path
//     );
// }
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Only link against C++ stdlib, NOT PyTorch
    // PyTorch will be provided by Python at runtime
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:warning=Using system PyTorch from Python (no direct linking)");
}

[package]
name = "llama-server-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
# The web framework
axum = "0.7"

# The async runtime
tokio = { version = "1", features = ["full"] }

# JSON serialization/deserialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Rust bindings for llama.cpp
# Check for the latest version on crates.io
llama-cpp-rs = { version = "0.2.0", features = ["llama-cpp-sys/default"] }

# For structured logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# For easy, one-time static initialization (optional but clean)
# once_cell = "1.19"

# For ergonomic error handling
anyhow = "1.0"

# --- GPU Acceleration (Optional) ---
# To enable GPU support, you must enable the correct feature flag for llama-cpp-rs.
# First, disable the default features, then pick ONE of the following backends.
# Make sure you have the corresponding toolkits (e.g., CUDA Toolkit) installed.
#
# [dependencies.llama-cpp-rs]
# version = "0.2.0"
# default-features = false
# features = ["llama-cpp-sys/cublas"] # For NVIDIA GPUs
# # features = ["llama-cpp-sys/metal"] # For Apple Silicon
# # features = ["llama-cpp-sys/clblast"] # For OpenCL
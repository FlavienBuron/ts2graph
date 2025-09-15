#!/usr/bin/env bash
set -e

echo "Installing rustup and cargo"
curl https://sh.rustup.rs -sSf | sh -s -- -y

# Ensure cargo binaries are on PATH
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
else
    echo "Rust environment file not found at $HOME/.cargo/env"
    exit 1
fi

echo "Check Rust installation"
rustc --version
cargo --version

echo "Creating Python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing Python requirements"
pip install --upgrade pip
pip install -r requirements.txt

echo "Building Rust extension with maturin"
cd ./graphs_transformations/bindings/ts2graph_rs/
maturin develop --release


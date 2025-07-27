#!/bin/bash
set -e

echo "Installing rustup and cargo"
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

echo "Check Rust installation"
rustc --version
cargo --version

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cd ./graphs_transformations/bindings/ts2graph_rs/

maturin develop --release

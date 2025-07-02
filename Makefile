.PHONY: check-plonky2 check-wasm32 ci-check clippy fmt lint test test-no-std test-suite wasm-test

check-plonky2:
	RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0" \
	RUST_LOG=1 \
	CARGO_INCREMENTAL=1 \
	RUST_BACKTRACE=1 \
	cargo check --manifest-path plonky2/Cargo.toml

check-wasm32:
	RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0" \
	RUST_LOG=1 \
	CARGO_INCREMENTAL=1 \
	RUST_BACKTRACE=1 \
	cargo check --manifest-path plonky2/Cargo.toml --target wasm32-unknown-unknown --no-default-features

ci-check: test-suite check-wasm32 test-no-std lint wasm-test

clippy:
	cargo clippy --all-features --all-targets -- -D warnings -A incomplete-features

fmt:
	cargo fmt --all --check

lint: clippy fmt

test:
	RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0" \
	RUST_LOG=1 \
	CARGO_INCREMENTAL=1 \
	RUST_BACKTRACE=1 \
	cargo test --workspace

test-no-std:
	RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0" \
	RUST_LOG=1 \
	CARGO_INCREMENTAL=1 \
	RUST_BACKTRACE=1 \
	cargo test --manifest-path plonky2/Cargo.toml --no-default-features --lib

test-suite: check-plonky2 test

wasm-test:
	cd plonky2/src/recursion \
	wasm-pack test --node

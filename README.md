# Plonky2 GoldiBear: an extended version of Plonky2 supporting both Goldilocks and BabyBear fields

Plonky2 GoldiBear implementation now it's generic on the field (the field must be two-adic and with at most 64 bits) enabling the support of both Goldilocks and BabyBear. 

Plonky2 GoldiBear introduces the support for recursive proofs composition over both fields, using Poseidon over Goldilocks and Poseidon2 over BabyBear. 
The implementations of the fields and Poseidon2 are taken directly from the [Plonky3](https://github.com/Plonky3/Plonky3) repo as dependencies.

This implementation solves also the [Plonky2 issue 456](https://github.com/0xPolygonZero/plonky2/issues/456) that was much more probable to happen using the BabyBear field (see [commit](https://github.com/telosnetwork/plonky2/commit/cf802b6d2c125a90f057ba7ad72ad0b4904fb450)).

## Documentation

For more details about the Plonky2 argument system, see this [writeup](plonky2/plonky2.pdf).

Polymer Labs has written up a helpful tutorial [here](https://polymerlabs.medium.com/a-tutorial-on-writing-zk-proofs-with-plonky2-part-i-be5812f6b798)!

## Examples

A good starting point for how to use Plonky2 for simple applications is the included examples:

* [`factorial`](plonky2/examples/factorial.rs): Proving knowledge of 100!
* [`fibonacci`](plonky2/examples/fibonacci.rs): Proving knowledge of the hundredth Fibonacci number
* [`range_check`](plonky2/examples/range_check.rs): Proving that a field element is in a given range
* [`square_root`](plonky2/examples/square_root.rs): Proving knowledge of the square root of a given field element

To run an example, use

```sh
cargo run --example <example_name>
```


## Building

Plonky2 requires a recent nightly toolchain, although we plan to transition to stable in the future.

To use a nightly toolchain for Plonky2 by default, you can run
```
rustup override set nightly
```
in the Plonky2 directory.


## Running

To see recursion performance, one can run this bench [`recursion`](plonky2/benches/recursion.rs), which tests both fields:

```sh
RUSTFLAGS=-Ctarget-cpu=native cargo bench recursion
```

## Testing

Tests can be simply run with:

```sh
cargo test
```

Use `--release` for faster execution.

### Wasm32

Given that the verification of proofs can be run also on different architectures (e.g. `wasm32`), there is a test that is meant to run on `wasm32` (`test_recursive_verifier_gl_regression`).

To run such test, it's required to have `nodejs` installed (tested with version `v23.1.0`) and `wasm-pack` (tested with version `0.13.0`). The latter can be installed with:

```sh
cargo install wasm-pack
```

Then, the test can be run with:

```sh
wasm-pack test --node
```

## Jemalloc

Plonky2 prefers the [Jemalloc](http://jemalloc.net) memory allocator due to its superior performance. To use it, include `jemallocator = "0.5.0"` in your `Cargo.toml` and add the following lines
to your `main.rs`:

```rust
use jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
```

Jemalloc is known to cause crashes when a binary compiled for x86 is run on an Apple silicon-based Mac under [Rosetta 2](https://support.apple.com/en-us/HT211861). If you are experiencing crashes on your Apple silicon Mac, run `rustc --print target-libdir`. The output should contain `aarch64-apple-darwin`. If the output contains `x86_64-apple-darwin`, then you are running the Rust toolchain for x86; we recommend switching to the native ARM version.

## Contributing guidelines

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Licenses

All crates of this monorepo are licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## Security

This code has not yet been audited, and should not be used in any production systems.

While Plonky2 is configurable, its defaults generally target 100 bits of security. The default FRI configuration targets 100 bits of *conjectured* security based on the conjecture in [ethSTARK](https://eprint.iacr.org/2021/582).

Plonky2's default hash function over Goldilocks is Poseidon, configured with 8 full rounds, 22 partial rounds, a width of 12 field elements (each ~64 bits), and an S-box of `x^7`. 
Over BabyBear is Poseidon2, configured with 8 full rounds, 13 partial rounds, a width of 16 field elements (each ~31 bits), and an S-box of `x^7`. [BBLP22]
(https://tosc.iacr.org/index.php/ToSC/article/view/9850) suggests that these configurations may have around 95 bits of security, falling a bit short of our 100 bit target.


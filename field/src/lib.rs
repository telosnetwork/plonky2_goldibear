#![no_std]
#![cfg_attr(
    all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
    feature(stdarch_x86_avx512)
)]
#![allow(incomplete_features)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![feature(specialization)]
#![cfg_attr(not(test), no_std)]
#![cfg(not(test))]
extern crate alloc;

pub(crate) mod arch;

pub mod batch_util;
pub mod cosets;
pub mod extension;
pub mod fft;
pub mod goldilocks_extensions;
pub mod goldilocks_field;
pub mod interpolation;
pub mod ops;
pub mod packable;
pub mod packed;
pub mod polynomial;
pub mod secp256k1_base;
pub mod secp256k1_scalar;
pub mod types;
pub mod zero_poly_coset;

pub mod monty;
#[cfg(test)]
mod field_testing;

#[cfg(test)]
mod prime_field_testing;

mod baby_bear;
mod babybear_extension;
mod babybear_mds;
mod baybybear_poseidon2;

pub use baby_bear::*;
pub use baybear_mds::*;
pub use baybear_poseidon2::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use x86_64_avx2::*;

#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
mod x86_64_avx512;
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub use x86_64_avx512::*;
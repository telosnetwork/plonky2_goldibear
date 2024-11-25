#![no_std]
#![allow(incomplete_features)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![feature(specialization)]
#![cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
    feature(stdarch_x86_avx512)
)]

extern crate alloc;

pub mod batch_util;
pub mod cosets;
pub mod extension;
pub mod extension_algebra;
pub mod fft;
pub mod interpolation;
pub mod ops;
pub mod polynomial;
//pub mod secp256k1_base;
//pub mod secp256k1_scalar;
#[cfg(test)]
mod field_testing;
pub mod types;
pub mod zero_poly_coset;

#[cfg(test)]
mod prime_field_testing;

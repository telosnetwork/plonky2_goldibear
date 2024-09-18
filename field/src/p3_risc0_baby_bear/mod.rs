#![no_std]
extern crate alloc;
mod field;
mod poseidon2;

pub use field::*;
pub use poseidon2::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use x86_64_avx2::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
mod x86_64_avx512;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub use x86_64_avx512::*;

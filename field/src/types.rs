use alloc::vec::Vec;
use p3_field::{TwoAdicField};

use rand::rngs::OsRng;

pub fn two_adic_subgroup<F:TwoAdicField>(n_log: usize) -> Vec<F> {
    let generator = F::two_adic_generator(n_log);
    generator.powers().take(1 << n_log).collect()
}
/// Sampling
pub trait Sample: Sized {
    /// Samples a single value using `rng`.
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized;

    /// Samples a single value using the [`OsRng`].
    #[inline]
    fn rand() -> Self {
        Self::sample(&mut OsRng)
    }

    /// Samples a [`Vec`] of values of length `n` using [`OsRng`].
    #[inline]
    fn rand_vec(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::rand()).collect()
    }

    /// Samples an array of values of length `N` using [`OsRng`].
    #[inline]
    fn rand_array<const N: usize>() -> [Self; N] {
        Self::rand_vec(N)
            .try_into()
            .ok()
            .expect("This conversion can never fail.")
    }
}



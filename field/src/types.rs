use alloc::vec::Vec;
use p3_baby_bear::BabyBear;

use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32, PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use rand::rngs::OsRng;
use rand::RngCore;

pub fn two_adic_subgroup<F: TwoAdicField>(n_log: usize) -> Vec<F> {
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

impl Sample for Goldilocks {
    fn sample<R>(rng: &mut R) -> Self
    where
        R: RngCore + ?Sized,
    {
        use rand::Rng;
        Self::from_canonical_u64(rng.gen_range(0..Self::ORDER_U64))
    }
}

impl<const D: usize, F: AbstractField + Sample + BinomiallyExtendable<D>> Sample
    for BinomialExtensionField<F, D>
{
    #[inline]
    fn sample<R>(_rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        <Self as AbstractExtensionField<F>>::from_base_slice(
            &(0..D).map(|_| F::rand()).collect::<Vec<_>>(),
        )
    }
}

impl Sample for BabyBear {
    fn sample<R>(rng: &mut R) -> Self
    where
        R: RngCore + ?Sized,
    {
        use rand::Rng;
        Self::from_canonical_u32(rng.gen_range(0..Self::ORDER_U32))
    }
}



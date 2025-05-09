use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_field::extension::{
    BinomialExtensionField, BinomiallyExtendable, HasTwoAdicBionmialExtension,
};
use p3_field::{
    AbstractExtensionField, AbstractField, ExtensionField, PrimeField32, PrimeField64, TwoAdicField,
};
use p3_goldilocks::Goldilocks;
use rand::rngs::OsRng;
use rand::RngCore;

pub fn two_adic_subgroup<F: TwoAdicField>(n_log: usize) -> Vec<F> {
    let generator = F::two_adic_generator(n_log);
    generator.powers().take(1 << n_log).collect()
}

pub trait HasExtension<const D: usize>:
    BinomiallyExtendable<D> + HasTwoAdicBionmialExtension<D>
{
    type Extension: ExtensionField<Self::F> + TwoAdicField;
}

impl<T: BinomiallyExtendable<D> + HasTwoAdicBionmialExtension<D>, const D: usize> HasExtension<D>
    for T
{
    type Extension = BinomialExtensionField<T::F, D>;
}

impl<F: HasExtension<D> + Sample, const D: usize> Sample for BinomialExtensionField<F, D> {
    fn sample<R>(_rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        Self::from_base_slice(&(0..D).map(|_| F::sample(&mut OsRng)).collect::<Vec<_>>())
    }
}

#[cfg(feature = "disable-randomness")]
pub mod tests_non_rand {
    extern crate std;

    use lazy_static::lazy_static;
    use rand_xoshiro::rand_core;
    lazy_static! {
        pub static ref value: std::sync::Mutex<u64> = std::sync::Mutex::new(1);
    }
    #[derive(Debug, Default)]
    pub struct NonRandomRng;

    impl NonRandomRng {
        pub fn reset() {
            let mut val = value.lock().unwrap();
            *val = 1;
        }
    }
    impl rand::RngCore for NonRandomRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            let mut val = value.lock().unwrap(); // Lock the mutex to access the value
            *val += 1;
            *val
        }

        fn fill_bytes(&mut self, _dest: &mut [u8]) {
            panic!();
        }

        fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), rand_core::Error> {
            panic!();
        }
    }
}
/// Sampling
pub trait Sample: Sized {
    /// Samples a single value using `rng`.
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized;

    /// Samples a single value using the [`OsRng`].
    #[cfg(not(feature = "disable-randomness"))]
    #[inline]
    fn rand() -> Self {
        Self::sample(&mut OsRng)
    }

    #[cfg(feature = "disable-randomness")]
    #[inline]
    fn rand() -> Self {
        use crate::types::tests_non_rand::NonRandomRng;
        Self::sample(&mut NonRandomRng)
    }

    /// Samples a [`Vec`] of values of length `n` using [`OsRng`].
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

impl Sample for BabyBear {
    fn sample<R>(rng: &mut R) -> Self
    where
        R: RngCore + ?Sized,
    {
        use rand::Rng;
        Self::from_canonical_u32(rng.gen_range(0..Self::ORDER_U32))
    }
}

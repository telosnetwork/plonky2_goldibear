use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_field::{AbstractField, Field};

use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::Sample;

use p3_field::extension::HasFrobenius;
use p3_field::extension::HasTwoAdicBionmialExtension;
use p3_field::TwoAdicField;
use p3_field::AbstractExtensionField;

#[macro_export]
macro_rules! test_field_arithmetic {
    ($field:ty) => {
        mod field_arithmetic {
            use alloc::vec::Vec;

            use num::bigint::BigUint;
            use rand::rngs::OsRng;
            use rand::{Rng, RngCore};
            use $crate::types::{Field, Sample};

            #[test]
            fn modular_reduction() {
                let mut rng = OsRng;
                for _ in 0..10 {
                    let x_lo = rng.next_u64();
                    let x_hi = rng.next_u32();
                    let x = (x_lo as u128) + ((x_hi as u128) << 64);
                    let a = <$field>::from_noncanonical_u128(x);
                    let b = <$field>::from_noncanonical_u96((x_lo, x_hi));
                    assert_eq!(a, b);
                }
            }

            #[test]
            fn batch_inversion() {
                for n in 0..20 {
                    let xs = (1..=n as u64)
                        .map(|i| <$field>::from_canonical_u64(i))
                        .collect::<Vec<_>>();
                    let invs = <$field>::batch_multiplicative_inverse(&xs);
                    assert_eq!(invs.len(), n);
                    for (x, inv) in xs.into_iter().zip(invs) {
                        assert_eq!(x * inv, <$field>::ONE);
                    }
                }
            }

            #[test]
            fn primitive_root_order() {
                let max_power = 8.min(<$field>::TWO_ADICITY);
                for n_power in 0..max_power {
                    let root = <$field>::primitive_root_of_unity(n_power);
                    let order = <$field>::generator_order(root);
                    assert_eq!(order, 1 << n_power, "2^{}'th primitive root", n_power);
                }
            }

            #[test]
            fn negation() {
                type F = $field;

                for x in [F::zero(), F::one(), F::two(), F::NEG_ONE] {
                    assert_eq!(x + -x, F::zero());
                }
            }

            #[test]
            fn exponentiation() {
                type F = $field;

                assert_eq!(F::zero().exp_u64(0), <F>::ONE);
                assert_eq!(F::one().exp_u64(0), <F>::ONE);
                assert_eq!(F::two().exp_u64(0), <F>::ONE);

                assert_eq!(F::zero().exp_u64(1), <F>::ZERO);
                assert_eq!(F::one().exp_u64(1), <F>::ONE);
                assert_eq!(F::two().exp_u64(1), <F>::TWO);

                assert_eq!(F::zero().kth_root_u64(1), <F>::ZERO);
                assert_eq!(F::one().kth_root_u64(1), <F>::ONE);
                assert_eq!(F::two().kth_root_u64(1), <F>::TWO);

                for power in 1..10 {
                    if F::is_monomial_permutation_u64(power) {
                        let x = F::rand();
                        assert_eq!(x.exp_u64(power).kth_root_u64(power), x);
                    }
                }
            }

            #[test]
            fn exponentiation_large() {
                type F = $field;

                let mut rng = OsRng;

                let base = F::rand();
                let pow = BigUint::from(rng.gen::<u64>());
                let cycles = rng.gen::<u32>();
                let mul_group_order = F::order() - 1u32;
                let big_pow = &pow + &mul_group_order * cycles;
                let big_pow_wrong = &pow + &mul_group_order * cycles + 1u32;

                assert_eq!(base.exp_biguint(&pow), base.exp_biguint(&big_pow));
                assert_ne!(base.exp_biguint(&pow), base.exp_biguint(&big_pow_wrong));
            }

            #[test]
            fn inverses() {
                type F = $field;

                let x = F::rand();
                let x1 = x.inverse();
                let x2 = x1.inverse();
                let x3 = x2.inverse();

                assert_eq!(x, x2);
                assert_eq!(x1, x3);
            }
        }
    };
}

#[allow(clippy::eq_op)]
pub(crate) fn test_add_neg_sub_mul<
    AF: AbstractField + BinomiallyExtendable<D> + Sample,
    const D: usize,
>() {
    let x = BinomialExtensionField::<AF, D>::rand();
    let y = BinomialExtensionField::<AF, D>::rand();
    let z = BinomialExtensionField::<AF, D>::rand();
    assert_eq!(x + (-x), BinomialExtensionField::<AF, D>::zeros());
    assert_eq!(-x, BinomialExtensionField::<AF, D>::zeros() - x);
    assert_eq!(x + x, x * BinomialExtensionField::<AF, D>::twos());
    assert_eq!(x * (-x), -AbstractField::square(&x));
    assert_eq!(x + y, y + x);
    assert_eq!(x * y, y * x);
    assert_eq!(x * (y * z), (x * y) * z);
    assert_eq!(x - (y + z), (x - y) - z);
    assert_eq!((x + y) - z, x + (y - z));
    assert_eq!(x * (y + z), x * y + x * z);
}

pub(crate) fn test_inv_div<AF: AbstractField + BinomiallyExtendable<D> + Sample, const D: usize>() {
    let x = BinomialExtensionField::<AF, D>::rand();
    let y = BinomialExtensionField::<AF, D>::rand();
    let z = BinomialExtensionField::<AF, D>::rand();
    assert_eq!(x * x.inverse(), BinomialExtensionField::<AF, D>::ones());
    assert_eq!(x.inverse() * x, BinomialExtensionField::<AF, D>::ones());
    assert_eq!(
        AbstractField::square(&x).inverse(),
        AbstractField::square(&x.inverse())
    );
    assert_eq!((x / y) * y, x);
    assert_eq!(x / (y * z), (x / y) / z);
    assert_eq!((x * y) / z, x * (y / z));
}

pub(crate) fn test_frobenius<
    AF: AbstractField + BinomiallyExtendable<D> + Sample,
    const D: usize,
>() {
    let x = BinomialExtensionField::<AF, D>::rand();
    assert_eq!(exp_biguint(x, &AF::order()), x.frobenius());
    for count in 2..D {
        assert_eq!(
            x.repeated_frobenius(count),
            (0..count).fold(x, |acc, _| acc.frobenius())
        );
    }
}

fn exp_biguint<AF: AbstractField + BinomiallyExtendable<D> + Sample, const D: usize>(
    x: BinomialExtensionField<AF, D>,
    power: &num::BigUint,
) -> BinomialExtensionField<AF, D> {
    let mut result = BinomialExtensionField::<AF, D>::ones();
    for &digit in power.to_u64_digits().iter().rev() {
        result = result.exp_power_of_2(64);
        result *= x.exp_u64(digit);
    }
    result
}

pub(crate) fn test_field_order<
    AF: AbstractField + BinomiallyExtendable<D> + Sample,
    const D: usize,
>() {
    let x = BinomialExtensionField::<AF, D>::rand();
    assert_eq!(
        exp_biguint::<AF, D>(x, &(AF::order() - 1u8)),
        BinomialExtensionField::<AF, D>::ones()
    );
}

pub(crate) fn test_power_of_two_gen<AF: AbstractField + TwoAdicField + HasTwoAdicBionmialExtension<D> + Sample, const D: usize>() {
    assert_eq!(
        exp_biguint(BinomialExtensionField::<AF, D>::from_base_slice(&AF::ext_generator())
            , &(BinomialExtensionField::<AF, D>::order() >> BinomialExtensionField::<AF, D>::TWO_ADICITY)),
            BinomialExtensionField::<AF,D>::from_base_slice(&AF::ext_two_adic_generator(BinomialExtensionField::<AF, D>::TWO_ADICITY)));
    assert_eq!(
        BinomialExtensionField::<AF,D>::from_base_slice(&AF::ext_two_adic_generator(BinomialExtensionField::<AF, D>::TWO_ADICITY))
            .exp_u64(1 << (BinomialExtensionField::<AF, D>::TWO_ADICITY - AF::TWO_ADICITY)),
            BinomialExtensionField::<AF,D>::from_base(AF::two_adic_generator(AF::TWO_ADICITY))
    );
}

#[macro_export]
macro_rules! test_field_extension {
    ($field:ty, $d:expr) => {
        mod field_extension {
            #[test]
            fn test_add_neg_sub_mul() {
                $crate::field_testing::test_add_neg_sub_mul::<$field, $d>();
            }
            #[test]
            fn test_inv_div() {
                $crate::field_testing::test_inv_div::<$field, $d>();
            }
            #[test]
            fn test_frobenius() {
                $crate::field_testing::test_frobenius::<$field, $d>();
            }
            #[test]
            fn test_field_order() {
                $crate::field_testing::test_field_order::<$field, $d>();
            }
            #[test]
            fn test_power_of_two_gen() {
                $crate::field_testing::test_power_of_two_gen::<$field, $d>();
            }
        }
    };
}

use p3_field::extension::{
    BinomialExtensionField, BinomiallyExtendable, HasFrobenius, HasTwoAdicBionmialExtension,
};
use p3_field::{AbstractExtensionField, AbstractField, Field, TwoAdicField};

use crate::packed::PackedField;
use crate::types::Sample;

#[macro_export]
macro_rules! test_field_arithmetic {
    ($field:ty) => {
        mod field_arithmetic {
            use alloc::vec::Vec;

            use p3_field::{batch_multiplicative_inverse, Field, AbstractField, TwoAdicField};
            use $crate::types::{Sample};


            #[test]
            fn batch_inversion() {

                for n in 0..20 {
                    let xs = (1..=n as u64)
                        .map(|i| <$field>::from_canonical_u64(i))
                        .collect::<Vec<_>>();

                    let invs = batch_multiplicative_inverse::<$field>(xs.as_slice());
                    assert_eq!(invs.len(), n);
                    for (x, inv) in xs.into_iter().zip(invs) {
                        assert_eq!(x * inv, <$field>::one());
                    }
                }
            }

            #[test]
            fn primitive_root_order() {
                let max_power = 8.min(<$field>::TWO_ADICITY);
                for n_power in 0..max_power {
                    let root = <$field>::two_adic_generator(n_power);
                    let order = root.powers().skip(1).position(|y| y.is_one()).unwrap() + 1;
                    assert_eq!(order, 1 << n_power, "2^{}'th primitive root", n_power);
                }
            }

            #[test]
            fn negation() {
                type F = $field;

                for x in [F::zero(), F::one(), F::two(), F::neg_one()] {
                    assert_eq!(x + -x, F::zero());
                }
            }

            #[test]
            fn exponentiation() {
                type F = $field;

                assert_eq!(F::zero().exp_u64(0), <F>::one());
                assert_eq!(F::one().exp_u64(0), <F>::one());
                assert_eq!(F::two().exp_u64(0), <F>::one());

                assert_eq!(F::zero().exp_u64(1), <F>::zero());
                assert_eq!(F::one().exp_u64(1), <F>::one());
                assert_eq!(F::two().exp_u64(1), <F>::two());
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

/// Test of consistency of the arithmetic operations.
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

/// Test of consistency of division.
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

/// Test that the Frobenius automorphism is consistent with the naive version.
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

/// Exponentiation of an extension field element by an arbitrary large integer.
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

/// Test that x^(|F| - 1) for a random (non-zero) x in F.
pub(crate) fn test_field_order<
    AF: AbstractField + BinomiallyExtendable<D> + Sample,
    const D: usize,
>() {
    let x = BinomialExtensionField::<AF, D>::rand();
    assert_eq!(
        exp_biguint::<AF, D>(x, &(BinomialExtensionField::<AF, D>::order() - 1u8)),
        BinomialExtensionField::<AF, D>::ones()
    );
}

/// Tests of consistency  the extension field generator,
/// the two_adicities of base and extension field and
/// the two_adic generators of the base field and the extension field.
pub(crate) fn test_power_of_two_gen<
    AF: AbstractField + TwoAdicField + HasTwoAdicBionmialExtension<D> + Sample,
    const D: usize,
>() {
    assert_eq!(
        exp_biguint(
            BinomialExtensionField::<AF, D>::from_base_slice(&AF::ext_generator()),
            &(BinomialExtensionField::<AF, D>::order()
                >> BinomialExtensionField::<AF, D>::TWO_ADICITY)
        ),
        BinomialExtensionField::<AF, D>::from_base_slice(&AF::ext_two_adic_generator(
            BinomialExtensionField::<AF, D>::TWO_ADICITY
        ))
    );
    assert_eq!(
        BinomialExtensionField::<AF, D>::from_base_slice(&AF::ext_two_adic_generator(
            BinomialExtensionField::<AF, D>::TWO_ADICITY
        ))
        .exp_u64(1 << (BinomialExtensionField::<AF, D>::TWO_ADICITY - AF::TWO_ADICITY)),
        BinomialExtensionField::<AF, D>::from_base(AF::two_adic_generator(AF::TWO_ADICITY))
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

#[cfg(test)]
mod tests {

    mod goldilocks {
        use crate::{test_field_arithmetic};
        test_field_arithmetic!(p3_goldilocks::Goldilocks);
    }
    mod goldilocks_ext {
        use crate::{test_field_arithmetic, test_field_extension};

        test_field_extension!(p3_goldilocks::Goldilocks, 2);
        test_field_arithmetic!(
            p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>
        );
    }
}
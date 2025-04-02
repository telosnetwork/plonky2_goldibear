use alloc::vec::Vec;

use p3_field::PrimeField64;

/// Generates a series of non-negative integers less than `modulus` which cover a range of
/// interesting test values.
fn test_inputs(modulus: u64) -> Vec<u64> {
    const CHUNK_SIZE: u64 = 10;

    (0..CHUNK_SIZE)
        .chain((1 << 31) - CHUNK_SIZE..(1 << 31) + CHUNK_SIZE)
        .chain((1 << 32) - CHUNK_SIZE..(1 << 32) + CHUNK_SIZE)
        .chain((1 << 63) - CHUNK_SIZE..(1 << 63) + CHUNK_SIZE)
        .chain(modulus - CHUNK_SIZE..modulus)
        .filter(|&x| x < modulus)
        .collect()
}

/// Apply the unary functions `op` and `expected_op`
/// coordinate-wise to the inputs from `test_inputs(modulus,
/// word_bits)` and panic if the two resulting vectors differ.
pub fn run_unaryop_test_cases<F, UnaryOp, ExpectedOp>(op: UnaryOp, expected_op: ExpectedOp)
where
    F: PrimeField64,
    UnaryOp: Fn(F) -> F,
    ExpectedOp: Fn(u64) -> u64,
{
    let inputs = test_inputs(F::ORDER_U64);
    let expected: Vec<_> = inputs.iter().map(|&x| expected_op(x)).collect();
    let output: Vec<_> = inputs
        .iter()
        .cloned()
        .map(|x| op(F::from_canonical_u64(x)).as_canonical_u64())
        .collect();
    // Compare expected outputs with actual outputs
    for i in 0..inputs.len() {
        assert_eq!(
            output[i], expected[i],
            "Expected {}, got {} for input {}",
            expected[i], output[i], inputs[i]
        );
    }
}

/// Apply the binary functions `op` and `expected_op` to each pair of inputs.
pub fn run_binaryop_test_cases<F, BinaryOp, ExpectedOp>(op: BinaryOp, expected_op: ExpectedOp)
where
    F: PrimeField64,
    BinaryOp: Fn(F, F) -> F,
    ExpectedOp: Fn(u64, u64) -> u64,
{
    let inputs = test_inputs(F::ORDER_U64);

    for &lhs in &inputs {
        for &rhs in &inputs {
            let lhs_f = F::from_canonical_u64(lhs);
            let rhs_f = F::from_canonical_u64(rhs);
            let actual = op(lhs_f, rhs_f).as_canonical_u64();
            let expected = expected_op(lhs, rhs);
            assert_eq!(
                actual, expected,
                "Expected {}, got {} for inputs ({}, {})",
                expected, actual, lhs, rhs
            );
        }
    }
}

#[macro_export]
macro_rules! test_prime_field_arithmetic {
    ($field:ty) => {
        mod prime_field_arithmetic {
            use core::ops::{Add, Mul, Neg, Sub};

            use p3_field::{AbstractField, Field, PrimeField64};

            #[test]
            fn arithmetic_addition() {
                let modulus = <$field>::ORDER_U64;
                $crate::prime_field_testing::run_binaryop_test_cases(<$field>::add, |x, y| {
                    ((x as u128 + y as u128) % (modulus as u128)) as u64
                })
            }

            #[test]
            fn arithmetic_subtraction() {
                let modulus = <$field>::ORDER_U64;
                $crate::prime_field_testing::run_binaryop_test_cases(<$field>::sub, |x, y| {
                    if x >= y {
                        x - y
                    } else {
                        modulus - y + x
                    }
                })
            }

            #[test]
            fn arithmetic_negation() {
                let modulus = <$field>::ORDER_U64;
                $crate::prime_field_testing::run_unaryop_test_cases(<$field>::neg, |x| {
                    if x == 0 {
                        0
                    } else {
                        modulus - x
                    }
                })
            }

            #[test]
            fn arithmetic_multiplication() {
                let modulus = <$field>::ORDER_U64;
                $crate::prime_field_testing::run_binaryop_test_cases(<$field>::mul, |x, y| {
                    ((x as u128) * (y as u128) % (modulus as u128)) as u64
                })
            }

            #[test]
            fn arithmetic_square() {
                let modulus = <$field>::ORDER_U64;
                $crate::prime_field_testing::run_unaryop_test_cases(
                    |x: $field| AbstractField::square(&x),
                    |x| ((x as u128 * x as u128) % (modulus as u128)) as u64,
                )
            }

            #[test]
            fn inversion() {
                let zero = <$field>::zero();
                let one = <$field>::one();
                let modulus = <$field>::ORDER_U64;

                assert_eq!(zero.try_inverse(), None);

                let inputs = $crate::prime_field_testing::test_inputs(modulus);

                for x in inputs {
                    if x != 0 {
                        let x = <$field>::from_canonical_u64(x);
                        let inv = x.inverse();
                        assert_eq!(x * inv, one);
                    }
                }
            }

            #[test]
            fn subtraction_double_wraparound() {
                type F = $field;

                let (a, b) = (
                    F::from_canonical_u64((F::ORDER_U64).div_ceil(2u64)),
                    F::two(),
                );
                let x = a * b;
                assert_eq!(x, F::one());
                assert_eq!(F::zero() - x, F::neg_one());
            }

            #[test]
            fn addition_double_wraparound() {
                type F = $field;

                let a = F::from_canonical_u64(u64::MAX % F::ORDER_U64);
                let b = F::neg_one();

                let c = (a + a) + (b + b);
                let d = (a + b) + (a + b);

                assert_eq!(c, d);
            }
        }
    };
}

#[cfg(test)]
mod tests {
    mod goldilocks {
        use crate::{test_field_arithmetic, test_prime_field_arithmetic};

        test_prime_field_arithmetic!(p3_goldilocks::Goldilocks);
        test_field_arithmetic!(p3_goldilocks::Goldilocks);
    }
    mod babybear {
        use crate::{test_field_arithmetic, test_prime_field_arithmetic};

        test_prime_field_arithmetic!(p3_baby_bear::BabyBear);
        test_field_arithmetic!(p3_baby_bear::BabyBear);
    }
}

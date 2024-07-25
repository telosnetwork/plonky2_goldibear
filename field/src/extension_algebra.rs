use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField64};
use p3_field::extension::{BinomialExtensionField};

use crate::types::HasExtension;

/// Let `F_D` be the optimal extension field `F[X]/(X^D-W)`. Then `ExtensionAlgebra<F_D>` is the quotient `F_D[X]/(X^D-W)`.
/// It's a `D`-dimensional algebra over `F_D` useful to lift the multiplication over `F_D` to a multiplication over `(F_D)^D`.
#[derive(Copy, Clone)]
pub struct ExtensionAlgebra<F: HasExtension<D>, const D: usize>(
    pub [F::Extension; D],
);

impl<F: HasExtension<D>, const D: usize> ExtensionAlgebra<F, D> {
    pub fn zero() -> Self {
        Self([<F::Extension as AbstractExtensionField<F>>::from_base(F::zero()); D])
    }

    pub fn one() -> Self {
        let mut res = Self::zero();
        res.0[0] = <F::Extension as AbstractExtensionField<F>>::from_base(F::one());
        res
    }

    pub const fn from_basefield_array(arr: [F::Extension; D]) -> Self {
        Self(arr)
    }

    pub const fn to_basefield_array(self) -> [F::Extension; D] {
        self.0
    }

    pub fn scalar_mul(&self, scalar: F::Extension) -> Self {
        let mut res = self.0;
        res.iter_mut().for_each(|x| {
            *x *= scalar;
        });
        Self(res)
    }

    pub fn from_base(x: F::Extension) -> Self {
        let mut arr = [<F::Extension as AbstractField>::zero(); D];
        arr[1] = x;
        Self(arr)
    }
}

// impl<F: HasExtension<D>, const D: usize> From<F::Extension>
//     for ExtensionAlgebra<F, D>
// {
//     fn from(x: F::Extension) -> Self {
//         let mut arr =
//             [F::Extension::from_base(F::zero()); D];
//         arr[0] = x;
//         Self(arr)
//     }
// }

impl<F: HasExtension<D>, const D: usize> Display for ExtensionAlgebra<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({})", self.0[0])?;
        for i in 1..D {
            write!(f, " + ({})*b^{i}", self.0[i])?;
        }
        Ok(())
    }
}

impl<F: HasExtension<D>, const D: usize> Debug for ExtensionAlgebra<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: HasExtension<D>, const D: usize> Neg for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        let mut arr = self.0;
        arr.iter_mut().for_each(|x| *x = -*x);
        Self(arr)
    }
}

impl<F: HasExtension<D>, const D: usize> Add for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = self.0;
        arr.iter_mut().zip(&rhs.0).for_each(|(x, &y)| *x += y);
        Self(arr)
    }
}

impl<F: HasExtension<D>, const D: usize> AddAssign for ExtensionAlgebra<F, D> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: HasExtension<D>, const D: usize> Sum for ExtensionAlgebra<F, D> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<F: HasExtension<D>, const D: usize> Sub for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = self.0;
        arr.iter_mut().zip(&rhs.0).for_each(|(x, &y)| *x -= y);
        Self(arr)
    }
}

impl<F: HasExtension<D>, const D: usize> SubAssign for ExtensionAlgebra<F, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: HasExtension<D>, const D: usize> Mul for ExtensionAlgebra<F, D> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::zero();
        let w = <F::Extension as AbstractExtensionField<F>>::from_base(F::w());
        for i in 0..D {
            for j in 0..D {
                res.0[(i + j) % D] += if i + j < D {
                    self.0[i] * rhs.0[j]
                } else {
                    w * self.0[i] * rhs.0[j]
                }
            }
        }
        res
    }
}

impl<F: HasExtension<D>, const D: usize> MulAssign for ExtensionAlgebra<F, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: HasExtension<D>, const D: usize> Product for ExtensionAlgebra<F, D> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

/// A polynomial in coefficient form.
#[derive(Clone, Debug)]
pub struct PolynomialCoeffsAlgebra<F: HasExtension<D>, const D: usize> {
    pub(crate) coeffs: Vec<ExtensionAlgebra<F, D>>,
}

impl<F: HasExtension<D>, const D: usize> PolynomialCoeffsAlgebra<F, D> {
    pub fn new(coeffs: Vec<ExtensionAlgebra<F, D>>) -> Self {
        PolynomialCoeffsAlgebra { coeffs }
    }

    pub fn eval(&self, x: ExtensionAlgebra<F, D>) -> ExtensionAlgebra<F, D> {
        self.coeffs
            .iter()
            .rev()
            .fold(ExtensionAlgebra::zero(), |acc, &c| acc * x + c)
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_with_powers(&self, powers: &[ExtensionAlgebra<F, D>]) -> ExtensionAlgebra<F, D> {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + c * x)
    }

    pub fn eval_base(&self, x: F::Extension) -> ExtensionAlgebra<F, D> {
        self.coeffs
            .iter()
            .rev()
            .fold(ExtensionAlgebra::zero(), |acc, &c| acc.scalar_mul(x) + c)
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_base_with_powers(&self, powers: &[F::Extension]) -> ExtensionAlgebra<F, D> {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + x.scalar_mul(c))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use itertools::Itertools;
    use p3_field::AbstractExtensionField;
    use p3_field::extension::BinomialExtensionField;

    use crate::extension_algebra::ExtensionAlgebra;
    use crate::types::{HasExtension, Sample};

    /// Tests that the multiplication on the extension algebra lifts that of the field extension.
    fn test_extension_algebra<F: HasExtension<D> + Sample, const D: usize>()
    where
        F::Extension: Sample {
        #[derive(Copy, Clone, Debug)]
        enum ZeroOne {
            Zero,
            One,
        }

        let to_field = |zo: &ZeroOne| match zo {
            ZeroOne::Zero => F::zero(),
            ZeroOne::One => F::one(),
        };
        let to_fields = |x: &[ZeroOne],
                         y: &[ZeroOne]|
         -> (F::Extension, F::Extension) {
            let mut arr0 = [F::zero(); D];
            let mut arr1 = [F::zero(); D];
            arr0.copy_from_slice(&x.iter().map(to_field).collect::<Vec<_>>());
            arr1.copy_from_slice(&y.iter().map(to_field).collect::<Vec<_>>());
            (
                <F::Extension as AbstractExtensionField<F>>::from_base_slice(&arr0),
                <F::Extension as AbstractExtensionField<F>>::from_base_slice(&arr1),
            )
        };

        // Standard MLE formula.
        let selector = |xs: Vec<ZeroOne>,
                        ts: &[F::Extension]|
         -> F::Extension {
            (0..2 * D)
                .map(|i| match xs[i] {
                    ZeroOne::Zero => {
                        <F::Extension as AbstractExtensionField<F>>::from_base(
                            F::one(),
                        ) - ts[i]
                    }
                    ZeroOne::One => ts[i],
                })
                .product()
        };

        let mul_mle = |ts: Vec<F::Extension>| -> [F::Extension; D] {
            let mut ans =
                [<F::Extension as AbstractExtensionField<F>>::from_base(F::zero());
                    D];
            for xs in (0..2 * D)
                .map(|_| vec![ZeroOne::Zero, ZeroOne::One])
                .multi_cartesian_product()
            {
                let (a, b) = to_fields(&xs[..D], &xs[D..]);
                let c = a * b;
                let res = selector(xs, &ts);
                let c_slice: &[F] = c.as_base_slice();
                for i in 0..D {
                    ans[i] += res
                        * <F::Extension as AbstractExtensionField<F>>::from_base(
                            c_slice[i],
                        );
                }
            }
            ans
        };

        let ts = <F::Extension as Sample>::rand_vec(2 * D);
        let mut arr0 =
            [<F::Extension as AbstractExtensionField<F>>::from_base(F::zero()); D];
        let mut arr1 =
            [<F::Extension as AbstractExtensionField<F>>::from_base(F::zero()); D];
        arr0.copy_from_slice(&ts[..D]);
        arr1.copy_from_slice(&ts[D..]);
        let x: ExtensionAlgebra<F, D> = ExtensionAlgebra::from_basefield_array(arr0);
        let y: ExtensionAlgebra<F, D> = ExtensionAlgebra::from_basefield_array(arr1);
        let z: ExtensionAlgebra<F, D> = x * y;

        assert_eq!(z.0, mul_mle(ts));
    }

    mod quadratic {
        use p3_goldilocks::Goldilocks;

        use super::*;

        #[test]
        fn test_algebra() {
            test_extension_algebra::<Goldilocks, 2>();
        }
    }

    mod quartic {
        use p3_baby_bear::BabyBear;

        use super::*;

        #[test]
        fn test_algebra() {
            test_extension_algebra::<BabyBear, 4>();
        }
    }
}

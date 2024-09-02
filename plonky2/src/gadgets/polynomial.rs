#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use p3_field::TwoAdicField;
use plonky2_field::types::HasExtension;

use crate::hash::hash_types::RichField;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::util::reducing::ReducingFactorTarget;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialCoeffsExtTarget<const D: usize>(pub Vec<ExtensionTarget<D>>);

impl<const D: usize> PolynomialCoeffsExtTarget<D> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn eval_scalar<F: RichField + HasExtension<D>, const NUM_HASH_OUT_ELTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        point: Target,
    ) -> ExtensionTarget<D>
    where
        F::Extension: TwoAdicField,
    {
        let point = builder.convert_to_ext(point);
        let mut point = ReducingFactorTarget::new(point);
        point.reduce(&self.0, builder)
    }

    pub fn eval<F: RichField + HasExtension<D>, const NUM_HASH_OUT_ELTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        point: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        F::Extension: TwoAdicField,
    {
        let mut point = ReducingFactorTarget::new(point);
        point.reduce(&self.0, builder)
    }
}

#[derive(Debug)]
pub struct PolynomialCoeffsExtAlgebraTarget<const D: usize>(pub Vec<ExtensionAlgebraTarget<D>>);

impl<const D: usize> PolynomialCoeffsExtAlgebraTarget<D> {
    pub fn eval_scalar<F, const NUM_HASH_OUT_ELTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        point: ExtensionTarget<D>,
    ) -> ExtensionAlgebraTarget<D>
    where
        F: RichField + HasExtension<D>,
        F::Extension: TwoAdicField,
    {
        let mut acc = builder.zero_ext_algebra();
        for &c in self.0.iter().rev() {
            acc = builder.scalar_mul_add_ext_algebra(point, acc, c);
        }
        acc
    }

    pub fn eval<F, const NUM_HASH_OUT_ELTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        point: ExtensionAlgebraTarget<D>,
    ) -> ExtensionAlgebraTarget<D>
    where
        F: RichField + HasExtension<D>,
        F::Extension: TwoAdicField,
    {
        let mut acc = builder.zero_ext_algebra();
        for &c in self.0.iter().rev() {
            acc = builder.mul_add_ext_algebra(point, acc, c);
        }
        acc
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_with_powers<F, const NUM_HASH_OUT_ELTS: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        powers: &[ExtensionAlgebraTarget<D>],
    ) -> ExtensionAlgebraTarget<D>
    where
        F: RichField + HasExtension<D>,
        F::Extension: TwoAdicField,
    {
        debug_assert_eq!(self.0.len(), powers.len() + 1);
        let acc = self.0[0];
        self.0[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| builder.mul_add_ext_algebra(c, x, acc))
    }
}

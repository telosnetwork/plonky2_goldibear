use p3_field::TwoAdicField;
use plonky2_field::types::HasExtension;

use crate::hash::hash_types::RichField;
use crate::iop::target::BoolTarget;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::AlgebraicHasher;

impl<F: RichField + HasExtension<D>, const D: usize> CircuitBuilder<F, D>
where
    F::Extension: TwoAdicField,
{
    pub fn permute<H: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>, const NUM_HASH_OUT_ELTS: usize>(
        &mut self,
        inputs: H::AlgebraicPermutation,
    ) -> H::AlgebraicPermutation {
        // We don't want to swap any inputs, so set that wire to 0.
        let _false = self._false();
        self.permute_swapped::<H, NUM_HASH_OUT_ELTS>(inputs, _false)
    }

    /// Conditionally swap two chunks of the inputs (useful in verifying Merkle proofs), then apply
    /// a cryptographic permutation.
    pub(crate) fn permute_swapped<H: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>, const NUM_HASH_OUT_ELTS: usize>(
        &mut self,
        inputs: H::AlgebraicPermutation,
        swap: BoolTarget,
    ) -> H::AlgebraicPermutation {
        H::permute_swapped(inputs, swap, self)
    }
}

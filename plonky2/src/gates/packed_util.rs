#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use p3_field::{PackedValue, TwoAdicField};
use plonky2_field::types::HasExtension;

use p3_field::PackedField;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::plonk::vars::{EvaluationVarsBaseBatch, EvaluationVarsBasePacked};

pub trait PackedEvaluableBase<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>: Gate<F, D, NUM_HASH_OUT_ELTS>
where
    F::Extension: TwoAdicField,
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars_base: EvaluationVarsBasePacked<P, NUM_HASH_OUT_ELTS>,
        yield_constr: StridedConstraintConsumer<P>,
    );

    /// Evaluates entire batch of points. Returns a matrix of constraints. Constraint `j` for point
    /// `i` is at `index j * batch_size + i`.
    fn eval_unfiltered_base_batch_packed(&self, vars_batch: EvaluationVarsBaseBatch<F, NUM_HASH_OUT_ELTS>) -> Vec<F> {
        let mut res = vec![F::zero(); vars_batch.len() * <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self)];
        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<F::Packing>();
        let leftovers_start = vars_batch.len() - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new(
                    &mut res[..],
                    vars_batch.len(),
                    F::Packing::WIDTH * i,
                ),
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new(&mut res[..], vars_batch.len(), leftovers_start + i),
            );
        }
        res
    }
}

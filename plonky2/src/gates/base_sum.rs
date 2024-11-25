#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};
use core::ops::Range;

use p3_field::{AbstractField, PackedField, PrimeField64};

use plonky2_field::types::HasExtension;

use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::log_floor;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate which can decompose a number into base B little-endian limbs.
#[derive(Copy, Clone, Debug)]
pub struct BaseSumGate<const B: usize> {
    pub num_limbs: usize,
}

impl<const B: usize> BaseSumGate<B> {
    pub const fn new(num_limbs: usize) -> Self {
        Self { num_limbs }
    }

    pub fn new_from_config<F: PrimeField64>(config: &CircuitConfig) -> Self {
        let num_limbs =
            log_floor(F::ORDER_U64 - 1, B as u64).min(config.num_routed_wires - Self::START_LIMBS);
        Self::new(num_limbs)
    }

    pub(crate) const WIRE_SUM: usize = 0;
    pub(crate) const START_LIMBS: usize = 1;

    /// Returns the index of the `i`th limb wire.
    pub(crate) const fn limbs(&self) -> Range<usize> {
        Self::START_LIMBS..Self::START_LIMBS + self.num_limbs
    }
}

impl<
        F: RichField + HasExtension<D>,
        const D: usize,
        const B: usize,
        const NUM_HASH_OUT_ELTS: usize,
    > Gate<F, D, NUM_HASH_OUT_ELTS> for BaseSumGate<B>
where
    
{
    fn id(&self) -> String {
        format!("{self:?} + Base: {B}")
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.num_limbs)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let num_limbs = src.read_usize()?;
        Ok(Self { num_limbs })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>) -> Vec<F::Extension> {
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires[self.limbs()].to_vec();
        let computed_sum = reduce_with_powers(
            &limbs,
            <F::Extension as AbstractField>::from_canonical_usize(B),
        );
        let mut constraints = vec![computed_sum - sum];
        for limb in limbs {
            constraints.push(
                (0..B)
                    .map(|i| limb - <F::Extension as AbstractField>::from_canonical_usize(i))
                    .product(),
            );
        }
        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(
        &self,
        vars_base: EvaluationVarsBaseBatch<F, NUM_HASH_OUT_ELTS>,
    ) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        let base = builder.constant(F::from_canonical_usize(B));
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires[self.limbs()].to_vec();
        let computed_sum = reduce_with_powers_ext_circuit(builder, &limbs, base);
        let mut constraints = vec![builder.sub_extension(computed_sum, sum)];
        for limb in limbs {
            constraints.push({
                let mut acc = builder.one_extension();
                (0..B).for_each(|i| {
                    // We update our accumulator as:
                    // acc' = acc (x - i)
                    //      = acc x + (-i) acc
                    // Since -i is constant, we can do this in one arithmetic_extension call.
                    let neg_i = -F::from_canonical_usize(i);
                    acc = builder.arithmetic_extension(F::one(), neg_i, acc, limb, acc)
                });
                acc
            });
        }
        constraints
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        let gen = BaseSplitGenerator::<B> {
            row,
            num_limbs: self.num_limbs,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    // 1 for the sum then `num_limbs` for the limbs.
    fn num_wires(&self) -> usize {
        1 + self.num_limbs
    }

    fn num_constants(&self) -> usize {
        0
    }

    // Bounded by the range-check (x-0)*(x-1)*...*(x-B+1).
    fn degree(&self) -> usize {
        B
    }

    // 1 for checking the sum then `num_limbs` for range-checking the limbs.
    fn num_constraints(&self) -> usize {
        1 + self.num_limbs
    }
}

impl<
        F: RichField + HasExtension<D>,
        const D: usize,
        const B: usize,
        const NUM_HASH_OUT_ELTS: usize,
    > PackedEvaluableBase<F, D, NUM_HASH_OUT_ELTS> for BaseSumGate<B>
where
    
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires.view(self.limbs());
        let computed_sum = reduce_with_powers(limbs, F::from_canonical_usize(B));

        yield_constr.one(computed_sum - sum);

        let constraints_iter = limbs.iter().map(|&limb| {
            (0..B)
                .map(|i| limb - F::from_canonical_usize(i))
                .product::<P>()
        });
        yield_constr.many(constraints_iter);
    }
}

#[derive(Debug, Default)]
pub struct BaseSplitGenerator<const B: usize> {
    row: usize,
    num_limbs: usize,
}

impl<
        F: RichField + HasExtension<D>,
        const B: usize,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    > SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for BaseSplitGenerator<B>
where
    
{
    fn id(&self) -> String {
        format!("BaseSplitGenerator + Base: {B}")
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(self.row, BaseSumGate::<B>::WIRE_SUM)]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let sum_value = witness
            .get_target(Target::wire(self.row, BaseSumGate::<B>::WIRE_SUM))
            .as_canonical_u64() as usize;
        debug_assert_eq!(
            (0..self.num_limbs).fold(sum_value, |acc, _| acc / B),
            0,
            "Integer too large to fit in given number of limbs"
        );

        let limbs = (BaseSumGate::<B>::START_LIMBS..BaseSumGate::<B>::START_LIMBS + self.num_limbs)
            .map(|i| Target::wire(self.row, i));
        let limbs_value = (0..self.num_limbs)
            .scan(sum_value, |acc, _| {
                let tmp = *acc % B;
                *acc /= B;
                Some(F::from_canonical_usize(tmp))
            })
            .collect::<Vec<_>>();

        for (b, b_value) in limbs.zip(limbs_value) {
            out_buffer.set_target(b, b_value);
        }
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.num_limbs)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let row = src.read_usize()?;
        let num_limbs = src.read_usize()?;
        Ok(Self { row, num_limbs })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_goldilocks::Goldilocks;

    use crate::gates::base_sum::BaseSumGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::GOLDILOCKS_NUM_HASH_OUT_ELTS;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<Goldilocks, _, 2, 4>(BaseSumGate::<6>::new(11))
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(BaseSumGate::<6>::new(11))
    }
}

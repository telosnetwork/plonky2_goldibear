#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
use core::ops::Range;

use p3_field::{AbstractExtensionField, PackedField};

use plonky2_field::types::HasExtension;

use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::WitnessGeneratorRef;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult};

/// A gate whose first four wires will be equal to a hash of public inputs.
#[derive(Debug)]
pub struct PublicInputGate;

impl PublicInputGate {
    pub(crate) const fn wires_public_inputs_hash() -> Range<usize> {
        0..4
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for PublicInputGate
where

{
    fn id(&self) -> String {
        "PublicInputGate".into()
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(
        _src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        Ok(Self)
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>) -> Vec<F::Extension> {
        Self::wires_public_inputs_hash()
            .zip(vars.public_inputs_hash.elements)
            .map(|(wire, hash_part)| vars.local_wires[wire] - F::Extension::from_base(hash_part))
            .collect()
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
        Self::wires_public_inputs_hash()
            .zip(vars.public_inputs_hash.elements)
            .map(|(wire, hash_part)| {
                let hash_part_ext = builder.convert_to_ext(hash_part);
                builder.sub_extension(vars.local_wires[wire], hash_part_ext)
            })
            .collect()
    }

    fn generators(
        &self,
        _row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        Vec::new()
    }

    fn num_wires(&self) -> usize {
        4
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        4
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    PackedEvaluableBase<F, D, NUM_HASH_OUT_ELTS> for PublicInputGate
where

{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        yield_constr.many(
            Self::wires_public_inputs_hash()
                .zip(vars.public_inputs_hash.elements)
                .map(|(wire, hash_part)| vars.local_wires[wire] - hash_part),
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;

    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::public_input::PublicInputGate;
    use crate::hash::hash_types::GOLDILOCKS_NUM_HASH_OUT_ELTS;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<Goldilocks, _, 2, 4>(PublicInputGate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(PublicInputGate)
    }
}

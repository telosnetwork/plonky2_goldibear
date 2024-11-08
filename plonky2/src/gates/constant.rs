#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use p3_field::PackedField;
use serde::{Deserialize, Serialize};

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
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate which takes a single constant parameter and outputs that value.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ConstantGate {
    pub(crate) num_consts: usize,
}

impl ConstantGate {
    pub const fn new(num_consts: usize) -> Self {
        Self { num_consts }
    }

    const fn const_input(&self, i: usize) -> usize {
        debug_assert!(i < self.num_consts);
        i
    }

    const fn wire_output(&self, i: usize) -> usize {
        debug_assert!(i < self.num_consts);
        i
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for ConstantGate
where

{
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.num_consts)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let num_consts = src.read_usize()?;
        Ok(Self { num_consts })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>) -> Vec<F::Extension> {
        (0..self.num_consts)
            .map(|i| {
                vars.local_constants[self.const_input(i)] - vars.local_wires[self.wire_output(i)]
            })
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
        (0..self.num_consts)
            .map(|i| {
                builder.sub_extension(
                    vars.local_constants[self.const_input(i)],
                    vars.local_wires[self.wire_output(i)],
                )
            })
            .collect()
    }

    fn generators(
        &self,
        _row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        vec![]
    }

    fn num_wires(&self) -> usize {
        self.num_consts
    }

    fn num_constants(&self) -> usize {
        self.num_consts
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        self.num_consts
    }

    fn extra_constant_wires(&self) -> Vec<(usize, usize)> {
        (0..self.num_consts)
            .map(|i| (self.const_input(i), self.wire_output(i)))
            .collect()
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    PackedEvaluableBase<F, D, NUM_HASH_OUT_ELTS> for ConstantGate
where

{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        yield_constr.many((0..self.num_consts).map(|i| {
            vars.local_constants[self.const_input(i)] - vars.local_wires[self.wire_output(i)]
        }));
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_goldilocks::Goldilocks;

    use crate::gates::constant::ConstantGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::GOLDILOCKS_NUM_HASH_OUT_ELTS;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let num_consts = CircuitConfig::standard_recursion_config_gl().num_constants;
        let gate = ConstantGate { num_consts };
        test_low_degree::<Goldilocks, _, 2, 4>(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let num_consts = CircuitConfig::standard_recursion_config_gl().num_constants;
        let gate = ConstantGate { num_consts };
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }
}

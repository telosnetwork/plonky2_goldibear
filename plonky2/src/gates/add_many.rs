#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};
use core::ops::Range;

use p3_field::{AbstractField, PrimeField64, TwoAdicField};
use plonky2_field::types::HasExtension;

use crate::field::packed::PackedField;
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
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate which can decompose a number into base B little-endian limbs.
#[derive(Copy, Clone, Debug)]
pub struct AddManyGate<const NUM_ADDENDS: usize> {
    pub num_ops: usize,
}

impl<const NUM_ADDENDS: usize> AddManyGate<NUM_ADDENDS> {
    pub const fn new(num_ops: usize) -> Self {
        Self { num_ops }
    }

    pub fn new_from_config<F: PrimeField64>(config: &CircuitConfig) -> Self {
        let wires_per_op = NUM_ADDENDS + 1;
        let num_ops = config.num_routed_wires / wires_per_op;
        Self::new(num_ops)
    }

    pub(crate) const fn wires_ith_op_addends(i: usize) -> Range<usize>
    {
        (NUM_ADDENDS + 1) * i..(NUM_ADDENDS + 1) * i + NUM_ADDENDS
    }

    pub(crate) const fn wire_ith_sum(i: usize) -> usize
    {
        (NUM_ADDENDS + 1) * i + NUM_ADDENDS
    }


}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize, const NUM_ADDENDS: usize> Gate<F, D, NUM_HASH_OUT_ELTS> for AddManyGate<NUM_ADDENDS>
where
    F::Extension: TwoAdicField,
{
    fn id(&self) -> String {
        format!("{self:?} + Number of addends: {NUM_ADDENDS}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>) -> Vec<F::Extension> {
        let mut constraints: Vec<F::Extension> = vec![];
        (0..self.num_ops).for_each(|i| {
            let computed_sum = Self::wires_ith_op_addends(i)
                .map(|j| vars.local_wires[j])
                .fold(F::Extension::zero(), |acc,i| acc + i);
            constraints.push(computed_sum - vars.local_wires[Self::wire_ith_sum(i)]);
        });
        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        (0..self.num_ops).for_each(|i| {
            let computed_sum = Self::wires_ith_op_addends(i)
                .map(|j| vars.local_wires[j])
                .fold(F::zero(), |acc,i| acc + i);
            yield_constr.one(computed_sum - vars.local_wires[Self::wire_ith_sum(i)]);
        });
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints: Vec<ExtensionTarget<D>> = vec![];
        (0..self.num_ops).for_each(|i| {
            let addends = Self::wires_ith_op_addends(i)
                .map(|j| vars.local_wires[j]);
            let computed_sum = builder.add_many_extension(addends);
            let sum = vars.local_wires[Self::wire_ith_sum(i)];
            constraints.push(builder.sub_extension(computed_sum, sum));
        });
        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    AddManyGenerator::<NUM_ADDENDS> {
                        row,
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops *(1 + NUM_ADDENDS)
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        self.num_ops
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize, const NUM_ADDENDS: usize> PackedEvaluableBase<F, D, NUM_HASH_OUT_ELTS>
    for AddManyGate<NUM_ADDENDS>
where
    F::Extension: TwoAdicField,
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) { 
        (0..self.num_ops).for_each(|i| {
            let computed_sum = Self::wires_ith_op_addends(i)
                .map(|j| vars.local_wires[j])
                .fold(P::zeros(), |acc,i| acc + i);
            yield_constr.one(computed_sum - vars.local_wires[Self::wire_ith_sum(i)]);
        });
    }
}

#[derive(Debug, Default)]
pub struct AddManyGenerator<const NUM_ADDENDS: usize> {
    row: usize,
    i: usize,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize, const NUM_ADDENDS: usize> SimpleGenerator<F, D, NUM_HASH_OUT_ELTS>
    for AddManyGenerator<NUM_ADDENDS>
where
    F::Extension: TwoAdicField,
{
    fn id(&self) -> String {
        "AddManyGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        AddManyGate::<NUM_ADDENDS>::wires_ith_op_addends(self.i).map(|j| Target::wire(self.row, j)).collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let addends = AddManyGate::<NUM_ADDENDS>::wires_ith_op_addends(self.i).map(get_wire);
        let sum_target = Target::wire(self.row, AddManyGate::<NUM_ADDENDS>::wire_ith_sum(self.i));
        let computed_sum = addends.fold(F::zero(), |acc,i| acc + i);
        out_buffer.set_target(sum_target, computed_sum)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_goldilocks::Goldilocks;

    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config_gl());
        test_low_degree::<Goldilocks, _, 2, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = 4;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config_gl());
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }
}


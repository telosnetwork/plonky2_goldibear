use core::marker::PhantomData;
use core::ops::Range;

use itertools::Itertools;
use p3_field::{AbstractExtensionField, AbstractField};

use plonky2_field::extension_algebra::ExtensionAlgebra;
use plonky2_field::types::HasExtension;

use crate::hash::hash_types::RichField;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

use super::gate::Gate;

/// Apply Mat4 Gate
#[derive(Clone, Debug, Default)]
pub struct ApplyMat4Gate<F: RichField + HasExtension<D>, const D: usize> {
    num_ops: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + HasExtension<D>, const D: usize> ApplyMat4Gate<F, D>
where
    
{
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        let wires_per_op = 8 * D;
        let num_ops = config.num_routed_wires / wires_per_op;
        Self {
            num_ops,
            _phantom: PhantomData,
        }
    }

    pub(crate) const fn wires_input(op: usize, i: usize) -> Range<usize> {
        assert!(i < 4);
        op * 8 * D + i * D..op * 8 * D + (i + 1) * D
    }

    pub(crate) const fn wires_output(op: usize, i: usize) -> Range<usize> {
        assert!(i < 4);
        op * 8 * D + (4 + i) * D..op * 8 * D + (4 + i + 1) * D
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for ApplyMat4Gate<F, D>
where
    
{
    fn id(&self) -> String {
        format!("{self:?} number of operations = {}", self.num_ops)
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &crate::plonk::circuit_data::CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> crate::util::serialization::IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(
        src: &mut crate::util::serialization::Buffer,
        _common_data: &crate::plonk::circuit_data::CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> crate::util::serialization::IoResult<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            num_ops: src.read_usize()?,
            _phantom: PhantomData,
        })
    }

    fn eval_unfiltered(
        &self,
        vars: crate::plonk::vars::EvaluationVars<F, D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<<F as HasExtension<D>>::Extension> {
        let mut constraints = vec![];
        (0..self.num_ops).for_each(|op| {
            let mut x: [ExtensionAlgebra<F, D>; 4] = (0..4)
                .map(|i| vars.get_local_ext_algebra(Self::wires_input(op, i)))
                .collect_vec()
                .try_into()
                .unwrap();
            let t01 = x[0] + x[1];
            let t23 = x[2] + x[3];
            let t0123 = t01 + t23;
            let t01123 = t0123 + x[1];
            let t01233 = t0123 + x[3];
            x[3] = t01233 + x[0].scalar_mul(F::Extension::two());
            x[1] = t01123 + x[2].scalar_mul(F::Extension::two());
            x[0] = t01123 + t01;
            x[2] = t01233 + t23;
            constraints.extend_from_slice(
                &(0..4)
                    .map(|i| vars.get_local_ext_algebra(Self::wires_output(op, i)))
                    .zip(x)
                    .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array())
                    .collect_vec(),
            )
        });
        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        mut yield_constr: super::util::StridedConstraintConsumer<F>,
    ) {
        (0..self.num_ops).for_each(|op| {
            let mut x: [F::Extension; 4] = (0..4)
                .map(|i| vars_base.get_local_ext(Self::wires_input(op, i)))
                .collect_vec()
                .try_into()
                .unwrap();
            let t01 = x[0] + x[1];
            let t23 = x[2] + x[3];
            let t0123 = t01 + t23;
            let t01123 = t0123 + x[1];
            let t01233 = t0123 + x[3];
            x[3] = t01233 + x[0].double();
            x[1] = t01123 + x[2].double();
            x[0] = t01123 + t01;
            x[2] = t01233 + t23;
            (0..4).for_each(|i| {
                let output: F::Extension = vars_base.get_local_ext(Self::wires_output(op, i));
                yield_constr.many::<[F; D]>((output - x[i]).as_base_slice().try_into().unwrap());
            });
        });
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut crate::plonk::circuit_builder::CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: crate::plonk::vars::EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<crate::iop::ext_target::ExtensionTarget<D>> {
        let mut constraints = vec![];
        // let gate = Self::new_from_config(&builder.config);
        (0..self.num_ops).for_each(|op| {
            let mut x: [ExtensionAlgebraTarget<D>; 4] = (0..4)
                .map(|i| vars.get_local_ext_algebra(Self::wires_input(op, i)))
                .collect_vec()
                .try_into()
                .unwrap();
            // for i in 0..D {
            //     let input: [ExtensionTarget<D>; 4] =
            //         (0..4).map(|j| x[j].0[i]).collect_vec().try_into().unwrap();
            //     let (row, op) = builder.find_slot(gate.clone(), &[], &[]);
            //     (0..4).for_each(|j| {
            //         builder.connect_extension(
            //             input[j],
            //             ExtensionTarget::<D>::from_range(
            //                 row,
            //                 ApplyMat4Gate::<F, D>::wires_input(op, i),
            //             ),
            //         );
            //         x[j].0[i] = ExtensionTarget::<D>::from_range(
            //             row,
            //             ApplyMat4Gate::<F, D>::wires_output(op, i),
            //         );
            //     });
            // }
            let two = builder
                .constant_ext_algebra(ExtensionAlgebra::<F, D>::from_base(F::Extension::two()));
            let t01 = builder.add_ext_algebra(x[0], x[1]);
            let t23 = builder.add_ext_algebra(x[2], x[3]);
            let t0123 = builder.add_ext_algebra(t01, t23);
            let t01123 = builder.add_ext_algebra(t0123, x[1]);
            let t01233 = builder.add_ext_algebra(t0123, x[3]);
            x[3] = builder.mul_add_ext_algebra(x[0], two, t01233);
            x[1] = builder.mul_add_ext_algebra(x[2], two, t01123);
            x[0] = builder.add_ext_algebra(t01123, t01);
            x[2] = builder.add_ext_algebra(t01233, t23);
            constraints.extend_from_slice(
                &(0..4)
                    .map(|i| vars.get_local_ext_algebra(Self::wires_output(op, i)))
                    .zip(x)
                    .flat_map(|(out, computed_out)| (builder.sub_ext_algebra(out, computed_out)).0)
                    .collect_vec(),
            )
        });
        constraints
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<crate::iop::generator::WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        (0..self.num_ops)
            .map(|op| WitnessGeneratorRef::new(ApplyMat4Generator { row, op }.adapter()))
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 8 * D
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 4 * D
    }
}

#[derive(Clone, Debug, Default)]
pub struct ApplyMat4Generator<const D: usize> {
    row: usize,
    op: usize,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for ApplyMat4Generator<D>
where
    
{
    fn id(&self) -> String {
        "ApplyMat4Generator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..4)
            .flat_map(|i| {
                Target::wires_from_range(self.row, ApplyMat4Gate::<F, D>::wires_input(self.op, i))
            })
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_local_ext_target =
            |wire_range| ExtensionTarget::<D>::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_ext_target(wire_range));

        let inputs: [_; 4] = (0..4)
            .map(|i| get_local_ext(ApplyMat4Gate::<F, D>::wires_input(self.op, i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut x = inputs.clone();
        let t01 = x[0] + x[1];
        let t23 = x[2] + x[3];
        let t0123 = t01 + t23;
        let t01123 = t0123 + x[1];
        let t01233 = t0123 + x[3];
        x[3] = t01233 + x[0] * F::Extension::two();
        x[1] = t01123 + x[2] * F::Extension::two();
        x[0] = t01123 + t01;
        x[2] = t01233 + t23;

        for (i, &out) in x.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_ext_target(ApplyMat4Gate::<F, D>::wires_output(self.op, i)),
                out,
            );
        }
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.op)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let row = src.read_usize()?;
        let op = src.read_usize()?;
        Ok(Self { row, op })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_baby_bear::BabyBear;

    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::BABYBEAR_NUM_HASH_OUT_ELTS;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, Poseidon2BabyBearConfig};

    #[test]
    fn low_degree() {
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config_gl());
        test_low_degree::<BabyBear, _, 4, 8>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 4;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config_gl());
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }
}
